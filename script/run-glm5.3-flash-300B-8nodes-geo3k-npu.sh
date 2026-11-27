#!/usr/bin/env bash
# Ascend950: Geo3K full288, 8 nodes x 8 NPUs, CP1; node21 submits.
# Same NEW run ID on all nodes. Response <=2048 tokens; prompt+response <=4096.
# Recipe layout follows Slime's scripts/run-glm5.3-flash-300B.sh.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
DEPS_ROOT="${GLM53_RELEASE_ENV:-/opt/glm53-release}"
REPO_ROOT="$CODE_ROOT"
SLIME_ROOT="$CODE_ROOT/src/slime-ascend/slime"
CONDA_ENV="$DEPS_ROOT/conda"
CANN_ROOT="$DEPS_ROOT/cann/9.2.0/cann-9.2.0"

# Cluster configuration. Training and rollout use the same 200G interface.
FULL_COMM_IFNAME="${FULL_COMM_IFNAME:-enp39s0f0}"
read -r -a NODE_IPS <<< "${GLM53_NODE_IPS:?Set GLM53_NODE_IPS to the space-separated communication addresses}"
NODE_LOG_START=21
LOG_ROOT="${LOG_ROOT:-$CODE_ROOT/all_logs}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/mnt/weight/GLM-5.3-Flash-BF16}"
PROMPT_DATA="${PROMPT_DATA:-${DEPS_ROOT%/*}/datasets/geo3k_imgurl/train.jsonl}"
NUM_ROLLOUT="${NUM_ROLLOUT:-3}"

# Reuse the model specification rather than maintaining a second architecture.
source "${SLIME_ROOT}/scripts/models/glm5.3-flash-300B.sh"
[[ ${NUM_EXPERTS:-288} == 288 ]] || { echo "This recipe requires 288 experts." >&2; exit 2; }
[[ $NUM_ROLLOUT =~ ^[1-9][0-9]*$ ]] || { echo "NUM_ROLLOUT must be positive." >&2; exit 2; }

# Training actors only: isolate HCCL ports from rollout; keep the exact post7
# TMS compatibility and MoE workspace limits used by the previous eight-node run.
TRAIN_ENV_JSON='{
  "HCCL_HOST_SOCKET_PORT_RANGE":"9200-9299",
  "HCCL_NPU_SOCKET_PORT_RANGE":"9300-9399",
  "GLM53_MOE_DISPATCH_CHUNK_SIZE":"512",
  "GLM53_EXPERT_TOKEN_CHUNK_SIZE":"1024",
  "TASK_QUEUE_ENABLE":"0",
  "GLM53_CONV_BACKEND":"ascendc",
  "GLM53_MHC_BACKEND":"ascendc",
  "GLM53_KPOOL_BACKEND":"ascendc",
  "GLM53_SWIGLU_BACKEND":"ascendc",
  "GLM53_KDA_SEQUENCE_MODE":"packed"
}'
# Initialize from the original HF weights; no optimizer/RNG resume or checkpoint writes.
# --save/--save-hf/--save-interval stay unset, disabling periodic and final saves.
CKPT_ARGS=(
    --hf-checkpoint "$HF_CHECKPOINT" --load "$HF_CHECKPOINT" --megatron-to-hf-mode raw
    --finetune --start-rollout-id 0
    --no-load-optim --no-load-rng --no-save-optim --freeze-params-name-list 'model.visual.*'
)
ROLLOUT_ARGS=(
    --prompt-data "$PROMPT_DATA" --input-key problem --label-key answer --apply-chat-template
    --multimodal-keys '{"image":"images"}'
    --rm-type "${RM_TYPE:-deepscaler}" --seed 522 --rollout-seed 42
    --num-rollout "$NUM_ROLLOUT" --rollout-batch-size 16 --n-samples-per-prompt 8
    --global-batch-size 128 --micro-batch-size 1
    --rollout-max-response-len 2048 --rollout-temperature 1 --rollout-top-p 1
    --rollout-max-prompt-len 2048 --rollout-max-context-len 4096
    --advantage-estimator grpo --pg-loss-type reinforce
    --rollout-data-transport object-store --use-rollout-logprobs
)
PERF_ARGS=(
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 4
    --decoder-first-pipeline-num-layers 12 --decoder-last-pipeline-num-layers 11
    --context-parallel-size 1 --expert-model-parallel-size 16 --expert-tensor-parallel-size 1
    --moe-token-dispatcher-type alltoall
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --seq-length 4096 --max-position-embeddings 65536
    # One full 4096-token sample fits the per-rank packing budget at CP1.
    --use-dynamic-batch-size --max-tokens-per-gpu 4096 --data-pad-size-multiplier 128
    --log-probs-chunk-size 256
)
OPTIMIZER_ARGS=(
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
    --adam-beta1 0.9 --adam-beta2 0.98
    --optimizer-cpu-offload --use-precision-aware-optimizer
    --no-pin-cpu-grads --overlap-cpu-optimizer-d2h-h2d
)
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 16
    --sglang-enable-deterministic-inference --sglang-linear-attn-decode-backend triton
    --sglang-device npu --sglang-attention-backend ascend --sglang-tp-size 16
    --sglang-dp-size 1 --sglang-attn-dp-size 16 --sglang-ep-size 16
    --sglang-enable-dp-lm-head --sglang-moe-dense-tp-size 1
    # Keep each image encoder unsharded, matching the training vision encoder.
    --sglang-mm-enable-dp-encoder --sglang-disable-shared-experts-fusion
    --sglang-weight-loader-prefetch-checkpoints --sglang-weight-loader-prefetch-num-threads 1
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.65}" --sglang-context-length 4096
    # Attention DP16 divides 8192 into one 512-token cache page per rank.
    --sglang-max-running-requests 64 --sglang-chunked-prefill-size 8192
    --sglang-disable-overlap-schedule --sglang-disable-prefill-cuda-graph --sglang-cuda-graph-bs-decode 1 2 4 8 16 24 32 64
    --sglang-page-size 512 --sglang-mamba-track-interval 512
    --sglang-mamba-ssm-dtype float32 --sglang-mamba-radix-cache-strategy extra_buffer
    --sglang-hybrid-cache-checkpoints-level 3
    --sglang-sampling-backend ascend --sglang-boundary-reduction rs+rsv --sglang-moe-a2a-backend none --sglang-kv-cache-dtype bfloat16 --sglang-watchdog-timeout 1200
)
MISC_ARGS=(
    --attention-dropout 0 --hidden-dropout 0 --bf16
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
    --distributed-timeout-minutes 60 --update-weight-buffer-size "${UPDATE_WEIGHT_BUFFER_SIZE:-1073741824}"
)
ARGS=(
    --actor-num-nodes 8 --actor-num-gpus-per-node 8 --colocate
    --rollout-health-check-first-wait 2400
    --train-env-vars "$TRAIN_ENV_JSON"
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}"
    "${PERF_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}"
)

# All numerical adaptations are in ordinary source files in CODE_ROOT.
# R3 and --check-weight-update-equal remain disabled in this recipe.
export GLM53_ZERO_MISMATCH=1
source "$SCRIPT_DIR/launch_common.sh"
glm53_eight_submit "$@"
