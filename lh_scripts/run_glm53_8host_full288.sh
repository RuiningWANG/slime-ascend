#!/usr/bin/env bash
# Eight physical hosts, eight containers, 128 Ascend dies; full 45L/288E GLM.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_ROOT="$(dirname "$REPO_ROOT")"
CONDA_ENV=/mnt/share/l00937981/conda/envs/glm53-flash-slime
SLIME_ROOT="$REPO_ROOT/slime"
FULL128_HELPER="$REPO_ROOT/runtime/glm53-full128.py"
FULL128_TOPOLOGY=${FULL128_TOPOLOGY:-$REPO_ROOT/runtime/glm53-full128-topology.json}
LOG_ROOT=${LOG_ROOT:-$DEPS_ROOT/cache/work/glm53-full128}
HF_CHECKPOINT=${HF_CHECKPOINT:-/mnt/share/l00804615/weight/GLM-5.3-Flash-BF16}
PROMPT_DATA=${PROMPT_DATA:-/mnt/share/l00804615/dataset/dapo-original-65877096/dapo-math-17k.slime.parquet}
PROFILE=${PROFILE:-smoke}
case "$PROFILE" in
    smoke) CONTEXT_LENGTH=${CONTEXT_LENGTH:-16384}; RESPONSE_LENGTH=${RESPONSE_LENGTH:-8192}
           ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-2}; NUM_ROLLOUT=${NUM_ROLLOUT:-1} ;;
    long)  CONTEXT_LENGTH=${CONTEXT_LENGTH:-65536}; RESPONSE_LENGTH=${RESPONSE_LENGTH:-32768}
           ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}; NUM_ROLLOUT=${NUM_ROLLOUT:-1000} ;;
    *) echo 'PROFILE must be smoke or long' >&2; exit 2 ;;
esac
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE*N_SAMPLES_PER_PROMPT))}
EP_SIZE=${EP_SIZE:-32}
CP_SIZE=${CP_SIZE:-8}
RUN_ID=inspection
for ((i=1; i<=$#; i++)); do
    if [[ ${!i} == --run-id ]]; then j=$((i+1)); RUN_ID=${!j:-}; fi
done
MODEL_ARGS=(
    --spec slime_plugins.models.glm5_next.model get_glm5_next_model_provider
    --model-name glm5_next
    --num-layers 45
    --hidden-size 4096
    --ffn-hidden-size 12288
    --num-attention-heads 64
    --num-query-groups 64
    --kv-channels 256
    --q-lora-rank 1536
    --kv-lora-rank 512
    --vocab-size 154880
    --make-vocab-size-divisible-by 64
    --disable-bias-linear
    --swiglu
    --untie-embeddings-and-output-weights
    --position-embedding-type none
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --rotary-base 800000
    --num-experts 288
    --moe-layer-freq '[0]*3+[1]*42'
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-topk 8
    --moe-router-score-function sigmoid
    --moe-router-pre-softmax
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk-scaling-factor 2.5
    --moe-aux-loss-coeff 0
    --moe-router-dtype fp32
    --moe-grouped-gemm
    --moe-permute-fusion
    --freeze-indexer
    --enable-experimental
)

TRAIN_ENV_JSON='{"HCCL_HOST_SOCKET_PORT_RANGE":"62000-62099","HCCL_NPU_SOCKET_PORT_RANGE":"63000-63099","GLM53_TMS_POST7_RELEASE_COMPAT":"1","GLM53_MOE_DISPATCH_CHUNK_SIZE":"512","GLM53_EXPERT_TOKEN_CHUNK_SIZE":"1024"}'
# Eight uniform containers, each exposing all 16 dies.
ARGS=(
    --rollout-session-id "glm53-full128-$RUN_ID"
    --actor-num-nodes 8 --actor-num-gpus-per-node 16 --num-gpus-per-node 16
    --rollout-num-gpus 128 --colocate --train-env-vars "$TRAIN_ENV_JSON"
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "$HF_CHECKPOINT" --load "$HF_CHECKPOINT" --megatron-to-hf-mode raw
    --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label --apply-chat-template
    --rm-type dapo --reward-key score --seed 522
    --num-rollout "$NUM_ROLLOUT" --num-steps-per-rollout 1
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE" --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
    --global-batch-size "$GLOBAL_BATCH_SIZE" --micro-batch-size 1
    --rollout-max-response-len "$RESPONSE_LENGTH" --rollout-max-context-len "$CONTEXT_LENGTH"
    --rollout-temperature 1 --rollout-top-p 1
    --advantage-estimator grpo --pg-loss-type reinforce
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 4
    --decoder-first-pipeline-num-layers 12 --decoder-last-pipeline-num-layers 11
    --context-parallel-size "$CP_SIZE" --expert-model-parallel-size "$EP_SIZE"
    --expert-tensor-parallel-size 1 --allgather-cp --moe-token-dispatcher-type alltoall
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --seq-length "$CONTEXT_LENGTH" --max-position-embeddings "$CONTEXT_LENGTH"
    --use-dynamic-batch-size --max-tokens-per-gpu 2048 --data-pad-size-multiple 128
    --log-probs-chunk-size 256
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
    --adam-beta1 0.9 --adam-beta2 0.98
    --no-load-optim --no-load-rng --no-save-optim --freeze-params-name-list 'model.visual.*'
    --optimizer-cpu-offload --optimizer-offload-fraction 0.8 --use-precision-aware-optimizer
    --overlap-cpu-optimizer-d2h-h2d
    --rollout-num-gpus-per-engine 64
    --sglang-language-only --sglang-device npu --sglang-attention-backend ascend
    --sglang-tp-size 64 --sglang-dp-size 1 --sglang-ep-size 1
    --sglang-mem-fraction-static 0.75 --sglang-context-length "$CONTEXT_LENGTH"
    --sglang-max-running-requests 128 --sglang-chunked-prefill-size 4096
    --sglang-disable-overlap-schedule --sglang-cuda-graph-max-bs-decode 128
    --sglang-page-size 64 --sglang-kv-cache-dtype bfloat16 --sglang-watchdog-timeout 3600
    --attention-dropout 0 --hidden-dropout 0 --bf16
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
    --distributed-timeout-minutes 30 --update-weight-buffer-size 1073741824
    --rollout-data-transport object-store
    --use-rollout-routing-replay --routing-replay-prefetch-microbatches 1
    --use-tis --tis-clip-low 0.5 --tis-clip 2.0
)
source "$REPO_ROOT/runtime/lh-full128-launch.sh"
glm53_full128_submit "$@"
