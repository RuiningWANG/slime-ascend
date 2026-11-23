#!/usr/bin/env bash
# Four TP2/DP2/EP2 serving engines, with real actor updates and R3 routing replay.
set -euo pipefail

case ${1:-} in
    -h|--help)
        echo "Usage: bash lh_scripts/run_glm53_single_node_ep2.sh [--check|--print-command]"
        echo "Set HF_CHECKPOINT (45-layer BF16 Top32), PROMPT_DATA (Geo3K problem/answer/images),"
        echo "RAY_NODE_IP, DEPS_ROOT, CONDA_ENV, GLM53_BI_DEPS and SGLANG_PYTHONPATH."
        echo "SIM_ROOT sets the archive root. NUM_ROLLOUT defaults to 1000; use 2 for the gate."
        exit 0 ;;
    --check|--print-command|"") ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
: "${HF_CHECKPOINT:?set the Top32 BF16 checkpoint directory}"
: "${PROMPT_DATA:?set the Geo3K JSONL path}"
NUM_ROLLOUT=${NUM_ROLLOUT:-1000}
[[ $NUM_ROLLOUT =~ ^[1-9][0-9]*$ ]] || { echo "Expected a positive NUM_ROLLOUT" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SLIME_ROOT="${SLIME_ROOT:-$SCRIPT_DIR/../slime}"
source "$SLIME_ROOT/scripts/models/glm5.3-flash-300B.sh"
for index in "${!MODEL_ARGS[@]}"; do
    if [[ ${MODEL_ARGS[$index]} == --num-experts ]]; then MODEL_ARGS[$((index+1))]=32; fi
done

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

ARGS=(
    --actor-num-nodes 1 --actor-num-gpus-per-node 8 --colocate
    --train-env-vars "$TRAIN_ENV_JSON"
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "$HF_CHECKPOINT" --load "$HF_CHECKPOINT" --megatron-to-hf-mode raw
    --finetune --start-rollout-id 0
    --no-load-optim --no-load-rng --no-save-optim --freeze-params-name-list 'model.visual.*'
    --prompt-data "$PROMPT_DATA" --input-key problem --label-key answer --apply-chat-template
    --multimodal-keys '{"image":"images"}' --rm-type deepscaler --seed 522
    --num-rollout "$NUM_ROLLOUT" --rollout-batch-size 16 --n-samples-per-prompt 8
    --global-batch-size 128 --micro-batch-size 1
    --rollout-max-response-len 2048 --rollout-temperature 1 --rollout-top-p 1
    --rollout-max-prompt-len 2048 --rollout-max-context-len 4096
    --advantage-estimator grpo --pg-loss-type reinforce
    --rollout-data-transport object-store --use-rollout-routing-replay
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 4
    --decoder-first-pipeline-num-layers 12 --decoder-last-pipeline-num-layers 11
    --context-parallel-size 1 --expert-model-parallel-size 2 --expert-tensor-parallel-size 1
    --moe-token-dispatcher-type alltoall
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --seq-length 4096 --max-position-embeddings 65536
    --use-dynamic-batch-size --max-tokens-per-gpu 4096 --data-pad-size-multiplier 128
    --log-probs-chunk-size 256
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
    --adam-beta1 0.9 --adam-beta2 0.98
    --optimizer-cpu-offload --use-precision-aware-optimizer
    --no-pin-cpu-grads --no-pin-cpu-params --overlap-cpu-optimizer-d2h-h2d
    --rollout-num-gpus-per-engine 2
    --sglang-device npu --sglang-attention-backend ascend --sglang-tp-size 2
    --sglang-dp-size 2 --sglang-ep-size 2 --sglang-enable-dp-attention
    --sglang-enable-dp-lm-head --sglang-moe-dense-tp-size 1
    --sglang-mm-enable-dp-encoder --sglang-disable-shared-experts-fusion
    --sglang-weight-loader-prefetch-checkpoints --sglang-weight-loader-prefetch-num-threads 1
    --sglang-mem-fraction-static 0.65 --sglang-context-length 4096
    # DP2 preserves the original 256 prefill tokens and four requests per DP lane.
    --sglang-max-running-requests 8 --sglang-chunked-prefill-size 512
    --sglang-disable-overlap-schedule --sglang-cuda-graph-bs 1 2 4
    --sglang-page-size 64 --sglang-kv-cache-dtype bfloat16 --sglang-watchdog-timeout 1200
    --attention-dropout 0 --hidden-dropout 0 --bf16
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
    --distributed-timeout-minutes 60 --update-weight-buffer-size 1073741824
    --check-weight-update-equal --sglang-enable-deterministic-inference
    --rollout-health-check-first-wait 2400
)

if [[ ${1:-} == --print-command ]]; then
    printf '%q ' "${CONDA_ENV:+$CONDA_ENV/bin/}python" -u "$SLIME_ROOT/train.py" "${ARGS[@]}"
    printf '\n'
    exit 0
fi

: "${RAY_NODE_IP:?set the Ray communication IP}"
: "${DEPS_ROOT:?set the directory containing the Ascend runtime dependencies}"
: "${CONDA_ENV:?set the Conda environment}"
: "${GLM53_BI_DEPS:?set the BI operator directory containing vendor-env.path and python/}"
: "${SGLANG_PYTHONPATH:?set the aligned SGLang repository python directory}"
[[ -r $PROMPT_DATA ]] || { echo "Cannot read $PROMPT_DATA" >&2; exit 2; }
/usr/bin/python3 -I -S - "$HF_CHECKPOINT" <<'PY_MODEL'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
config = json.loads((path / "config.json").read_text())
text = config.get("text_config", config)
if (text["num_hidden_layers"], text["n_routed_experts"], text["num_experts_per_tok"]) != (45, 32, 8):
    sys.exit("The EP2 simulation expects 45 layers, 32 routed experts and Top8")
if text.get("dtype", text.get("torch_dtype")) != "bfloat16" or any(c.get("quantization_config") for c in (config, text)):
    sys.exit("The EP2 simulation expects an unquantized BF16 checkpoint")
shards = set(json.loads((path / "model.safetensors.index.json").read_text())["weight_map"].values())
if not shards or not all((path / shard).is_file() for shard in shards):
    sys.exit("Incomplete checkpoint shards")
PY_MODEL

export LOG_ROOT="${LOG_ROOT:-${SIM_ROOT:-$PWD/runs/$(date +%Y%m%d)_$(hostname -s)}/all_logs}"
export GLM53_NUM_NPUS=8 GLM53_LOGP_PARITY_ONLY=0 GLM53_TRAIN_ROLLOUT_PARITY=1
export GLM53_R3_DIRECT_WORKERS=1 GLM53_REPLAY_SERVING_MOE_EP2=1 GLM53_STRICT_LOSS_AUDIT=1
export GLM53_NPU_DETERMINISTIC_LEVEL_3=1 GLM53_BI_KDA_MODE=per-sequence GLM53_BI_KDA_DECODE_PER_SEQUENCE=1
export GLM53_KDA_GROUPED_RECURRENCE=1 GLM53_KDA_GROUPED_CONV=1 GLM53_TUNE_KDA_ZERO_STATE=1 GLM53_TUNE_KDA_TILE=64
export GLM53_STRICT_SGLANG_BI=1 GLM53_STRICT_KDA_RADIX=1 GLM53_STRICT_INDEXER=1
export GLM53_STRICT_KPOOL_CACHE=1 GLM53_STRICT_KPOOL_DECODE=1 GLM53_STRICT_DECODE_INDEXER_WEIGHTS=1
export GLM53_STRICT_SPARSE_ATTENTION=1 GLM53_STRICT_FINAL_MEAN=1
# Serving uses dense TP1 here. TP8 projection emulation belongs to the TP8 recipe.
export GLM53_STRICT_KDA_PROJECT=0 GLM53_STRICT_MLA_PROJECT=0 GLM53_STRICT_DENSE_PROJECT=0 GLM53_STRICT_MOE_PROJECT=0
export GLM53_CONV_BACKEND=ascendc GLM53_MHC_BACKEND=ascendc GLM53_KPOOL_BACKEND=ascendc GLM53_SWIGLU_BACKEND=ascendc
export GLM53_KDA_SEQUENCE_MODE=packed TASK_QUEUE_ENABLE=0 ASCEND_USE_FIA=1
export HCCL_OP_EXPANSION_MODE=AICPU_TS HCCL_BUFFSIZE=256 DEEPEP_HCCL_BUFFSIZE=256
export HCCL_EXEC_TIMEOUT=3600 HCCL_CONNECT_TIMEOUT=3600 ACL_DEVICE_SYNC_TIMEOUT=3600
export SGLANG_PLUGINS=smg-servicer SGLANG_HEALTH_CHECK_TIMEOUT=540
source "$SCRIPT_DIR/../runtime/lh-ascend-runtime.sh"
glm53_submit "$@"
