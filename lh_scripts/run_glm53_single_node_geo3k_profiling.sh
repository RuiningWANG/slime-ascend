#!/usr/bin/env bash
# GLM-5.3-Flash, native single-node recipe (MFU-profiling variant).
# Derived from run_glm53_single_node_geo3k.sh. Only changes:
#   1. NUM_ROLLOUT defaults to 4 (short profiling run).
#   2. GLM53_PROFILE_* knobs (Megatron train-step torch_npu.profiler hook),
#      injected via --train-env-vars like the rh_*-train-profiling recipes.
#      Hook module: scripts/lib/glm53/runtime/glm53_train_profiler.py
#      Output: $GLM53_PROFILE_DIR/trainstep<S>-<E>/rank<N>/*_ascend_pt/
#              ASCEND_PROFILER_OUTPUT/{trace_view.json,kernel_details.csv,...}
#   3. Registers --custom-megatron-before-train-step-hook-path when profiling
#      is on (default ON in this variant).
# Usage:
#   bash lh_scripts/run_glm53_single_node_geo3k_profiling.sh            # profile train step 2, rank 0
#   PROFILE_RANKS=0,4 GLM53_PROFILE_STEP_START=2 GLM53_PROFILE_STEP_END=2 \
#       bash lh_scripts/run_glm53_single_node_geo3k_profiling.sh
#   GLM53_PROFILE_TRAIN=0 bash ...   # plain run, no profiling
set -euo pipefail
export TASK_QUEUE_ENABLE=0
# Validated fused operators and packed KDA (override via environment for A/B tests).
export GLM53_KPOOL_BACKEND=${GLM53_KPOOL_BACKEND:-ascendc}
export GLM53_MHC_BACKEND=${GLM53_MHC_BACKEND:-ascendc}
export GLM53_SWIGLU_BACKEND=${GLM53_SWIGLU_BACKEND:-ascendc}
export GLM53_CONV_BACKEND=${GLM53_CONV_BACKEND:-ascendc}
export GLM53_KDA_SEQUENCE_MODE=${GLM53_KDA_SEQUENCE_MODE:-packed}
export GLM53_KDA_FWD_HEAD_CHUNK_SIZE=${GLM53_KDA_FWD_HEAD_CHUNK_SIZE:-64}
export GLM53_KDA_BWD_HEAD_CHUNK_SIZE=${GLM53_KDA_BWD_HEAD_CHUNK_SIZE:-64}
# Keep model offload for colocated online rollout/training.
# CPU packed boundaries and ViT alignment use the model's existing implementation.

# Ray address on this machine's communication NIC; taken from runtime/host.env
# (COMM_IFNAME) unless overridden, so the script works on any rack host.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/../runtime/host.env"
RAY_NODE_IP=${RAY_NODE_IP:-$(ip -o -4 addr show dev "$COMM_IFNAME" | awk '{print $4}' | cut -d/ -f1)}
[[ -n $RAY_NODE_IP ]] || { echo "No IPv4 address on $COMM_IFNAME" >&2; exit 2; }
echo "[glm53-profiling] RAY_NODE_IP=$RAY_NODE_IP (COMM_IFNAME=$COMM_IFNAME)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../runtime/lh-ascend-runtime.sh"

export HCCL_EXEC_TIMEOUT=3600 HCCL_CONNECT_TIMEOUT=3600 ACL_DEVICE_SYNC_TIMEOUT=3600

HF_CHECKPOINT=/mnt/weight/GLM-5.3-moe16_geo3k/glm53_geo3k_sft/iter_0000400_bf16
PROMPT_DATA=/mnt/share/r30009656/dataset/glm53_rl/train.jsonl
EXPERIMENT_NAME=glm53_geo3k
# Short run: profiling needs only ~4 rollouts (1 train step per rollout).
NUM_ROLLOUT=${NUM_ROLLOUT:-4}
RM_TYPE=math
CP_SIZE=1

# Prompt limit includes the chat template. Total context capacity includes prompt and response.
ROLLOUT_MAX_RESPONSE_LEN=2048
ROLLOUT_MAX_PROMPT_LEN=1024
MAX_CONTEXT_LEN=4096

# Model architecture (top16 checkpoint).
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
    --num-experts 16
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

# Native MoE workspace limits. Set both to 0 for the original A/B baseline.
GLM53_MOE_DISPATCH_CHUNK_SIZE=${GLM53_MOE_DISPATCH_CHUNK_SIZE:-0}
GLM53_EXPERT_TOKEN_CHUNK_SIZE=${GLM53_EXPERT_TOKEN_CHUNK_SIZE:-0}

# ---- Training profiling (opt-in, ON by default in this variant) --------------
# Profiles the Megatron train steps (forward/backward/optimizer) with
# torch_npu.profiler via the slime before-train-step hook. Steps are GLOBAL
# training-step indices (0-based, matches "train/step" in slime logs; this
# recipe has 1 train step per rollout). Step 2 is the third train step, safely
# past warmup.
GLM53_PROFILE_TRAIN=${GLM53_PROFILE_TRAIN:-${PROFILE:-1}}
GLM53_PROFILE_DIR=${GLM53_PROFILE_DIR:-/mnt/share/l00804615/glm5.3_flash_native/start/glm53-profiles-geo3k}
GLM53_PROFILE_STEP_START=${GLM53_PROFILE_STEP_START:-${PROFILE_STEP_START:-2}}
GLM53_PROFILE_STEP_END=${GLM53_PROFILE_STEP_END:-${PROFILE_STEP_END:-2}}
GLM53_PROFILE_RANKS=${GLM53_PROFILE_RANKS:-${PROFILE_RANKS:-0}}
GLM53_PROFILE_LEVEL=${GLM53_PROFILE_LEVEL:-${PROFILE_LEVEL:-Level1}}
GLM53_PROFILE_AIC=${GLM53_PROFILE_AIC:-${PROFILE_AIC:-AiCoreNone}}
GLM53_PROFILE_WITH_STACK=${GLM53_PROFILE_WITH_STACK:-${PROFILE_WITH_STACK:-0}}
GLM53_PROFILE_MEMORY=${GLM53_PROFILE_MEMORY:-0}
GLM53_PROFILE_MICROBATCHES=${GLM53_PROFILE_MICROBATCHES:-0}
mkdir -p "$GLM53_PROFILE_DIR"

TRAIN_ENV_JSON=$(/usr/bin/python3 -I -S - "$TRAIN_ENV_JSON" "$GLM53_MOE_DISPATCH_CHUNK_SIZE" "$GLM53_EXPERT_TOKEN_CHUNK_SIZE" \
    "$GLM53_PROFILE_TRAIN" "$GLM53_PROFILE_DIR" \
    "$GLM53_PROFILE_STEP_START" "$GLM53_PROFILE_STEP_END" "$GLM53_PROFILE_RANKS" \
    "$GLM53_PROFILE_LEVEL" "$GLM53_PROFILE_AIC" "$GLM53_PROFILE_WITH_STACK" \
    "$GLM53_PROFILE_MEMORY" "$GLM53_PROFILE_MICROBATCHES" <<'PY_CHUNKS'
import json
import sys
values = json.loads(sys.argv[1])
for key, raw in zip(("GLM53_MOE_DISPATCH_CHUNK_SIZE", "GLM53_EXPERT_TOKEN_CHUNK_SIZE"), sys.argv[2:4]):
    size = int(raw)
    assert size >= 0, key
    values[key] = str(size)
profile_keys = (
    "GLM53_PROFILE_TRAIN", "GLM53_PROFILE_DIR",
    "GLM53_PROFILE_STEP_START", "GLM53_PROFILE_STEP_END", "GLM53_PROFILE_RANKS",
    "GLM53_PROFILE_LEVEL", "GLM53_PROFILE_AIC", "GLM53_PROFILE_WITH_STACK",
    "GLM53_PROFILE_MEMORY", "GLM53_PROFILE_MICROBATCHES",
)
for key, value in zip(profile_keys, sys.argv[4:]):
    values[key] = value
print(json.dumps(values))
PY_CHUNKS
)

# Rollout, optimizer, parallelism and SGLang settings.
ARGS=(
    --actor-num-nodes 1 --actor-num-gpus-per-node 8 --colocate
    --train-env-vars "$TRAIN_ENV_JSON"
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "$HF_CHECKPOINT" --load "$HF_CHECKPOINT" --megatron-to-hf-mode raw
    --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label --apply-chat-template
    --multimodal-keys '{"image":"images"}'
    --rm-type "${RM_TYPE:-deepscaler}" --seed 522
    --num-rollout "${NUM_ROLLOUT:-4}" --rollout-batch-size 8 --n-samples-per-prompt 8
    --global-batch-size 64 --micro-batch-size 1
    --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN" --rollout-temperature 1 --rollout-top-p 1
    --rollout-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN" --rollout-max-context-len "$MAX_CONTEXT_LEN"
    --advantage-estimator grpo --pg-loss-type reinforce
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 4
    --decoder-first-pipeline-num-layers 12 --decoder-last-pipeline-num-layers 11
    --context-parallel-size "${CP_SIZE:-1}" --expert-model-parallel-size 2 --expert-tensor-parallel-size 1
    --allgather-cp --moe-token-dispatcher-type alltoall
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --seq-length "$MAX_CONTEXT_LEN" --max-position-embeddings "$MAX_CONTEXT_LEN"
    # Packing budget only: an oversized individual sample is scheduled alone.
    --use-dynamic-batch-size --max-tokens-per-gpu 2048 --data-pad-size-multiplier 1024
    --log-probs-chunk-size 1024
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
    --no-load-optim --no-load-rng --no-save-optim --freeze-params-name-list 'model.visual.*'
    --optimizer-cpu-offload --use-precision-aware-optimizer
    --overlap-cpu-optimizer-d2h-h2d
    --rollout-num-gpus-per-engine 8
    --sglang-device npu --sglang-attention-backend ascend --sglang-tp-size 8
    --sglang-dp-size 1 --sglang-ep-size 1
    # Keep each image encoder unsharded to match training ViT numerics.
    --sglang-mm-enable-dp-encoder
    --sglang-mem-fraction-static 0.55 --sglang-context-length "$MAX_CONTEXT_LEN"
    --sglang-max-running-requests 32 --sglang-chunked-prefill-size 1024
    --sglang-disable-overlap-schedule --sglang-cuda-graph-bs 1 2 4 8 16 24 32
    --sglang-page-size 64 --sglang-kv-cache-dtype bfloat16 --sglang-watchdog-timeout 1200
    --attention-dropout 0 --hidden-dropout 0 --bf16
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
    --distributed-timeout-minutes 20 --update-weight-buffer-size 268435456
    --rollout-data-transport object-store
    --check-weight-update-equal
    --use-rollout-routing-replay
)

# Optional fixed-input numerical diagnostics.
if [[ -n ${SAVE_DEBUG_ROLLOUT_DATA:-} ]]; then
    # Durable rollout capture enables recovery; keep optimizer/RNG checkpoint state.
    diagnostic_args=()
    for arg in "${ARGS[@]}"; do
        [[ "$arg" == --no-save-optim || "$arg" == --no-save-rng ]] || diagnostic_args+=("$arg")
    done
    ARGS=("${diagnostic_args[@]}" --ckpt-format torch_dist --save-debug-rollout-data "$SAVE_DEBUG_ROLLOUT_DATA")
    unset diagnostic_args
fi
[[ -z ${SAVE_DEBUG_TRAIN_DATA:-} ]] || ARGS+=(--save-debug-train-data "$SAVE_DEBUG_TRAIN_DATA")

# Register the Megatron train-step profiling hook (must come after ARGS is built).
if [[ "$GLM53_PROFILE_TRAIN" == "1" ]]; then
    ARGS+=(--custom-megatron-before-train-step-hook-path glm53_train_profiler.before_train_step)
fi

glm53_submit "$@"
