#!/usr/bin/env bash
# GLM-5.3-Flash, native single-node recipe. Edit the parameters below.
# --check performs static validation; a normal launch retains exclusive resource gates.
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

# Ray address on this machine's communication NIC; change when moving hosts.
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo "Usage: bash lh_scripts/run_glm53_train_rollout_parity.sh [--check]"
    echo "Set RAY_NODE_IP, DEPS_ROOT, CONDA_ENV, HF_CHECKPOINT, PROMPT_DATA and GLM53_BI_DEPS."
    echo "Set SGLANG_PYTHONPATH to the python directory of the matching SGLang branch."
    echo "For the aligned TP8 performance configuration, set GLM53_TUNE_ORDERED_SUM=1,"
    echo "GLM53_TUNE_KDA_ZERO_STATE=1, GLM53_TUNE_KDA_TILE=64 and GLM53_TUNE_MOE_UP=1."
    exit 0
fi
: "${RAY_NODE_IP:?set the Ray communication IP}"
: "${DEPS_ROOT:?set the directory containing the Ascend runtime dependencies}"
: "${CONDA_ENV:?set the path to the Conda environment}"
: "${HF_CHECKPOINT:?set the GLM-5.3 BF16 checkpoint directory}"
: "${PROMPT_DATA:?set the rollout prompt dataset}"
: "${GLM53_BI_DEPS:?set the directory containing python/ and vendor-env.path}"
: "${SGLANG_PYTHONPATH:?set the matching SGLang branch python directory}"
export GLM53_LOGP_PARITY_ONLY=1
GLM53_ZERO_MISMATCH_PROFILE=${GLM53_ZERO_MISMATCH_PROFILE:-tp8}
GLM53_ZERO_MISMATCH_ARM=${GLM53_ZERO_MISMATCH_ARM:-aligned}
case "$GLM53_ZERO_MISMATCH_ARM" in
    aligned|native) ;;
    *) echo "Expected GLM53_ZERO_MISMATCH_ARM=aligned or native" >&2; exit 2 ;;
esac
if [[ $GLM53_ZERO_MISMATCH_ARM == aligned ]]; then
    export GLM53_TRAIN_ROLLOUT_PARITY=1
    export GLM53_BI_KDA_MODE=per-sequence GLM53_BI_KDA_DECODE_PER_SEQUENCE=1
else
    export GLM53_TRAIN_ROLLOUT_PARITY=0
    export GLM53_BI_KDA_MODE=native GLM53_BI_KDA_DECODE_PER_SEQUENCE=0
fi
export GLM53_NPU_DETERMINISTIC_LEVEL_3=1
export GLM53_NUM_NPUS=8
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NUM_ROLLOUT=${NUM_ROLLOUT:-2}
export ASCEND_USE_FIA=1
export HCCL_OP_EXPANSION_MODE=AICPU_TS
export HCCL_BUFFSIZE=256 DEEPEP_HCCL_BUFFSIZE=256

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SLIME_ROOT="$(cd -- "$SCRIPT_DIR/../slime" && pwd)"
source "${SCRIPT_DIR}/../runtime/lh-ascend-runtime.sh"

export HCCL_EXEC_TIMEOUT=3600 HCCL_CONNECT_TIMEOUT=3600 ACL_DEVICE_SYNC_TIMEOUT=3600

EXPERIMENT_NAME=glm53_geo3k
RM_TYPE=${RM_TYPE:-math}
CP_SIZE=1

# Prompt limit includes the chat template. Total context capacity includes prompt and response.
ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-1024}
case "$GLM53_ZERO_MISMATCH_PROFILE" in
    tp8)
        ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-2048}
        MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-4096}
        BATCH_ARGS=(--rollout-batch-size 8 --n-samples-per-prompt 8 --global-batch-size 64
                    --data-pad-size-multiplier 1024 --overlap-cpu-optimizer-d2h-h2d)
        SERVING_ARGS=(--rollout-num-gpus-per-engine 8 --sglang-tp-size 8
                      --sglang-mem-fraction-static 0.55 --sglang-max-running-requests 32
                      --sglang-cuda-graph-bs 1 2 4 8 16 24 32)
        for key in SGLANG_BI KDA_RADIX INDEXER SPARSE_ATTENTION KDA_PROJECT MLA_PROJECT \
                   DENSE_PROJECT MOE_PROJECT FINAL_MEAN KPOOL_CACHE KPOOL_DECODE DECODE_INDEXER_WEIGHTS; do
            if [[ $GLM53_ZERO_MISMATCH_ARM == aligned ]]; then
                export "GLM53_STRICT_${key}=1"
            else
                export "GLM53_STRICT_${key}=0"
            fi
        done
        if [[ $GLM53_ZERO_MISMATCH_ARM == aligned ]]; then
            group_kda=1
            if [[ ${GLM53_GEO3K_EVAL:-0} == 1 ]]; then
                group_kda=0
            fi
            export GLM53_KDA_GROUPED_RECURRENCE=${GLM53_KDA_GROUPED_RECURRENCE:-$group_kda}
            export GLM53_KDA_GROUPED_CONV=${GLM53_KDA_GROUPED_CONV:-$group_kda}
        fi
        ;;
    tp1)
        ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-64}
        MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-2048}
        BATCH_ARGS=(--rollout-batch-size 2 --n-samples-per-prompt 2 --global-batch-size 4
                    --data-pad-size-multiplier 64 --no-pin-cpu-grads --no-pin-cpu-params
                    --use-rollout-logprobs)
        SERVING_ARGS=(--rollout-num-gpus 1 --rollout-num-gpus-per-engine 1 --sglang-tp-size 1
                      --sglang-mem-fraction-static 0.82 --sglang-max-running-requests 8
                      --sglang-enable-deterministic-inference --sglang-disable-radix-cache
                      --sglang-mamba-ssm-dtype float32 --sglang-disable-cuda-graph)
        ;;
    *) echo "Expected GLM53_ZERO_MISMATCH_PROFILE=tp8 or tp1" >&2; exit 2 ;;
esac

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
TRAIN_ENV_JSON=$(/usr/bin/python3 -I -S - "$TRAIN_ENV_JSON" "$GLM53_MOE_DISPATCH_CHUNK_SIZE" "$GLM53_EXPERT_TOKEN_CHUNK_SIZE" <<'PY_CHUNKS'
import json
import sys
values = json.loads(sys.argv[1])
for key, raw in zip(("GLM53_MOE_DISPATCH_CHUNK_SIZE", "GLM53_EXPERT_TOKEN_CHUNK_SIZE"), sys.argv[2:]):
    size = int(raw)
    assert size >= 0, key
    values[key] = str(size)
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
    --num-rollout "${NUM_ROLLOUT:-2}" --micro-batch-size 1
    "${BATCH_ARGS[@]}"
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
    --use-dynamic-batch-size --max-tokens-per-gpu 2048
    --log-probs-chunk-size 1024
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
    --no-load-optim --no-load-rng --no-save-optim --freeze-params-name-list 'model.visual.*'
    --optimizer-cpu-offload --use-precision-aware-optimizer
    "${SERVING_ARGS[@]}"
    --sglang-device npu --sglang-attention-backend ascend
    --sglang-dp-size 1 --sglang-ep-size 1
    # Keep each image encoder unsharded to match training ViT numerics.
    --sglang-mm-enable-dp-encoder
    --sglang-context-length "$MAX_CONTEXT_LEN"
    --sglang-chunked-prefill-size 1024 --sglang-disable-overlap-schedule
    --sglang-page-size 64 --sglang-kv-cache-dtype bfloat16 --sglang-watchdog-timeout 1200
    --attention-dropout 0 --hidden-dropout 0 --bf16
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
    --distributed-timeout-minutes 20 --update-weight-buffer-size 268435456
    --rollout-data-transport object-store
    --check-weight-update-equal
    --use-rollout-routing-replay
)

# Run a standalone Geo3K scoring pass. Use a distinct LOG_ROOT for each arm.
if [[ ${GLM53_GEO3K_EVAL:-0} == 1 ]]; then
    if [[ $NUM_ROLLOUT != 0 || $GLM53_ZERO_MISMATCH_PROFILE != tp8 ]]; then
        echo "Geo3K scoring requires NUM_ROLLOUT=0 and the tp8 profile" >&2
        exit 2
    fi
    : "${GLM53_PHASE_DIR:?set the directory for the Geo3K evaluation receipts}"
    : "${GLM53_CHECKPOINT_FINGERPRINT:?set the checkpoint content manifest path}"
    : "${GLM53_EVAL_CACHE_ENDPOINT:?set the SGLang worker HTTP origin for prefix-cache flushes}"
    export GLM53_PHASE_DIR GLM53_CHECKPOINT_FINGERPRINT GLM53_EVAL_CACHE_ENDPOINT
    if [[ $GLM53_ZERO_MISMATCH_ARM == native ]]; then
        export GLM53_EVAL_ROUNDS=${GLM53_EVAL_ROUNDS:-3}
    else
        export GLM53_EVAL_ROUNDS=${GLM53_EVAL_ROUNDS:-1}
    fi
    ARGS+=(--eval-interval 1 --eval-prompt-data geo3k "$PROMPT_DATA"
          --n-samples-per-eval-prompt 8 --eval-input-key prompt --eval-label-key label
          --eval-temperature 1 --eval-top-p 1
          --eval-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN"
          --eval-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN"
          --eval-max-context-len "$MAX_CONTEXT_LEN"
          --eval-function-path slime.rollout.glm53_geo3k_eval.generate_three_rounds
          --custom-eval-rollout-log-function-path slime.rollout.glm53_geo3k_eval.save)
fi

# Optional fixed-input numerical diagnostics.
if [[ ${GLM53_GEO3K_REPLAY:-0} == 1 ]]; then
    ARGS+=(--rollout-function-path slime.rollout.glm53_rollout_replay.generate)
fi
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

glm53_submit "$@"
