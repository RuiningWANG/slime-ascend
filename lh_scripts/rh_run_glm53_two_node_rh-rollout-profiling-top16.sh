#!/usr/bin/env bash
# GLM-5.3-Flash top16, single-node rollout-only profiling (8 NPUs).
# --debug-rollout-only skips the Megatron train actor.
# One SGLang engine: TP=8 / DP=1 / EP=1; NUM_ROLLOUT and prompts unchanged.
# Run this script only on NODE_IPS[0], with a new --run-id.
# The existing launcher supports a one-entry NODE_IPS array and retains its
# exclusive resource gate. The historical filename is kept for compatibility.
#
# Inference profiling (opt-in, disabled by default):
#   GLM53_ROLLOUT_PROFILE=1 bash <this-script> --run-id <new run id>
# The RolloutManager starts SGLang's /start_profile on every engine before
# rollout GLM53_ROLLOUT_PROFILE_ROLLOUT_IDS and stops it after the rollout
# (or automatically after GLM53_ROLLOUT_PROFILE_NUM_STEPS scheduler forward
# steps per stage). Traces land under
#   $GLM53_ROLLOUT_PROFILE_DIR/<host>_<pid>_<ts>_ascend_pt/ASCEND_PROFILER_OUTPUT/
# on the shared filesystem; inspect them with msinsight or read
# op_statistic.csv / kernel_details.csv / trace_view.json directly.
set -euo pipefail

# Fusion backends must be exported so each node's raylet (and every worker
# actor) inherits them.
export GLM53_MHC_BACKEND=ascendc
export GLM53_CONV_BACKEND=ascendc
export GLM53_KPOOL_BACKEND=${GLM53_KPOOL_BACKEND:-ascendc}
export GLM53_SWIGLU_BACKEND=torch

export GLM53_KDA_SEQUENCE_MODE=packed
export GLM53_KDA_FWD_HEAD_CHUNK_SIZE=64
export GLM53_KDA_BWD_HEAD_CHUNK_SIZE=64

NUM_ROLLOUT=${NUM_ROLLOUT:-10}
ROLLOUT_ONLY=${ROLLOUT_ONLY:-1}
if [[ "$ROLLOUT_ONLY" != "1" ]]; then
    echo "This recipe is inference-only: ROLLOUT_ONLY must stay 1" >&2
    exit 2
fi
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA:-}" ]]; then
    echo "LOAD_DEBUG_ROLLOUT_DATA bypasses inference; this rollout-only recipe does not accept it" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEPS_ROOT="${DEPS_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}"
SLIME_ROOT="${DEPS_ROOT}/slime-ascend/slime"
CONDA_ENV="${CONDA_ENV:-${DEPS_ROOT%/*}/conda/envs/lhw_glm_53_flash_native}"
CANN_ROOT="${CANN_ROOT:-${DEPS_ROOT%/*}/cann/9.2.0/cann-9.2.0}"

# Single-node configuration; NODE_IPS may override the local communication IP.
FULL_COMM_IFNAME=${FULL_COMM_IFNAME:-enp39s0f0}
read -r -a NODE_IPS <<< "${NODE_IPS:-197.13.23.1}"
[[ ${#NODE_IPS[@]} == 1 ]] || { echo "This recipe requires exactly one NODE_IPS entry" >&2; exit 2; }

LOG_ROOT=/mnt/share/l00804615/glm5.3_flash_native/start/logs/

# SFT checkpoint: 16 routed experts.
HF_CHECKPOINT=${HF_CHECKPOINT:-/mnt/weight/GLM-5.3-moe16_geo3k/glm53_geo3k_sft/iter_0000400_bf16}
# Fixed-length rollout prompt set: sample 0 of
# 20261030_214729_rh24_rollout_1087987/rollout-0.pt (source_index 0 of
# glm53_dapo_sft_rl.jsonl) copied 200x, built 2026-11-12. The prompt is an
# OpenAI message list; the GLM template must be applied with
# --apply-chat-template.
PROMPT_DATA=${PROMPT_DATA:-/mnt/share/l00804615/glm5.3_flash_native/cache/work/20261112_perf_4k_200/dapo_4k_first_prompt_x200.jsonl}

# Save each rollout for later --load-debug-rollout-data training replay.
# A separate directory per launch avoids overwriting previous runs.
ROLLOUT_SAVE_DIR=${ROLLOUT_SAVE_DIR:-/mnt/share/l00804615/glm5.3_flash_native/cache/work/$(date +%Y%m%d_%H%M%S)_rh2n_rollout_$$}
SAVE_DEBUG_ROLLOUT_DATA="${ROLLOUT_SAVE_DIR}/rollout-{rollout_id}.pt"
printf 'Rollout replay files: %s\n' "$SAVE_DEBUG_ROLLOUT_DATA"
RM_TYPE=deepscaler
CP_SIZE=1
EP_SIZE=${EP_SIZE:-1}

# Fixed at 4k: every rollout is forced to generate exactly 4096 response
# tokens. --rollout-ignore-eos makes SGLang ignore EOS and run to
# max_new_tokens, so the length does not depend on where EOS would land.
ROLLOUT_MAX_RESPONSE_LEN=4096
# Prompt limit includes the chat template (the templated prompt is 161 tokens).
ROLLOUT_MAX_PROMPT_LEN=1024
MAX_CONTEXT_LEN=16384

# SGLang memory knobs. The single TP8 engine runs on one node, so every rank
# uses the same static fraction. Raise it if the KV cache is too small; lower
# it if activation/workspace allocation OOMs.
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.88}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-32}

# ---- Inference (SGLang) profiling (opt-in) -----------------------------------
# GLM53_ROLLOUT_PROFILE=1 turns it on. The SGLang /start_profile endpoint is
# driven by slime's RolloutManager (--rollout-profile-* arguments) for the
# 0-based rollout ids below; with BY_STAGE=1 prefill and decode are captured
# as separate traces, NUM_STEPS applying to each stage.
# The profile directory is derived from --run-id for this run.
profile_run_tag=""
prev_arg=""
for arg in "$@"; do
    if [[ "$prev_arg" == "--run-id" ]]; then
        profile_run_tag=$arg
        break
    fi
    prev_arg=$arg
done
profile_run_tag=${profile_run_tag:-$(date +%Y%m%d_%H%M%S)}
GLM53_ROLLOUT_PROFILE=${GLM53_ROLLOUT_PROFILE:-1}
GLM53_ROLLOUT_PROFILE_DIR=${GLM53_ROLLOUT_PROFILE_DIR:-/mnt/share/l00804615/glm5.3_flash_native/profiler_lh/glm53-rollout-profiles/${profile_run_tag}}
GLM53_ROLLOUT_PROFILE_ROLLOUT_IDS=${GLM53_ROLLOUT_PROFILE_ROLLOUT_IDS:-2}
GLM53_ROLLOUT_PROFILE_NUM_STEPS=${GLM53_ROLLOUT_PROFILE_NUM_STEPS:-8}
GLM53_ROLLOUT_PROFILE_ACTIVITIES=${GLM53_ROLLOUT_PROFILE_ACTIVITIES:-CPU,GPU}
GLM53_ROLLOUT_PROFILE_BY_STAGE=${GLM53_ROLLOUT_PROFILE_BY_STAGE:-1}
GLM53_ROLLOUT_PROFILE_WITH_STACK=${GLM53_ROLLOUT_PROFILE_WITH_STACK:-0}
GLM53_ROLLOUT_PROFILE_RECORD_SHAPES=${GLM53_ROLLOUT_PROFILE_RECORD_SHAPES:-1}

if [[ "$GLM53_ROLLOUT_PROFILE" == "1" ]]; then
    mkdir -p "$GLM53_ROLLOUT_PROFILE_DIR"
    # Default output dir if a manual /start_profile omits output_dir; the
    # explicit per-request output_dir below is what the trace actually uses.
    export SGLANG_TORCH_PROFILER_DIR="$GLM53_ROLLOUT_PROFILE_DIR"
    printf 'SGLang rollout profiles: %s (rollout ids %s, %s steps/stage, by_stage=%s)\n' \
        "$GLM53_ROLLOUT_PROFILE_DIR" "$GLM53_ROLLOUT_PROFILE_ROLLOUT_IDS" \
        "$GLM53_ROLLOUT_PROFILE_NUM_STEPS" "$GLM53_ROLLOUT_PROFILE_BY_STAGE"
fi

# Model architecture (16 routed experts, topk 8).
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
    --moe-shared-expert-overlap
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

# Training-actor-only environment; the actor is never instantiated with
# --debug-rollout-only, but the launcher prints this block and keeps the
# recipe consistent with the two-node training runs.
TRAIN_ENV_JSON_BASE='{"HCCL_HOST_SOCKET_PORT_RANGE":"9200-9299","HCCL_NPU_SOCKET_PORT_RANGE":"9300-9399","HCCL_OP_EXPANSION_MODE":"AICPU_TS","GLM53_TMS_POST7_RELEASE_COMPAT":"1","TASK_QUEUE_ENABLE":"0"}'
TRAIN_ENV_JSON=$(/usr/bin/python3 -I -S - \
    "$TRAIN_ENV_JSON_BASE" \
    "$GLM53_MOE_DISPATCH_CHUNK_SIZE" "$GLM53_EXPERT_TOKEN_CHUNK_SIZE" \
    <<'PY_CHUNKS'
import json
import sys
values = json.loads(sys.argv[1])
for key, raw in zip(("GLM53_MOE_DISPATCH_CHUNK_SIZE", "GLM53_EXPERT_TOKEN_CHUNK_SIZE"), sys.argv[2:4]):
    size = int(raw)
    assert size >= 0, key
    values[key] = str(size)
print(json.dumps(values))
PY_CHUNKS
)

# Rollout, optimizer, parallelism and SGLang settings. The training-side
# optimizer/parallelism values are kept consistent with the source recipe but
# are never used: --debug-rollout-only returns from the actor init immediately.
ARGS=(
    --actor-num-nodes 1 --actor-num-gpus-per-node 8 --num-gpus-per-node 8 --colocate
    # Rollout-only run: the actor is never built, so keep the engines resident
    # (--colocate would otherwise enable the SGLang memory-saver offload).
    --no-offload-train --no-offload-rollout
    --train-env-vars "$TRAIN_ENV_JSON"
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "$HF_CHECKPOINT" --load "$HF_CHECKPOINT" --megatron-to-hf-mode raw
    # PROMPT_DATA is an OpenAI message list; render it with the GLM chat template.
    --apply-chat-template
    --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label
    --rm-type "${RM_TYPE:-deepscaler}" --seed 522
    # 200 identical prompts x 1 sample = 200 fixed-length 4k generations per
    # rollout, served by the single TP8 engine.
    --num-rollout "${NUM_ROLLOUT:-2}" --rollout-batch-size 200 --n-samples-per-prompt 1
    --rollout-ignore-eos
    --global-batch-size 64 --micro-batch-size 1
    --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN" --rollout-temperature 1 --rollout-top-p 1
    --rollout-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN" --rollout-max-context-len "$MAX_CONTEXT_LEN"
    --advantage-estimator grpo --pg-loss-type reinforce
    # World 8 / (TP 8 x PP 1 x CP 1) -> DP 1. PP=1 means all 45 layers on
    # one rank; the uneven-pipeline --decoder-first/last-pipeline-num-layers
    # knobs must NOT be set here.
    --tensor-model-parallel-size 8 --pipeline-model-parallel-size 1
    --context-parallel-size "${CP_SIZE:-1}" --expert-model-parallel-size "${EP_SIZE}" --expert-tensor-parallel-size 1
    --allgather-cp --moe-token-dispatcher-type alltoall
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --seq-length "$MAX_CONTEXT_LEN" --max-position-embeddings "$MAX_CONTEXT_LEN"
    # Packing budget only: an oversized individual sample is scheduled alone.
    --use-dynamic-batch-size --max-tokens-per-gpu 4096 --data-pad-size-multiplier 1024
    --log-probs-chunk-size 1024
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
    --no-load-optim --no-load-rng --no-save-optim --freeze-params-name-list 'model.visual.*'
    --optimizer-cpu-offload --use-precision-aware-optimizer
    --overlap-cpu-optimizer-d2h-h2d
    # DDP grad-reduce overlap only. --overlap-param-gather is deliberately NOT
    # enabled: slime calls DDP.disable_forward_pre_hook() at the start of every
    # rollout, and the second call raises KeyError on the already-removed handles.
    --overlap-grad-reduce
    # One TP8 rollout engine on the local 8 NPUs.
    --rollout-num-gpus-per-engine 8
    --sglang-device npu --sglang-attention-backend ascend --sglang-tp-size 8
    --sglang-dp-size 1 --sglang-ep-size 1
    # Keep the original static memory fraction.
    --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" --sglang-context-length "$MAX_CONTEXT_LEN"
    --sglang-max-running-requests "$SGLANG_MAX_RUNNING_REQUESTS" --sglang-chunked-prefill-size 1024
    --sglang-disable-overlap-schedule --sglang-cuda-graph-bs 1 2 4 8 16 24 32
    --sglang-page-size 64 --sglang-kv-cache-dtype bfloat16 --sglang-watchdog-timeout 1200
    --attention-dropout 0 --hidden-dropout 0 --bf16
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
    # Multi-node rendezvous budget, as in the two-node training recipe.
    --distributed-timeout-minutes 60 --update-weight-buffer-size 268435456
    --rollout-data-transport object-store
    # --save-debug-rollout-data "$SAVE_DEBUG_ROLLOUT_DATA"
    --use-rollout-routing-replay
    # Inference only: never build the Megatron train actor.
    --debug-rollout-only
)

# Opt-in SGLang rollout profiling. These arguments are only added when
# GLM53_ROLLOUT_PROFILE=1, so normal rollout runs behave exactly as before.
if [[ "$GLM53_ROLLOUT_PROFILE" == "1" ]]; then
    ARGS+=(
        --rollout-profile-dir "$GLM53_ROLLOUT_PROFILE_DIR"
        --rollout-profile-rollout-ids "$GLM53_ROLLOUT_PROFILE_ROLLOUT_IDS"
        --rollout-profile-num-steps "$GLM53_ROLLOUT_PROFILE_NUM_STEPS"
        --rollout-profile-activities "$GLM53_ROLLOUT_PROFILE_ACTIVITIES"
    )
    [[ "$GLM53_ROLLOUT_PROFILE_BY_STAGE" == "1" ]] && ARGS+=(--rollout-profile-by-stage)
    [[ "$GLM53_ROLLOUT_PROFILE_WITH_STACK" == "1" ]] && ARGS+=(--rollout-profile-with-stack)
    [[ "$GLM53_ROLLOUT_PROFILE_RECORD_SHAPES" == "1" ]] && ARGS+=(--rollout-profile-record-shapes)
fi

# Operations only: argument parsing, checks, Ray lifecycle and environment setup.
source "${REPO_ROOT}/runtime/lh-two-node-launch.sh"
glm53_two_submit "$@"
