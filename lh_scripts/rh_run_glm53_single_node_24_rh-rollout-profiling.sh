#!/usr/bin/env bash
# GLM-5.3-Flash **full-size** (288 routed experts) rollout-only profiling recipe.
#
# Inference only: --debug-rollout-only skips the Megatron train actor entirely
# (MegatronTrainRayActor.init returns immediately), so the NPUs are used by the
# SGLang engine alone. Runs NUM_ROLLOUT rollout steps and exits.
#
# Parallel: SGLang TP=8 / EP=1 / DP=1 -> one engine spanning all 8 NPUs.
# The actor's TP/PP/EP below are kept consistent but are never instantiated.
#
# Fixed-length workload: the prompt set is one sample repeated 200x and the
# rollout runs with --rollout-max-response-len 4096 + --rollout-ignore-eos, so
# every generation is exactly 4096 tokens regardless of where EOS would land.
set -euo pipefail

# Fusion backends must be exported so the Ray runtime env (and actor) inherits them.
export GLM53_MHC_BACKEND=ascendc
export GLM53_CONV_BACKEND=ascendc
export GLM53_KPOOL_BACKEND=${GLM53_KPOOL_BACKEND:-ascendc}
export GLM53_SWIGLU_BACKEND=torch

export GLM53_KDA_SEQUENCE_MODE=packed
export GLM53_KDA_FWD_HEAD_CHUNK_SIZE=64
export GLM53_KDA_BWD_HEAD_CHUNK_SIZE=64

# Replay the same saved batch for 10 training steps by default.
# LOAD_DEBUG_ROLLOUT_DATA=${LOAD_DEBUG_ROLLOUT_DATA-/mnt/share/l00804615/glm5.3_flash_native/cache/work/20261029_115048_node25_rollout_once_retry/rollouts/rollout-0-response8192-64samples.pt}
# 4096-response truncation of repeat_64x8192 (same 64 samples, same prompts;
# built by repeat_64x4096/build_dataset.py and validated with the slime loader).
# LOAD_DEBUG_ROLLOUT_DATA=${LOAD_DEBUG_ROLLOUT_DATA-/mnt/share/l00937981/glm_53_flash_slime/cache/rollouts/dapo_8k_114_first_20261106/repeat_64x4096/rollout_0.pt}
NUM_ROLLOUT=${NUM_ROLLOUT:-10}
ROLLOUT_ONLY=${ROLLOUT_ONLY:-1}

if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA:-}" && "${ROLLOUT_ONLY:-0}" == "1" ]]; then
    echo "LOAD_DEBUG_ROLLOUT_DATA and ROLLOUT_ONLY=1 are mutually exclusive" >&2
    exit 2
fi

ray stop -f

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT=${LOG_ROOT:-/mnt/share/l00937981/glm_53_flash_slime/start/logs/}

# RAY_NODE_IP must exist before lh-ascend-runtime.sh is sourced (it exports
# no_proxy with this value). COMM_IFNAME lives in runtime/host.env.
source "${SCRIPT_DIR}/../runtime/host.env"
# This container has no iproute2 and exposes eth0 instead of the host VLAN
# interface, so fall back to `hostname -I` when `ip`/COMM_IFNAME are unusable.
if command -v ip >/dev/null 2>&1 && [[ -d /sys/class/net/$COMM_IFNAME ]]; then
    RAY_NODE_IP=$(ip -o -4 addr show dev "$COMM_IFNAME" | awk '{print $4}' | cut -d/ -f1)
else
    RAY_NODE_IP=$(hostname -I | awk '{print $1}')
fi
[[ -n $RAY_NODE_IP ]] || { echo "No IPv4 address on $COMM_IFNAME" >&2; exit 2; }

# Single 8-NPU node. Keep GLM53_NUM_NPUS and the visible device list in sync
# with the single-node runtime's 8 Ray NPU resources.
GLM53_NUM_NPUS=8
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# The public CANN package eagerly builds several extensions. Cache those
# builds before Ray starts so they do not run while a pipeline peer is waiting
# for its first tensor.
export GLM53_PRELOAD_CANN_OPS=${GLM53_PRELOAD_CANN_OPS:-1}

source "${SCRIPT_DIR}/../runtime/lh-ascend-runtime.sh"

# host.env points GLOO/HCCL at a VLAN interface that only exists on the bare
# host; fall back to eth0 inside the container. Same fix as
# run_glm53_single_node_dapo_8k.sh.
if [[ ! -d /sys/class/net/$HCCL_SOCKET_IFNAME ]]; then
    export GLOO_SOCKET_IFNAME=${GLM53_COMM_IFNAME:-eth0}
    export HCCL_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
fi

# Full-size checkpoint: 288 routed experts (599 GB on disk).
# Do NOT use the top16/top32 SFT checkpoints here; this recipe must use the
# full-size 288-expert weights.
HF_CHECKPOINT=${HF_CHECKPOINT:-/mnt/weight/GLM-5.3-Flash-BF16}
# Fixed-length rollout prompt set: the first sample of repeat_64x4096 copied 200x
# (dapo_4k_first_prompt_x200.jsonl, built 2026-11-10, sha256 in the .manifest.json).
# The prompt is an OpenAI message list (source row 0), so the GLM template
# must be applied via --apply-chat-template.
# PROMPT_DATA=${PROMPT_DATA:-/mnt/share/r30009656/code_project/GML5-3_s-b/glm53_32k_n1024/dataset/dapo_4k_first_prompt_x200.jsonl}
PROMPT_DATA=${PROMPT_DATA:-/mnt/share/l00804615/glm5.3_flash_native/cache/work/20261112_perf_4k_200/dapo_4k_first_prompt_x200.jsonl}
EXPERIMENT_NAME=glm53
# Save each rollout for later --load-debug-rollout-data training replay.
# A separate directory per launch avoids overwriting previous runs.
ROLLOUT_SAVE_DIR=${ROLLOUT_SAVE_DIR:-/mnt/share/l00804615/glm5.3_flash_native/cache/work/$(date +%Y%m%d_%H%M%S)_rh24_rollout_$$}
SAVE_DEBUG_ROLLOUT_DATA="${ROLLOUT_SAVE_DIR}/rollout-{rollout_id}.pt"
printf 'Rollout replay files: %s\n' "$SAVE_DEBUG_ROLLOUT_DATA"
RM_TYPE=deepscaler
CP_SIZE=1

# Fixed at 4k: every rollout is forced to generate exactly 4096 response tokens.
# --rollout-ignore-eos makes SGLang ignore EOS and run to max_new_tokens, so the
# length does not depend on whether the model would have stopped early.
ROLLOUT_MAX_RESPONSE_LEN=4096
# Prompt limit includes the chat template (the templated prompt is 161 tokens).
ROLLOUT_MAX_PROMPT_LEN=1024
MAX_CONTEXT_LEN=16384

# 8-card memory knobs. Raise the static fraction if KV cache is too small;
# lower it if activation/workspace allocation OOMs. Lower max-running-requests
# if SGLang reports KV-cache OOM at 32 concurrent 4k requests.
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.88}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-32}

# ---- Training profiling (opt-in) ---------------------------------------------
# Profiles the Megatron train steps (forward/backward/optimizer) with
# torch_npu.profiler via the slime before-train-step hook. Disabled by default.
# Steps are GLOBAL training-step indices (0-based, same as "train/step" in the
# logs; this recipe has 1 train step per rollout).
#   PROFILE=1 bash rh_run_glm53_single_node_24_rh.sh
#   PROFILE=1 PROFILE_STEP_START=3 PROFILE_STEP_END=3 PROFILE_RANKS=0,1 ...
# Hook module: scripts/lib/glm53/runtime/glm53_train_profiler.py
# Output: $GLM53_PROFILE_DIR/trainstep<S>-<E>/rank<N>/*_ascend_pt/ASCEND_PROFILER_OUTPUT/
GLM53_PROFILE_TRAIN=${GLM53_PROFILE_TRAIN:-${PROFILE:-0}}
GLM53_PROFILE_DIR=${GLM53_PROFILE_DIR:-/mnt/share/l00804615/glm5.3_flash_native/start/glm53-profiles-stack}
GLM53_PROFILE_STEP_START=${GLM53_PROFILE_STEP_START:-${PROFILE_STEP_START:-3}}
GLM53_PROFILE_STEP_END=${GLM53_PROFILE_STEP_END:-${PROFILE_STEP_END:-3}}
GLM53_PROFILE_RANKS=${GLM53_PROFILE_RANKS:-${PROFILE_RANKS:-0}}
GLM53_PROFILE_LEVEL=${GLM53_PROFILE_LEVEL:-${PROFILE_LEVEL:-Level1}}
GLM53_PROFILE_AIC=${GLM53_PROFILE_AIC:-${PROFILE_AIC:-AiCoreNone}}
GLM53_PROFILE_WITH_STACK=${GLM53_PROFILE_WITH_STACK:-${PROFILE_WITH_STACK:-0}}
GLM53_TB_LOG=${GLM53_TB_LOG:-0}
GLM53_PROFILE_MEMORY=${GLM53_PROFILE_MEMORY:-0}
GLM53_PROFILE_MICROBATCHES=${GLM53_PROFILE_MICROBATCHES:-0}

# ---- Device memory snapshot (ON by default for this variant) ----------------
# Records one whole train step's allocator history with stacks='all' and dumps
# a pickle at the next step's hook. With 1 train step per rollout, keep
# NUM_ROLLOUT >= GLM53_MEM_SNAPSHOT_STEP + 2 so the dump step exists.
# The hook currently records global rank 0 only (PP stage 0). Inspect the
# pickle with https://pytorch.org/memory_viz (memory_viz.py).
GLM53_MEM_SNAPSHOT=${GLM53_MEM_SNAPSHOT:-0}
GLM53_MEM_SNAPSHOT_STEP=${GLM53_MEM_SNAPSHOT_STEP:-3}
GLM53_MEM_SNAPSHOT_FILE=${GLM53_MEM_SNAPSHOT_FILE:-/mnt/share/l00804615/glm5.3_flash_native/start/glm53-mem-snapshots/glm53_train_mem_step${GLM53_MEM_SNAPSHOT_STEP}.pickle}
GLM53_MEM_SNAPSHOT_MAX_ENTRIES=${GLM53_MEM_SNAPSHOT_MAX_ENTRIES:-1000000}
mkdir -p "$(dirname "$GLM53_MEM_SNAPSHOT_FILE")"

# Model architecture (full-size checkpoint: 288 routed experts, topk 8).
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
# GLM53_MOE_DISPATCH_CHUNK_SIZE=${GLM53_MOE_DISPATCH_CHUNK_SIZE:-512}
# GLM53_EXPERT_TOKEN_CHUNK_SIZE=${GLM53_EXPERT_TOKEN_CHUNK_SIZE:-1024}
GLM53_MOE_DISPATCH_CHUNK_SIZE=${GLM53_MOE_DISPATCH_CHUNK_SIZE:-0}
GLM53_EXPERT_TOKEN_CHUNK_SIZE=${GLM53_EXPERT_TOKEN_CHUNK_SIZE:-0}
TRAIN_ENV_JSON=$(/usr/bin/python3 -I -S - \
    "$TRAIN_ENV_JSON" \
    "$GLM53_MOE_DISPATCH_CHUNK_SIZE" "$GLM53_EXPERT_TOKEN_CHUNK_SIZE" \
    "$GLM53_PROFILE_TRAIN" "$GLM53_PROFILE_DIR" \
    "$GLM53_PROFILE_STEP_START" "$GLM53_PROFILE_STEP_END" "$GLM53_PROFILE_RANKS" \
    "$GLM53_PROFILE_LEVEL" "$GLM53_PROFILE_AIC" "$GLM53_PROFILE_WITH_STACK" \
    "$GLM53_MEM_SNAPSHOT" "$GLM53_MEM_SNAPSHOT_STEP" \
    "$GLM53_MEM_SNAPSHOT_FILE" "$GLM53_MEM_SNAPSHOT_MAX_ENTRIES" \
    "$GLM53_TB_LOG" "$GLM53_PROFILE_MEMORY" "$GLM53_PROFILE_MICROBATCHES" \
    <<'PY_CHUNKS'
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
    "GLM53_MEM_SNAPSHOT", "GLM53_MEM_SNAPSHOT_STEP",
    "GLM53_MEM_SNAPSHOT_FILE", "GLM53_MEM_SNAPSHOT_MAX_ENTRIES",
    "GLM53_TB_LOG",
    "GLM53_PROFILE_MEMORY",
    "GLM53_PROFILE_MICROBATCHES",
)
for key, value in zip(profile_keys, sys.argv[4:]):
    values[key] = value
print(json.dumps(values))
PY_CHUNKS
)

# Rollout, optimizer, parallelism and SGLang settings.
ARGS=(
    --actor-num-nodes 1 --actor-num-gpus-per-node 8 --num-gpus-per-node 8 --colocate
    # Rollout-only run: the actor is never built, so keep the engine resident
    # (--colocate would otherwise enable the SGLang memory-saver offload).
    --no-offload-train --no-offload-rollout
    --train-env-vars "$TRAIN_ENV_JSON"
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "$HF_CHECKPOINT" --load "$HF_CHECKPOINT" --megatron-to-hf-mode raw
    # PROMPT_DATA is an OpenAI message list; render it with the GLM chat template.
    --apply-chat-template
    --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label
    --rm-type "${RM_TYPE:-deepscaler}" --seed 522
    # 200 identical prompts x 1 sample = 200 fixed-length 4k generations per rollout.
    --num-rollout "${NUM_ROLLOUT:-2}" --rollout-batch-size 200 --n-samples-per-prompt 1
    --rollout-ignore-eos
    --global-batch-size 64 --micro-batch-size 1
    --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN" --rollout-temperature 1 --rollout-top-p 1
    --rollout-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN" --rollout-max-context-len "$MAX_CONTEXT_LEN"
    --advantage-estimator grpo --pg-loss-type reinforce
    # TP=8 x PP=1 -> DP=1. Only the SGLang engine is instantiated in this
    # rollout-only recipe; these stay consistent for the (skipped) actor.
    # PP=1 means all 45 layers on one rank, so the uneven-pipeline
    # --decoder-first/last-pipeline-num-layers knobs must NOT be set.
    --tensor-model-parallel-size 8 --pipeline-model-parallel-size 1
    --context-parallel-size "${CP_SIZE:-1}" --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
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
    # One engine across all 8 NPUs: TP=8 / DP=1 / EP=1.
    --rollout-num-gpus-per-engine 8
    --sglang-device npu --sglang-attention-backend ascend --sglang-tp-size 8
    --sglang-dp-size 1 --sglang-ep-size 1
    # Full-size weights are ~599 GB / 8 ranks = ~75 GB per card. This needs
    # 96 GB HBM per NPU; on 64 GB cards the full-size checkpoint cannot fit.
    # 0.88 x 96 GB = 84.5 GB static budget leaves ~9.5 GB for KV/activation.
    --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" --sglang-context-length "$MAX_CONTEXT_LEN"
    --sglang-max-running-requests "$SGLANG_MAX_RUNNING_REQUESTS" --sglang-chunked-prefill-size 1024
    --sglang-disable-overlap-schedule --sglang-cuda-graph-bs 1 2 4 8 16 24 32
    --sglang-page-size 64 --sglang-kv-cache-dtype bfloat16 --sglang-watchdog-timeout 1200
    --attention-dropout 0 --hidden-dropout 0 --bf16
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
    --distributed-timeout-minutes 20 --update-weight-buffer-size 268435456
    --rollout-data-transport object-store
    # --save-debug-rollout-data "$SAVE_DEBUG_ROLLOUT_DATA"
    --use-rollout-routing-replay
)

# Attach the profiling/memory hook only when requested; normal runs stay untouched.
if [[ "$GLM53_PROFILE_TRAIN" == "1" || "$GLM53_MEM_SNAPSHOT" == "1" ]]; then
    ARGS+=(--custom-megatron-before-train-step-hook-path glm53_train_profiler.before_train_step)
fi

# Optional rollout capture without initializing or training the actor model.
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA:-}" ]]; then
    ARGS+=(--load-debug-rollout-data "$LOAD_DEBUG_ROLLOUT_DATA")
elif [[ "${ROLLOUT_ONLY:-0}" == "1" ]]; then
    ARGS+=(--debug-rollout-only)
else
    # This check resets serving weights and requires the training actor to restore them.
    ARGS+=(--check-weight-update-equal)
fi

glm53_submit "$@"