#!/usr/bin/env bash
# GLM-5.3-Flash, native single-node recipe. Edit the parameters below.
# Train-memory variant: replay one saved rollout and record the allocator
# history of one whole train step (stacks='all') for memory_viz.
# --check performs static validation; a normal launch retains exclusive resource gates.
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
LOAD_DEBUG_ROLLOUT_DATA=${LOAD_DEBUG_ROLLOUT_DATA-/mnt/share/l00937981/glm_53_flash_slime/cache/rollouts/dapo_8k_114_first_20261106/repeat_64x8192/rollout_0.pt}
NUM_ROLLOUT=${NUM_ROLLOUT:-10}
ROLLOUT_ONLY=${ROLLOUT_ONLY:-0}

if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA:-}" && "${ROLLOUT_ONLY:-0}" == "1" ]]; then
    echo "LOAD_DEBUG_ROLLOUT_DATA and ROLLOUT_ONLY=1 are mutually exclusive" >&2
    exit 2
fi

ray stop -f

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT=/mnt/share/l00937981/glm_53_flash_slime/start/logs/

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

# This box (A3) has 16 NPUs. Must be set before lh-ascend-runtime.sh, which
# otherwise defaults to 8 devices / 8 Ray NPU resources.
GLM53_NUM_NPUS=16
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

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

HF_CHECKPOINT=/mnt/share/r30009656/code_project/GML5-3_s-b/start/CKPT/Geo3k/glm53_geo3k_sft-2/iter_0000400_bf16
# Same dataset the replayed rollout was generated from (see run_manifest.json in
# cache/rollouts/dapo_8k_114_first_20261106).
PROMPT_DATA=/mnt/share/r30009656/code_project/GML5-3_s-b/glm53_32k_n1024/dataset/dapo-math-17k_lt8k_sft_183_slime.jsonl
EXPERIMENT_NAME=glm53
# Save each rollout for later --load-debug-rollout-data training replay.
# A separate directory per launch avoids overwriting previous runs.
ROLLOUT_SAVE_DIR=${ROLLOUT_SAVE_DIR:-/mnt/share/l00804615/glm5.3_flash_native/cache/work/$(date +%Y%m%d_%H%M%S)_rh24_rollout_$$}
SAVE_DEBUG_ROLLOUT_DATA="${ROLLOUT_SAVE_DIR}/rollout-{rollout_id}.pt"
printf 'Rollout replay files: %s\n' "$SAVE_DEBUG_ROLLOUT_DATA"
RM_TYPE=deepscaler
CP_SIZE=1

# Prompt limit includes the chat template. Total context capacity includes prompt and response.
ROLLOUT_MAX_RESPONSE_LEN=8192
ROLLOUT_MAX_PROMPT_LEN=1024
MAX_CONTEXT_LEN=16384

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
    --actor-num-nodes 1 --actor-num-gpus-per-node 16 --num-gpus-per-node 16 --colocate
    # Keep the training actor and rollout engine resident; --colocate otherwise
    # enables both offloads. Safe for debug-replay (train-only) runs.
    --no-offload-train --no-offload-rollout
    --train-env-vars "$TRAIN_ENV_JSON"
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "$HF_CHECKPOINT" --load "$HF_CHECKPOINT" --megatron-to-hf-mode raw
    --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label --apply-chat-template
    --rm-type "${RM_TYPE:-deepscaler}" --seed 522
    --num-rollout "${NUM_ROLLOUT:-2}" --rollout-batch-size 8 --n-samples-per-prompt 8
    --global-batch-size 64 --micro-batch-size 1
    --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN" --rollout-temperature 1 --rollout-top-p 1
    --rollout-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN" --rollout-max-context-len "$MAX_CONTEXT_LEN"
    --advantage-estimator grpo --pg-loss-type reinforce
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 4
    # 45 = 12 + 12 + 12 + 9. The last stage also carries the final norm and the
    # LM head (154880 x 4096 plus the logits GEMM, ~3.5 layer-equivalents), so
    # 12/11/11/11 left it ~1 stage slower and rank0 blocked in PP p2p (38.7 s
    # per profiled step). 12/12/12/9 balances the stages; middle stages need
    # (45 - 12 - 9) % (PP - 2) == 0.
    --decoder-first-pipeline-num-layers 12 --decoder-last-pipeline-num-layers 9
    # 16 NPUs: TP=1 x PP=4 -> DP=4, EP=4 splits the 16 experts 4-per-rank
    # (EP=2 held 8 experts/rank and OOMed on 8 NPUs).
    --context-parallel-size "${CP_SIZE:-1}" --expert-model-parallel-size 4 --expert-tensor-parallel-size 1
    --allgather-cp --moe-token-dispatcher-type alltoall
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --seq-length "$MAX_CONTEXT_LEN" --max-position-embeddings "$MAX_CONTEXT_LEN"
    # Packing budget only: an oversized individual sample is scheduled alone.
    --use-dynamic-batch-size --max-tokens-per-gpu 8192 --data-pad-size-multiplier 1024
    --log-probs-chunk-size 1024
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
    --no-load-optim --no-load-rng --no-save-optim --freeze-params-name-list 'model.visual.*'
    --optimizer-cpu-offload --use-precision-aware-optimizer
    --overlap-cpu-optimizer-d2h-h2d
    # DDP grad-reduce overlap only. --overlap-param-gather is deliberately NOT
    # enabled: slime calls DDP.disable_forward_pre_hook() at the start of every
    # rollout, and the second call raises KeyError on the already-removed handles.
    --overlap-grad-reduce
    --rollout-num-gpus-per-engine 16
    --sglang-device npu --sglang-attention-backend ascend --sglang-tp-size 4
    --sglang-dp-size 1 --sglang-ep-size 1
    --sglang-mem-fraction-static 0.55 --sglang-context-length "$MAX_CONTEXT_LEN"
    --sglang-max-running-requests 32 --sglang-chunked-prefill-size 1024
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
