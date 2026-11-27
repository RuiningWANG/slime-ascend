#!/usr/bin/env bash
# Standalone GLM-5.3 Flash DAPO launcher for 8 training NPUs on 22 and 8 rollout NPUs on 23.
# Training PP4/CP1/EP2; rollout TP8; routing replay enabled.
set -euo pipefail
umask 077
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export WORK_ROOT=${WORK_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)}
[[ -r "$WORK_ROOT/activate.sh" ]] || { echo "Missing $WORK_ROOT/activate.sh" >&2; exit 1; }
# Vendor environment scripts can reference unset variables.
set +u
source "$WORK_ROOT/activate.sh"
set -euo pipefail
export HF_CHECKPOINT=${HF_CHECKPOINT:-/mnt/share/l00937981/iter_0003000_bf16}
export PROMPT_DATA=${PROMPT_DATA:-/mnt/share/l00937981/glm53_dapo_sft_rl.jsonl}
export NUM_ROLLOUT=${NUM_ROLLOUT:-100}
export RM_TYPE=${RM_TYPE:-deepscaler}
CHECK_ONLY=${CHECK_ONLY:-0}
if [[ ${1:-} == --dapo-math ]]; then
    shift # DAPO is the default profile; accept the legacy selector.
fi
[[ $NUM_ROLLOUT =~ ^[1-9][0-9]*$ ]] || { echo 'NUM_ROLLOUT must be a positive integer' >&2; exit 2; }
[[ $CHECK_ONLY == 0 || $CHECK_ONLY == 1 ]] || { echo 'CHECK_ONLY must be 0 or 1' >&2; exit 2; }
python -I -S - "$HF_CHECKPOINT" "$PROMPT_DATA" "$SLIME_ROOT" <<'PY_INPUTS'
import json, pathlib, sys
model, data, code = map(pathlib.Path, sys.argv[1:])
assert (code/'train.py').is_file(), f'Missing training code: {code}/train.py'
config=json.loads((model/'config.json').read_text())
config=config.get('text_config', config)
for key, expected in [('num_hidden_layers',45), ('n_routed_experts',16), ('num_experts_per_tok',8)]:
    assert config.get(key) == expected, f'{key}: expected {expected}, got {config.get(key)}'
indexes=list(model.glob('*.safetensors.index.json')) + list(model.glob('*.bin.index.json'))
if indexes:
    for index in indexes:
        for shard in set(json.loads(index.read_text())['weight_map'].values()):
            assert (model/shard).is_file(), f'Missing model shard: {shard}'
else:
    assert list(model.glob('*.safetensors')) or list(model.glob('pytorch_model*.bin')), 'No model weights found'
with data.open() as stream:
    first=next((json.loads(line) for line in stream if line.strip()), None)
# The repository launcher selects problem/answer for its local Geo3K dataset.
keys=('problem','answer') if str(data)=='/mnt/share/l00804615/datasets/geo3k_imgurl/train.jsonl' else ('prompt','label')
assert isinstance(first,dict) and all(k in first for k in keys), f'JSONL needs {keys} fields'
# DAPO is text-only; prompt/label validation above is sufficient.
print(f'Inputs OK: model={model}, data={data}, fields={keys}, python={sys.executable}', flush=True)
PY_INPUTS
if ((CHECK_ONLY)); then
    echo 'Input/path checks passed; no Ray or training was started.'
    exit 0
fi

# Standalone single-node DAPO run directory and actor settings.
run_dir=
input_key=prompt
label_key=label
geo3k_data=/mnt/share/l00804615/datasets/geo3k_imgurl/train.jsonl
if [[ ${1:-} != --dapo-math && -n ${WORK_ROOT:-} &&
      -d "$WORK_ROOT/src/slime-ascend" ]]; then
    export HF_CHECKPOINT=${HF_CHECKPOINT:-/mnt/share/l00937981/iter_0003000_bf16}
    export PROMPT_DATA="${PROMPT_DATA:-$geo3k_data}"
    unset MULTIMODAL_KEYS
    [[ -r "$HF_CHECKPOINT/config.json" && -r "$PROMPT_DATA" ]] || {
        echo "Missing checkpoint or DAPO data: $HF_CHECKPOINT, $PROMPT_DATA" >&2
        exit 1
    }

    umask 077
    mkdir -p "$WORK_ROOT/cache/"{work,archive}
    run_dir=$(mktemp -d "$WORK_ROOT/cache/work/$(date +%Y%m%d_%H%M%S)_dapo_split_$(hostname -s)_XXXXXX")
    finish_run() {
        local status=$?
        local archive="$WORK_ROOT/cache/archive/${run_dir##*/}"
        printf '%s\n' "$status" >"$run_dir/run.status"
        mv -T -- "$run_dir" "$archive" && printf 'Run logs: %s\n' "$archive"
    }
    trap finish_run EXIT
    printf 'Run logs: %s\n' "$run_dir"
    # Megatron must keep both CPU backups inside the training actor's TMS region.
    cat >"$run_dir/actor.yaml" <<'YAML'
megatron:
  - role: actor
    overrides:
      disable_param_buffers_cpu_backup: false
      disable_grad_buffers_cpu_backup: false
YAML
    if [[ "$PROMPT_DATA" == "$geo3k_data" ]]; then
        input_key=problem
        label_key=answer
    fi
    export HF_HOME="${HF_HOME:-$WORK_ROOT/cache/huggingface}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORK_ROOT/cache/xdg}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORK_ROOT/cache/triton}"
    export ASCEND_WORK_PATH="$run_dir/ascend" ASCEND_PROCESS_LOG_PATH="$run_dir/ascend/plog"
    mkdir -p "$ASCEND_PROCESS_LOG_PATH"
fi

: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the model directory}"
: "${PROMPT_DATA:?Set PROMPT_DATA to the training JSONL}"
SLIME_ROOT=${SLIME_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../slime" && pwd)}
NUM_ROLLOUT=${NUM_ROLLOUT:-100}
ROLLOUT_SESSION_ID=${ROLLOUT_SESSION_ID:-$(python -c 'import uuid; print(uuid.uuid4().hex)')}
ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-1024}
PROFILE_OPTIMIZER_ARGS=()
PROFILE_SGLANG_ARGS=()
# DAPO long-response profile; use the native PP4/CP1 execution layout.
MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-16384}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-8192}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
export GLM53_MOE_DISPATCH_CHUNK_SIZE=0
export GLM53_EXPERT_TOKEN_CHUNK_SIZE=0
export GLM53_KDA_SEQUENCE_MODE=packed GLM53_KDA_FWD_HEAD_CHUNK_SIZE=64 GLM53_KDA_BWD_HEAD_CHUNK_SIZE=64
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:False,garbage_collection_threshold:0.8}
train_expansion_mode=AI_CPU
# Ray actors inherit these settings through the job's runtime environment.
export SGLANG_ADMIN_API_KEY="${SGLANG_ADMIN_API_KEY:-$(python -c 'import secrets; print(secrets.token_urlsafe(32))')}"
export HCCL_OP_EXPANSION_MODE=AI_CPU SLIME_DESTROY_WORLD_PROCESS_GROUP=0
export ASCEND_USE_FIA=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-256} DEEPEP_HCCL_BUFFSIZE=${DEEPEP_HCCL_BUFFSIZE:-256}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-300} HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}
export ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-1900}
# Cross-node weight synchronization must use a routable interface.
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-9000-9099}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-9100-9199}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-enp35s0f2} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp35s0f2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export USE_LIBUV=${USE_LIBUV:-0} CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export SLIME_ACCELERATOR=npu TMS_HOOK_MODE=preload TMS_PRESERVE_PRELOAD_MODE=1
export SGLANG_MEMORY_SAVER_CUDA_GRAPH=true SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1
export SGLANG_NPU_HC_PRE_TAIL_GUARD=1 SGLANG_ENABLE_JIT_DEEPGEMM=False
export SGLANG_RUST_BUILD_MODE=never
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0

NUM_EXPERTS=$(python - "$HF_CHECKPOINT/config.json" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    config = json.load(stream)
print(config.get("text_config", config)["n_routed_experts"])
PY
)
MODEL_ARGS=(
    --spec slime_plugins.models.glm5_next.model get_glm5_next_model_provider
    --model-name glm5_next
    --num-layers 45 --hidden-size 4096 --ffn-hidden-size 12288
    --num-attention-heads 64 --num-query-groups 64 --kv-channels 256
    --q-lora-rank 1536 --kv-lora-rank 512
    --vocab-size 154880 --make-vocab-size-divisible-by 64
    --disable-bias-linear --swiglu --untie-embeddings-and-output-weights
    --position-embedding-type none --normalization RMSNorm --norm-epsilon 1e-5
    --rotary-base 800000 --num-experts "$NUM_EXPERTS"
    --moe-layer-freq '[0]*3+[1]*42'
    --moe-ffn-hidden-size 2048 --moe-shared-expert-intermediate-size 2048
    --moe-router-topk 8 --moe-router-score-function sigmoid --moe-router-pre-softmax
    --moe-router-enable-expert-bias --moe-router-bias-update-rate 0
    --moe-router-load-balancing-type seq_aux_loss --moe-router-topk-scaling-factor 2.5
    --moe-aux-loss-coeff 0 --moe-router-dtype fp32
    --moe-grouped-gemm --moe-permute-fusion --freeze-indexer --enable-experimental
)

CLUSTER_ARGS=(
    --actor-num-nodes 1 --actor-num-gpus-per-node 8 --num-gpus-per-node 8 --rollout-num-gpus 8
)

CHECKPOINT_ARGS=(
    --hf-checkpoint "$HF_CHECKPOINT" --load "$HF_CHECKPOINT" --megatron-to-hf-mode raw
)

DATA_ARGS=(
    --prompt-data "$PROMPT_DATA" --input-key "$input_key" --label-key "$label_key" --apply-chat-template
)

ROLLOUT_ARGS=(
    --rm-type "${RM_TYPE:-deepscaler}" --seed 522
    --num-rollout "$NUM_ROLLOUT" --num-steps-per-rollout 1 --rollout-session-id "$ROLLOUT_SESSION_ID"
    --rollout-batch-size 8 --n-samples-per-prompt 8
    --global-batch-size 64 --micro-batch-size 1
    --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN" --rollout-temperature 1 --rollout-top-p 1
    --rollout-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN" --rollout-max-context-len "$MAX_CONTEXT_LEN"
    --advantage-estimator grpo --pg-loss-type reinforce
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 4
    --decoder-first-pipeline-num-layers 12 --decoder-last-pipeline-num-layers 11
    --context-parallel-size 1 --expert-model-parallel-size 2 --expert-tensor-parallel-size 1
    --allgather-cp --moe-token-dispatcher-type alltoall
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --seq-length "$MAX_CONTEXT_LEN" --max-position-embeddings "$MAX_CONTEXT_LEN"
    --use-dynamic-batch-size --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU" --data-pad-size-multiplier 1024
    --log-probs-chunk-size 1024
)

OPTIMIZER_ARGS=(
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
    --no-load-optim --no-load-rng --no-save-optim --freeze-params-name-list 'model.visual.*'
    --optimizer-cpu-offload --use-precision-aware-optimizer
    "${PROFILE_OPTIMIZER_ARGS[@]}"
    --overlap-cpu-optimizer-d2h-h2d
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 8
    --sglang-device npu --sglang-attention-backend ascend --sglang-tp-size 8
    --sglang-dp-size 1 --sglang-ep-size 1
    "${PROFILE_SGLANG_ARGS[@]}"
    --sglang-mem-fraction-static 0.55 --sglang-context-length "$MAX_CONTEXT_LEN"
    --sglang-max-running-requests 32 --sglang-chunked-prefill-size 1024
    --sglang-disable-overlap-schedule --sglang-cuda-graph-bs-decode 1 2 4 8 16 24 32
    --sglang-page-size 512 --sglang-mamba-track-interval 512
    --sglang-kv-cache-dtype bfloat16 --sglang-watchdog-timeout 1200
)

TRAINING_ARGS=(
    --attention-dropout 0 --hidden-dropout 0 --bf16
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
    --check-for-nan-in-loss-and-grad --distributed-timeout-minutes 60 --update-weight-buffer-size 268435456
    --rollout-data-transport object-store
    --update-weight-transport disk
    --update-weight-disk-dir "$WORK_ROOT/cache/weight_sync/${run_dir##*/}"
    --check-weight-update-equal
    --use-rollout-routing-replay
)

if [[ -n "$run_dir" ]]; then
    # Slime assigns rollout_seed + sample index within each prompt group.
    ROLLOUT_ARGS+=(--rollout-seed 42)
    # NPU deterministic inference requires the optional batch_invariant_ops extension.
    if [[ ${GLM53_DETERMINISTIC_INFERENCE:-0} == 1 ]]; then
        SGLANG_ARGS+=(--sglang-enable-deterministic-inference)
    fi
    TRAINING_ARGS+=(--megatron-config-path "$run_dir/actor.yaml")
fi

# Geo3K uses image inputs and checkpoint-specific chat-template options.
if [[ -n ${MULTIMODAL_KEYS:-} ]]; then DATA_ARGS+=(--multimodal-keys "$MULTIMODAL_KEYS"); fi
if [[ -n ${CHAT_TEMPLATE_KWARGS:-} ]]; then DATA_ARGS+=(--apply-chat-template-kwargs "$CHAT_TEMPLATE_KWARGS"); fi
TRAIN_ENV_JSON=$(python - "$train_expansion_mode" <<'PY'
import json
import os
import sys

settings = {k: v for k, v in os.environ.items() if k.startswith("GLM53_")}
settings.update(
    HCCL_OP_EXPANSION_MODE=sys.argv[1],
    HCCL_HOST_SOCKET_PORT_RANGE="9200-9299",
    HCCL_NPU_SOCKET_PORT_RANGE="9300-9399",
)
settings.update(json.loads(os.environ.get("TRAIN_ENV_JSON", "{}")))
print(json.dumps(settings))
PY
)
CLUSTER_ARGS+=(--train-env-vars "$TRAIN_ENV_JSON")

RUNTIME_ENV=$(python - <<'PY'
import json
import os

prefixes = ("GLM53_", "SGLANG_", "TMS_", "HCCL_", "FLA_NPU_")
worker_keys = {
    "ACL_DEVICE_SYNC_TIMEOUT", "ASCEND_CUSTOM_OPP_PATH", "ASCEND_USE_FIA", "CUDA_DEVICE_MAX_CONNECTIONS",
    "DEEPEP_HCCL_BUFFSIZE", "GLOO_SOCKET_IFNAME", "OMP_NUM_THREADS", "PYTORCH_NPU_ALLOC_CONF",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES", "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO",
    "SLIME_ACCELERATOR", "SLIME_DESTROY_WORLD_PROCESS_GROUP", "TOKENIZERS_PARALLELISM",
    "TORCH_DEVICE_BACKEND_AUTOLOAD", "USE_LIBUV",
}
env = {k: v for k, v in os.environ.items() if k.startswith(prefixes) or k in worker_keys}
print(json.dumps({"env_vars": env}))
PY
)
# Use a dedicated two-node Ray cluster. Start its head on 177.48.22.1,
# submit this job before joining 23, then join 23 after the actor PG reserves
# all 8 NPUs on 22. This prevents PACK from reversing the requested roles.
submit_job() {
    : "${RAY_ADDRESS:?Set RAY_ADDRESS to the dedicated Ray head HTTP dashboard URL}"
    export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1,177.48.22.1,177.48.23.1"
    export NO_PROXY="$no_proxy"
    ray job submit "$@" 2>&1 | tee "$run_dir/run.log"
}

submit_job --runtime-env-json "$RUNTIME_ENV" -- python -u "$SLIME_ROOT/train.py" "${MODEL_ARGS[@]}" \
    "${CLUSTER_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" "$@"
