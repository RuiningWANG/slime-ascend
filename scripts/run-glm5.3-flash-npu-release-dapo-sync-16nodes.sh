#!/usr/bin/env bash
# 16 nodes x 8 NPUs: 8 training nodes + 8 separate rollout nodes.
# Train TP1/PP4/CP4/EP16 (DP4); rollout 4 replicas x TP16 (2 nodes/replica).
# Reference: gpu_train_log.log online hyperparameters (config.yaml is a different offline test).
# NPU kernels and inference TP layout are adapted; training hyperparameters match the log.
# All nodes need the same shared paths, environment and routable HCCL interface.
# Start a dedicated Ray cluster with the 8 training nodes, submit this script,
# then join the 8 rollout nodes after the 64 training bundles are reserved.
# Each Ray node must advertise exactly 8 NPUs. No --colocate, --offload-train or --offload-rollout (optimizer offload is separate).
set -euo pipefail
umask 077
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export WORK_ROOT=${WORK_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)}
# Prefer deployment activation; fall back to this release's bundled environment.
set +u
if [[ -r "$WORK_ROOT/activate.sh" ]]; then
    source "$WORK_ROOT/activate.sh"
else
    export CONDA_PREFIX="$WORK_ROOT/env"
    export PATH="$CONDA_PREFIX/bin:$PATH"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
    source "$WORK_ROOT/cann/ascend-toolkit/set_env.sh"
    source "$WORK_ROOT/cann/nnal/atb/set_env.sh" --cxx_abi=1
    source "$WORK_ROOT/cann/cann-9.2.0/share/info/ascendnpu-ir/bin/set_env.sh"
    source "$ASCEND_OPP_PATH/vendors/custom_transformer/bin/set_env.bash"
    source "$CONDA_PREFIX/lib/python3.12/site-packages/deep_ep/vendors/hwcomputing/bin/set_env.bash"
    source "$CONDA_PREFIX/lib/python3.12/site-packages/fla_npu/opp/vendors/fla_npu_transformer/bin/set_env.bash"
    export PYTHONPATH="$WORK_ROOT/src/slime-ascend/scripts/lib/glm53/runtime:$WORK_ROOT/src/slime-ascend/slime:$WORK_ROOT/src/Megatron-LM:$WORK_ROOT/src/MegatronAdaptor:$WORK_ROOT/src/TransformerEngineNPU:$WORK_ROOT/src/sglang/python:${PYTHONPATH:-}"
fi
set -euo pipefail
export SLIME_ROOT=${SLIME_ROOT:-$WORK_ROOT/src/slime-ascend/slime}
export GLM53_NATIVE_ROOT="$WORK_ROOT/src"
export TORCH_DEVICE_BACKEND_AUTOLOAD=1 FLA_NPU_DISABLE_PTH=1 PYTHONNOUSERSITE=1
# Online GPU baseline: 128 prompts x 4 responses = GBS512, one update/rollout.
default_rounds=24; default_rm=deepscaler
rollout_batch=128; samples=4; global_batch=512; oversampling=128
default_context=65536; default_response=57344; sequence_length=65536
pad_multiplier=128; weight_buffer=1073741824; distributed_timeout=60
PROFILE_BATCH_ARGS=(--use-dynamic-batch-size --rollout-shuffle)
PROFILE_OPTIMIZER_ARGS=(--optimizer-cpu-offload --optimizer-offload-fraction 0.5 --overlap-cpu-optimizer-d2h-h2d)
export HF_CHECKPOINT=${HF_CHECKPOINT:-/mnt/weight/GLM-5.3-Flash-BF16}
export PROMPT_DATA=${PROMPT_DATA:-/mnt/share/l00804615/datasets/dapo-math-17k/dapo-math-17k.jsonl}
export NUM_ROLLOUT=${NUM_ROLLOUT:-$default_rounds}
export RM_TYPE=${RM_TYPE:-$default_rm}
CHECK_ONLY=${CHECK_ONLY:-0}
if [[ ${1:-} == --dapo-math ]]; then
    shift # DAPO is the default profile; accept the legacy selector.
fi
for arg in "$@"; do
    case "$arg" in
        --colocate|--offload-train|--offload-rollout|--debug-train-only|--debug-rollout-only|--rollout-external|--release-train)
            echo "Unsupported override for the non-colocated 16-node launcher: $arg" >&2; exit 2 ;;
    esac
done
[[ $NUM_ROLLOUT =~ ^[1-9][0-9]*$ ]] || { echo 'NUM_ROLLOUT must be a positive integer' >&2; exit 2; }
[[ $CHECK_ONLY == 0 || $CHECK_ONLY == 1 ]] || { echo 'CHECK_ONLY must be 0 or 1' >&2; exit 2; }
python -I -S - "$HF_CHECKPOINT" "$PROMPT_DATA" "$SLIME_ROOT" <<'PY_INPUTS'
import json, pathlib, sys
model, data, code = map(pathlib.Path, sys.argv[1:])
assert (code/'train.py').is_file(), f'Missing training code: {code}/train.py'
config=json.loads((model/'config.json').read_text())
config=config.get('text_config', config)
for key, expected in [('num_hidden_layers',45), ('n_routed_experts',288), ('num_experts_per_tok',8)]:
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
# Full DAPO JSONL uses chat-message prompts and string labels.
keys=('prompt','label')
assert isinstance(first,dict) and all(k in first for k in keys), f'JSONL needs {keys} fields'
# DAPO is text-only; prompt/label validation above is sufficient.
print(f'Inputs OK: model={model}, data={data}, fields={keys}, python={sys.executable}', flush=True)
PY_INPUTS

# Shared run directory for the 16-node non-colocated job.
run_dir=
input_key=prompt
label_key=label
if [[ ${1:-} != --dapo-math && -n ${WORK_ROOT:-} &&
      -d "$WORK_ROOT/src/slime-ascend" ]]; then
    export HF_CHECKPOINT=${HF_CHECKPOINT:-/mnt/weight/GLM-5.3-Flash-BF16}
    unset MULTIMODAL_KEYS
    [[ -r "$HF_CHECKPOINT/config.json" && -r "$PROMPT_DATA" ]] || {
        echo "Missing checkpoint or DAPO data: $HF_CHECKPOINT, $PROMPT_DATA" >&2
        exit 1
    }

    umask 077
    mkdir -p "$WORK_ROOT/cache/"{work,archive}
    run_dir=$(mktemp -d "$WORK_ROOT/cache/work/$(date +%Y%m%d_%H%M%S)_dapo_split_$(hostname -s)_XXXXXX")
    # One-based optimizer update; this launcher runs one update per rollout.
    export GLM53_PROFILE_UPDATE=3 GLM53_PROFILE_DIR="$run_dir/profiling"
    export SGLANG_PROFILE_V2=0 SGLANG_PROFILE_RECORD_SHAPES=1
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
    export HF_HOME="${HF_HOME:-$WORK_ROOT/cache/huggingface}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORK_ROOT/cache/xdg}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORK_ROOT/cache/triton}"
    export ASCEND_WORK_PATH="$run_dir/ascend" ASCEND_PROCESS_LOG_PATH="$run_dir/ascend/plog"
    mkdir -p "$ASCEND_PROCESS_LOG_PATH"
fi

: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the model directory}"
: "${PROMPT_DATA:?Set PROMPT_DATA to the training JSONL}"
SLIME_ROOT=${SLIME_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../slime" && pwd)}
NUM_ROLLOUT=${NUM_ROLLOUT:-$default_rounds}
ROLLOUT_SESSION_ID=${ROLLOUT_SESSION_ID:-$(python -c 'import uuid; print(uuid.uuid4().hex)')}
ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-4096}
PROFILE_SGLANG_ARGS=()
# Keep sequence length separate from the rollout context cap, as in YAML.
MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-$default_context}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-$default_response}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384}
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
    --actor-num-nodes 8 --actor-num-gpus-per-node 8 --num-gpus-per-node 8 --rollout-num-gpus 64
)

CHECKPOINT_ARGS=(
    --hf-checkpoint "$HF_CHECKPOINT" --load "$HF_CHECKPOINT"
)

DATA_ARGS=(
    --prompt-data "$PROMPT_DATA" --input-key "$input_key" --label-key "$label_key" --apply-chat-template
)

ROLLOUT_ARGS=(
    --rm-type "$RM_TYPE" --seed 1234
    --num-rollout "$NUM_ROLLOUT" --num-steps-per-rollout 1 --rollout-session-id "$ROLLOUT_SESSION_ID"
    --rollout-batch-size "$rollout_batch" --n-samples-per-prompt "$samples" --over-sampling-batch-size "$oversampling"
    --global-batch-size "$global_batch" --micro-batch-size 1
    --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN" --rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1
    --rollout-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN" --rollout-max-context-len "$MAX_CONTEXT_LEN"
    --advantage-estimator grpo --pg-loss-type ppo
    --disable-grpo-std-normalization --calculate-per-token-loss
    --use-tis --tis-clip-low 0.5 --tis-clip 2.0
    --eps-clip 0.2 --eps-clip-high 0.28 --kl-coef 0 --kl-loss-coef 0 --entropy-coef 0
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 4
    --decoder-first-pipeline-num-layers 12 --decoder-last-pipeline-num-layers 11
    --context-parallel-size 4 --expert-model-parallel-size 16 --expert-tensor-parallel-size 1
    --allgather-cp --moe-token-dispatcher-type alltoall
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --seq-length "$sequence_length" --max-position-embeddings "$sequence_length"
    "${PROFILE_BATCH_ARGS[@]}" --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU" --data-pad-size-multiplier "$pad_multiplier"
    --log-probs-chunk-size 256
)

OPTIMIZER_ARGS=(
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
    --no-load-optim --no-load-rng --no-save-optim --no-save-rng --freeze-params-name-list 'model.visual.*'
    --use-precision-aware-optimizer --use-distributed-optimizer
    "${PROFILE_OPTIMIZER_ARGS[@]}"
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 16
    # Slime derives tensor parallel size from rollout-num-gpus-per-engine / PP.
    --sglang-device npu --sglang-attention-backend ascend
    --sglang-dp-size 1 --sglang-ep-size 1
    "${PROFILE_SGLANG_ARGS[@]}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.9}" --sglang-context-length "${SGLANG_CONTEXT_LENGTH:-65600}"
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-64}" --sglang-chunked-prefill-size 2048
    --sglang-disable-overlap-schedule --sglang-cuda-graph-bs-decode 1 2 4 8 16 24 32
    --sglang-page-size 512 --sglang-mamba-track-interval 512
    --sglang-kv-cache-dtype bfloat16 --sglang-watchdog-timeout 3600
)

# Each job owns a unique shared directory; all training/inference hosts must see it.
mkdir -p "$WORK_ROOT/cache/weight_sync"
weight_sync_dir=$(mktemp -d "$WORK_ROOT/cache/weight_sync/dapo_$(date +%Y%m%d_%H%M%S)_XXXXXX")
printf 'Shared weight sync directory: %s\n' "$weight_sync_dir"

TRAINING_ARGS=(
    --attention-dropout 0 --hidden-dropout 0 --bf16
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
    # NaN checking is enabled by Megatron by default in this version.
    --distributed-timeout-minutes "$distributed_timeout" --update-weight-buffer-size "$weight_buffer"
    --rollout-data-transport object-store
    # Full HF checkpoint written to shared disk, then reloaded after each rollout.
    --update-weight-mode full --update-weight-transport disk
    --update-weight-disk-dir "$weight_sync_dir"
    # Explicit user override of the GPU baseline: replay each sample's rollout routes.
    --use-rollout-routing-replay
    --skip-eval-before-train --log-interval 100
)

if [[ -n "$run_dir" ]]; then
    # Slime assigns rollout_seed + sample index within each prompt group.
    ROLLOUT_ARGS+=(--rollout-seed 1234)
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
    "TORCH_DEVICE_BACKEND_AUTOLOAD", "USE_LIBUV", "PYTHONPATH", "LD_LIBRARY_PATH",
    "CONDA_PREFIX", "PATH", "ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_AICPU_PATH",
    "FLA_NPU_DISABLE_PTH", "PYTHONNOUSERSITE", "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
}
env = {k: v for k, v in os.environ.items() if k.startswith(prefixes) or k in worker_keys}
print(json.dumps({"env_vars": env}))
PY
)
# Ray owns scheduling; no host IPs or destructive cluster reset are hard-coded.
# Join training nodes first to enforce physical train/rollout separation.
if ((CHECK_ONLY)); then
    printf 'GPU-log profile; training=8x8 TP1/PP4/CP4/EP16 DP4; rollout=8x8, 4xTP16\n'
    printf '%q ' python -u "$SLIME_ROOT/train.py" "${MODEL_ARGS[@]}" "${CLUSTER_ARGS[@]}" \
        "${CHECKPOINT_ARGS[@]}" "${DATA_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${PARALLEL_ARGS[@]}" \
        "${OPTIMIZER_ARGS[@]}" "${SGLANG_ARGS[@]}" "${TRAINING_ARGS[@]}" "$@"
    printf '\nChecks passed; no Ray job or training was started.\n'
    exit 0
fi
submit_job() {
    : "${RAY_ADDRESS:?Set RAY_ADDRESS to the dedicated Ray head HTTP dashboard URL}"
    # Cluster operators must add all 16 node IPs to NO_PROXY when using a proxy.
    export no_proxy="${no_proxy:-${NO_PROXY:-}},localhost,127.0.0.1"
    export NO_PROXY="$no_proxy"
    ray job submit --address "$RAY_ADDRESS" "$@" 2>&1 | tee "$run_dir/run.log"
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
