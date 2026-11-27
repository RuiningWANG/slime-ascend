#!/usr/bin/env bash
# Standalone GLM-5.3 Flash Geo3K launcher for eight colocated A5 NPUs.
# Training PP4/CP2/EP2; rollout TP8; routing replay enabled.
set -euo pipefail
umask 077
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export WORK_ROOT=${WORK_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)}
[[ -r "$WORK_ROOT/activate.sh" ]] || { echo "Missing $WORK_ROOT/activate.sh" >&2; exit 1; }
# Vendor environment scripts can reference unset variables.
set +u
source "$WORK_ROOT/activate.sh"
set -euo pipefail
export HF_CHECKPOINT=${HF_CHECKPOINT:-/mnt/weight/GLM-5.3-moe16_geo3k/glm53_geo3k_sft/iter_0000400_bf16}
export PROMPT_DATA=${PROMPT_DATA:-/mnt/share/r30009656/dataset/glm53_rl/train.jsonl}
export NUM_ROLLOUT=${NUM_ROLLOUT:-10}
CHECK_ONLY=${CHECK_ONLY:-0}
if [[ ${1:-} == --dapo-math ]]; then
    echo 'This release entry point uses the Geo3K profile; --dapo-math is unsupported.' >&2
    exit 2
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
assert 'images' in first, 'Geo3K JSONL needs images field'
print(f'Inputs OK: model={model}, data={data}, fields={keys}, python={sys.executable}', flush=True)
PY_INPUTS
if ((CHECK_ONLY)); then
    echo 'Input/path checks passed; no Ray or training was started.'
    exit 0
fi

# The release environment supplies the single-node Geo3K profile.
run_dir=
input_key=prompt
label_key=label
geo3k_data=/mnt/share/l00804615/datasets/geo3k_imgurl/train.jsonl
if [[ ${1:-} != --dapo-math && -n ${WORK_ROOT:-} &&
      -d "$WORK_ROOT/src/slime-ascend" ]]; then
    export HF_CHECKPOINT=${HF_CHECKPOINT:-/mnt/weight/GLM-5.3-moe16_geo3k/glm53_geo3k_sft/iter_0000400_bf16}
    export PROMPT_DATA="${PROMPT_DATA:-$geo3k_data}"
    if [[ -z ${MULTIMODAL_KEYS:-} ]]; then export MULTIMODAL_KEYS='{"image":"images"}'; fi
    [[ -r "$HF_CHECKPOINT/config.json" && -r "$PROMPT_DATA" ]] || {
        echo "Missing checkpoint or Geo3K data: $HF_CHECKPOINT, $PROMPT_DATA" >&2
        exit 1
    }

    umask 077
    mkdir -p "$WORK_ROOT/cache/"{work,archive}
    run_dir=$(mktemp -d "$WORK_ROOT/cache/work/$(date +%Y%m%d_%H%M%S)_geo3k_$(hostname -s)_XXXXXX")
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
NUM_ROLLOUT=${NUM_ROLLOUT:-3}
ROLLOUT_SESSION_ID=${ROLLOUT_SESSION_ID:-$(python -c 'import uuid; print(uuid.uuid4().hex)')}
ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-1024}
PROFILE_OPTIMIZER_ARGS=()
PROFILE_SGLANG_ARGS=()
if [[ ${1:-} == --dapo-math ]]; then
    shift
    MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-16384}
    ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-8192}
    MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
    export GLM53_MOE_DISPATCH_CHUNK_SIZE=512 GLM53_EXPERT_TOKEN_CHUNK_SIZE=1024
    export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:False}
    train_expansion_mode=AICPU_CacheDisable
    PROFILE_OPTIMIZER_ARGS+=(--no-pin-cpu-grads)
else
    MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-8192}
    ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-4096}
    MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-4096}
    export GLM53_MOE_DISPATCH_CHUNK_SIZE=0 GLM53_EXPERT_TOKEN_CHUNK_SIZE=0
    export GLM53_KDA_SEQUENCE_MODE=packed GLM53_KDA_FWD_HEAD_CHUNK_SIZE=64 GLM53_KDA_BWD_HEAD_CHUNK_SIZE=64
    export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:False,garbage_collection_threshold:0.8}
    train_expansion_mode=AI_CPU
    PROFILE_SGLANG_ARGS+=(--sglang-mm-enable-dp-encoder)
fi
# Ray actors inherit these settings through the job's runtime environment.
export SGLANG_ADMIN_API_KEY="${SGLANG_ADMIN_API_KEY:-$(python -c 'import secrets; print(secrets.token_urlsafe(32))')}"
export HCCL_OP_EXPANSION_MODE=AI_CPU SLIME_DESTROY_WORLD_PROCESS_GROUP=0
export ASCEND_USE_FIA=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-256} DEEPEP_HCCL_BUFFSIZE=${DEEPEP_HCCL_BUFFSIZE:-256}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-300} HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}
export ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-1900}
# The colocated rollout and training communicators need separate port ranges.
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-9000-9099}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-9100-9199}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-lo} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
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
    --actor-num-nodes 1 --actor-num-gpus-per-node 8 --num-gpus-per-node 8 --colocate
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
    --context-parallel-size 2 --expert-model-parallel-size 2 --expert-tensor-parallel-size 1
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
    --overlap-grad-reduce
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
submit_job() {
    if [[ -z "$run_dir" ]]; then
        exec ray job submit "$@"
    fi
    if [[ -n ${RAY_ADDRESS:-} || -n ${RAY_API_SERVER_ADDRESS:-} ]]; then
        ray job submit "$@" 2>&1 | tee "$run_dir/run.log"
        return
    fi

    # Serialize private eight-card runs on the same node, including concurrent starts.
    exec {lock_fd}>"$WORK_ROOT/cache/geo3k-$(hostname -s).lock"
    flock -n "$lock_fd" || { echo "Another Geo3K run owns this node." >&2; return 1; }
    npu-smi info >"$run_dir/npu-before.log"
    if [[ $(grep -c 'No running processes found in NPU' "$run_dir/npu-before.log") != 8 ]] ||
       pgrep -x raylet >/dev/null || pgrep -x gcs_server >/dev/null; then
        echo "Local startup needs eight idle NPUs and a free Ray node. Set RAY_ADDRESS to use an existing cluster." >&2
        return 1
    fi
    local client_ip client_port server_ip server_port
    read -r client_ip client_port server_ip server_port <<<"${SSH_CONNECTION:-}"
    if [[ -z ${SLIME_HOST_IP:-} ]]; then
        if [[ -n "$server_ip" && "$server_ip" != *:* ]]; then
            export SLIME_HOST_IP="$server_ip"
        else
            export SLIME_HOST_IP="$(hostname -I | awk '{print $1}')"
        fi
    fi
    export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1,$SLIME_HOST_IP"
    export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$no_proxy"
    export RAY_USAGE_STATS_ENABLED=0
    ulimit -c 0
    ulimit -Sn 65535
    # Ray owns this cluster's lifetime. Use a short local path for its Unix sockets.
    python -u - "$run_dir" "$@" <<'PY' 2>&1 | tee "$run_dir/run.log"
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import ray

run_dir = Path(sys.argv[1])
os.chdir(run_dir)


def terminate(signum, frame):
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, terminate)
with tempfile.TemporaryDirectory(prefix="glm53-ray-", dir="/tmp") as temp:
    try:
        ctx = ray.init(address="local", _node_ip_address=os.environ["SLIME_HOST_IP"],
                       _temp_dir=temp, include_dashboard=True, dashboard_host="127.0.0.1",
                       num_cpus=min(96, len(os.sched_getaffinity(0))),
                       object_store_memory=32 * 1024**3)
        resources = ray.cluster_resources()
        if resources.get("NPU") != 8 or sum(node["Alive"] for node in ray.nodes()) != 1:
            raise RuntimeError(f"Expected one eight-NPU node, got {resources}")
        address = "http://" + ctx.address_info["webui_url"]
        result = subprocess.run(["ray", "job", "submit", "--address", address, *sys.argv[2:]])
        sys.exit(result.returncode)
    finally:
        ray.shutdown()
        logs = Path(temp, "session_latest", "logs")
        if logs.is_dir():
            shutil.copytree(logs, run_dir / "ray-logs", symlinks=True)
PY
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
