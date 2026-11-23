
# Prevent Conda/Python startup hooks from initializing a backend before SDK setup.
export TORCH_DEVICE_BACKEND_AUTOLOAD FLA_NPU_DISABLE_PTH PYTHONNOUSERSITE
unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD ASCEND_CUSTOM_OPP_PATH ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_CANN_PACKAGE_PATH ASCEND_HOME ASCEND_ATB_HOME ATB_HOME_PATH ASDSIP_HOME_PATH
set +u
source "${GLM53_ACTIVATE_SCRIPT:-$DEPS_ROOT/activate.sh}"
source "$FLA_PYTHON_ROOT/fla_npu/opp/vendors/fla_npu_transformer/bin/set_env.bash"
if [[ ${GLM53_LOGP_PARITY_ONLY:-0} == 1 || ${GLM53_REPLAY_SERVING_MOE_EP2:-0} == 1 ]]; then
    source "$(cat "$GLM53_BI_DEPS/vendor-env.path")"
fi
# FLA 26.10 embeds Conv1d. Registering the legacy private vendor first
# mixes its kernels with the new op_api and fails workspace creation (561002).
set -u
# TorchNPU walks ASCEND_HOME_PATH recursively during import. Keep an equivalent
# local directory view so version discovery does not walk the SDK over NFS.
ASCEND_HOME_PATH=$(/usr/bin/python3 -I -S - "$CANN_ROOT" <<'PY_CANN_VIEW'
import pathlib,shutil,sys,tempfile
source=pathlib.Path(sys.argv[1]).resolve()
view=pathlib.Path(tempfile.mkdtemp(prefix='glm53-cann-'))
for entry in source.iterdir():
    (view/entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
for name in ('ascend_toolkit_install.info', 'ascend_all_cann_install.info'):
    info = source/'aarch64-linux'/name
    if info.is_file():
        shutil.copyfile(info, view/name)
        break
else:
    raise FileNotFoundError('CANN installation metadata missing: '+str(source))
print(view)
PY_CANN_VIEW
)
export ASCEND_HOME_PATH
export PATH="$CONDA_ENV/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$DEPS_ROOT/slime-ascend/scripts/lib/glm53/runtime:$SLIME_ROOT:$DEPS_ROOT/Megatron-LM:$DEPS_ROOT/MegatronAdaptor:$DEPS_ROOT/TransformerEngineNPU:${GLM53_MEGATRON_BRIDGE_ROOT:-$DEPS_ROOT/Megatron-Bridge/src}:${SGLANG_PYTHONPATH:-$DEPS_ROOT/sglang/python}:${PYTHONPATH:-}"
export GLM53_NATIVE_ROOT="$DEPS_ROOT"
if [[ ${GLM53_LOGP_PARITY_ONLY:-0} == 1 || ${GLM53_REPLAY_SERVING_MOE_EP2:-0} == 1 ]]; then
    export PYTHONPATH="$GLM53_BI_DEPS/python:$GLM53_RUNTIME_DIR/../scripts/lib/glm53/runtime:$PYTHONPATH"
    export GLM53_NATIVE_HC_SHIM=1 SGLANG_NPU_HC_PRE_TAIL_GUARD=1
fi
# New native training workers use the validated CANN mHC path; torch remains an A/B override.
export GLM53_MHC_BACKEND="${GLM53_MHC_BACKEND:-ascendc}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy RAY_ADDRESS
export no_proxy="localhost,127.0.0.1,$RAY_NODE_IP" NO_PROXY="localhost,127.0.0.1,$RAY_NODE_IP"
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Keep optional Rust extensions from triggering a toolchain build at startup.
export SGLANG_RUST_BUILD_MODE=never
export TMS_HOOK_MODE TMS_PRESERVE_PRELOAD_MODE
export SLIME_ACCELERATOR USE_LIBUV SGLANG_MEMORY_SAVER_CUDA_GRAPH GLM53_RAY_SIGINT_NONE_COMPAT SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT SLIME_HOST_IP
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES
export HCCL_CONNECT_TIMEOUT HCCL_EXEC_TIMEOUT ACL_DEVICE_SYNC_TIMEOUT
export SGLANG_OPT_DEEPGEMM_HC_PRENORM SGLANG_OPT_USE_TILELANG_MHC_PRE SGLANG_OPT_USE_TILELANG_MHC_POST SGLANG_ENABLE_JIT_DEEPGEMM
export PYTHONDONTWRITEBYTECODE OMP_NUM_THREADS TOKENIZERS_PARALLELISM ASCEND_RT_VISIBLE_DEVICES
# Values are defined once in host.env; export them to Ray and its workers.
export GLOO_SOCKET_IFNAME HCCL_SOCKET_IFNAME
export HCCL_HOST_SOCKET_PORT_RANGE HCCL_NPU_SOCKET_PORT_RANGE
# This compatibility flag is injected into training workers via TRAIN_ENV_JSON only.
unset GLM53_TMS_POST7_RELEASE_COMPAT

glm53_submit() {
    if [[ ${1:-} == --help || ${1:-} == -h ]]; then
        echo 'Usage: bash lh_scripts/run_glm53_single_node.sh [--check]'; return
    fi
    if [[ ${1:-} == --check ]]; then
        "$CONDA_ENV/bin/python" - <<'PY_CHECK'
import os,importlib
for key in ('CONDA_PREFIX','PYTHONPATH','ASCEND_HOME_PATH','ASCEND_CUSTOM_OPP_PATH','GLOO_SOCKET_IFNAME','HCCL_SOCKET_IFNAME'):
    print(key+'='+os.environ.get(key,''),flush=True)
for name in ('torch','torch_npu','numpy','pyarrow','ray','slime','sglang','megatron','custom_ops'):
    m=importlib.import_module(name)
    print(name,getattr(m,'__version__',''),getattr(m,'__file__',''),flush=True)
print('Environment imports passed',flush=True)
PY_CHECK
        return
    fi
    [[ $# == 0 ]] || { echo 'Unexpected launcher arguments' >&2; return 2; }
    ulimit -c 0
    ulimit -n 65535
    mkdir -p "$LOG_ROOT" "/tmp/glm53-lh-$UID"
    RUN_DIR=$(mktemp -d "$LOG_ROOT/$(date +%Y%m%d_%H%M%S)_$(hostname -s)_XXXXXX")
    local run_id="$(basename "$RUN_DIR")"
    if [[ ${GLM53_STRICT_LOSS_AUDIT:-0} == 1 ]]; then
        export GLM53_PHASE_DIR="$RUN_DIR"
        if [[ ${GLM53_REPLAY_SERVING_MOE_EP2:-0} == 1 ]] && (( NUM_ROLLOUT <= 2 )); then
            ARGS+=(--save-debug-train-data "$RUN_DIR/train_data/{rollout_id}.pt")
        fi
    fi
    exec > >(tee -a "$RUN_DIR/stdout.log") 2>&1
    echo "[glm53] Run directory: $RUN_DIR"
    exec 9>"/tmp/glm53-lh-$UID/single-node.lock"
    flock -n 9 || { echo 'Another lh single-node run holds the lock'; return 75; }
    npu-smi info > "$RUN_DIR/npu-before.txt"
    if [[ ${GLM53_LOGP_PARITY_ONLY:-0} == 1 || ${GLM53_REPLAY_SERVING_MOE_EP2:-0} == 1 ]]; then
        if [[ $(grep -c 'No running processes' "$RUN_DIR/npu-before.txt") -ne "$GLM53_NUM_NPUS" ]]; then
            echo 'The eight-card parity recipe requires all eight NPUs to be free' >&2
            return 75
        fi
        export GLM53_PARITY_REPORT_DIR="$RUN_DIR"
        export GLM53_BATCH_REPORT="$RUN_DIR/batch-invariance.json"
    fi
    export TMPDIR=$(mktemp -d /tmp/g53lh.XXXXXX)
    export RAY_TMPDIR="$TMPDIR/ray" TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RUN_DIR/triton}" TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$RUN_DIR/torch-extensions}"
    export HF_MODULES_CACHE="$RUN_DIR/hf_modules"
    mkdir -p "$RAY_TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$HF_MODULES_CACHE"
    if [[ ${GLM53_PRELOAD_CANN_OPS:-0} == 1 ]]; then
        echo "[glm53] Preloading CANN operators before pipeline communication"
        "$CONDA_ENV/bin/python" -u - <<'PY_PRELOAD_CANN'
import json
import time
start = time.monotonic()
# Match worker imports before CANN computes its compiler include/library paths.
# Optional DeepEP imports can add vendor paths even when DeepEP is unavailable.
from slime.backends.megatron_utils.actor import MegatronTrainRayActor  # noqa: F401
import cann_ops_transformer
ops = cann_ops_transformer.ops
required = (
    'mhc_pre_sinkhorn', 'mhc_post', 'mhc_pre_sinkhorn_backward', 'mhc_post_backward',
    'key_pool', 'pool_key_indexer', 'sparse_flash_mla_metadata', 'sparse_flash_mla',
    'sparse_flash_mla_grad_metadata', 'sparse_flash_mla_grad',
)
missing = [name for name in required if not callable(getattr(ops, name, None))]
if missing:
    raise RuntimeError(f'CANN operators missing: {missing}')
# Compile the active entry-point providers, preserving installed custom overrides.
import importlib
precompiled = []
for operator, builder_name in (
    ('sparse_flash_mla', 'sparse_flash_mla_op_builder'),
    ('sparse_flash_mla_grad', 'smlag_op_builder'),
):
    provider = importlib.import_module(getattr(ops, operator).__module__)
    builder = getattr(provider, builder_name)
    print('GLM53_CANN_PRELOAD_BUILD ' + builder.name, flush=True)
    builder.load()
    precompiled.append(builder.name)
print('GLM53_CANN_PRELOAD_READY ' + json.dumps({
    'seconds': round(time.monotonic() - start, 3),
    'package': cann_ops_transformer.__file__,
    'precompiled': precompiled,
    'operators': {name: getattr(ops, name).__module__ for name in required},
}), flush=True)
PY_PRELOAD_CANN
    fi
    printf '%q ' "$CONDA_ENV/bin/python" -u "$SLIME_ROOT/train.py" "${ARGS[@]}" > "$RUN_DIR/command.sh"
    printf '\n' >> "$RUN_DIR/command.sh"
    "$CONDA_ENV/bin/python" - <<'PY_ENV' > "$RUN_DIR/runtime-env.json"
import os,json
keys={'PATH','PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD','SLIME_ACCELERATOR','SLIME_HOST_IP','CUDA_DEVICE_MAX_CONNECTIONS','ACL_DEVICE_SYNC_TIMEOUT','USE_LIBUV','no_proxy','NO_PROXY','OMP_NUM_THREADS','TOKENIZERS_PARALLELISM','PYTHONDONTWRITEBYTECODE','PYTHONNOUSERSITE','TORCH_DEVICE_BACKEND_AUTOLOAD','TMPDIR','HF_MODULES_CACHE','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR'}
prefix=('ASCEND','HCCL','GLOO','RAY_','PYTORCH_','SGLANG_','TMS_','DEEPEP_','CANN','ATB','FLA_','GLM53_')
print(json.dumps({'worker_process_setup_hook':'glm53_ctor_setup_v5.install','env_vars':{k:v for k,v in os.environ.items() if k in keys or k.startswith(prefix)}}))
PY_ENV
    # Ray workers inherit the head process working directory.
    cd "$SLIME_ROOT"
    setsid "$CONDA_ENV/bin/ray" start --head --block --node-ip-address="$RAY_NODE_IP" --port="$RAY_GCS_PORT" \
        --dashboard-host=127.0.0.1 --dashboard-port="$RAY_DASHBOARD_PORT" --object-manager-port="$RAY_OBJECT_MANAGER_PORT" --node-manager-port="$RAY_NODE_MANAGER_PORT" --ray-client-server-port="$RAY_CLIENT_SERVER_PORT" --dashboard-agent-grpc-port="$RAY_DASHBOARD_AGENT_GRPC_PORT" --dashboard-agent-listen-port="$RAY_DASHBOARD_AGENT_LISTEN_PORT" --runtime-env-agent-port="$RAY_RUNTIME_ENV_AGENT_PORT" --metrics-export-port="$RAY_METRICS_EXPORT_PORT" --min-worker-port="$RAY_WORKER_MIN_PORT" --max-worker-port="$RAY_WORKER_MAX_PORT" --num-cpus=48 --resources="{\"NPU\":$GLM53_NUM_NPUS}" \
        --temp-dir="$RAY_TMPDIR" --disable-usage-stats > "$RUN_DIR/ray-start.log" 2>&1 &
    local ray_pid=$!
    trap 'kill -TERM -- -'"$ray_pid"' 2>/dev/null || true; sleep 3; kill -KILL -- -'"$ray_pid"' 2>/dev/null || true' EXIT
    local ready=0
    for ((i=0; i<300; i++)); do
        # A 401/403 still means the dashboard is serving: on hosts with Ray token
        # auth enabled we must not spin here for the whole budget. Submit will
        # then fail fast with the real authentication error instead.
        if /usr/bin/python3 -I -S -c '
import sys, urllib.request, urllib.error
try:
    urllib.request.build_opener(urllib.request.ProxyHandler({})).open(sys.argv[1], timeout=3).close()
except urllib.error.HTTPError as exc:
    sys.exit(0 if exc.code in (401, 403) else 1)
except Exception:
    sys.exit(1)
' "http://127.0.0.1:$RAY_DASHBOARD_PORT/api/jobs/" >/dev/null 2>&1; then ready=1; break; fi
        kill -0 "$ray_pid" 2>/dev/null || { cat "$RUN_DIR/ray-start.log"; return 1; }
        sleep 2
    done
    (( ready )) || { cat "$RUN_DIR/ray-start.log"; return 1; }
    "$CONDA_ENV/bin/ray" job submit --submission-id="lh-$run_id" --address="http://127.0.0.1:$RAY_DASHBOARD_PORT" \
        --runtime-env-json="$(cat "$RUN_DIR/runtime-env.json")" -- bash "$RUN_DIR/command.sh"
}
