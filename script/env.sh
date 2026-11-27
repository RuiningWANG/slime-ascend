#!/usr/bin/env bash
# Function definitions only; inspection modes do not activate CANN or import Torch.
GLM53_EIGHT_COMMON_ENV=(
    SLIME_ACCELERATOR=npu                         # Slime accelerator selection.
    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7     # The eight physical NPUs per node.
    RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1  # Slime assigns NPU IDs.
    RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0          # Keep IDs for zero-GPU Ray actors.
    USE_LIBUV=1                                  # Multiplexed TCPStore; CPU compatibility probe passed.
    HCCL_BUFFSIZE=256                            # HCCL communicator buffer, MiB.
    HCCL_HOST_SOCKET_PORT_RANGE=9000-9099          # Rollout host ports; train uses 9200.
    HCCL_NPU_SOCKET_PORT_RANGE=9100-9199           # Rollout NPU ports; train uses 9300.
    HCCL_CONNECT_TIMEOUT=3600                     # Seconds, including first-step JIT.
    HCCL_EXEC_TIMEOUT=3600
    ACL_DEVICE_SYNC_TIMEOUT=3600
    TASK_QUEUE_ENABLE=0                          # Ascend host task queue, same as baseline.
    PYTORCH_NPU_ALLOC_CONF=expandable_segments:False # Existing TMS/offload allocation mode.
    TMS_HOOK_MODE=preload                         # Native allocator interception mode.
    TMS_PRESERVE_PRELOAD_MODE=1                   # Do not silently switch to torch hooks.
    SGLANG_MEMORY_SAVER_CUDA_GRAPH=true           # Manage graph pools through TMS on NPU.
    SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1        # Keep transferable weights in ND format.
    ASCEND_USE_FIA=1                             # Ascend fused attention path.
    SGLANG_ENABLE_JIT_DEEPGEMM=False              # Do not build CUDA DeepGEMM on NPU.
    SGLANG_OPT_DEEPGEMM_HC_PRENORM=0              # Keep the previous native mHC selection.
    SGLANG_OPT_USE_TILELANG_MHC_PRE=0
    SGLANG_OPT_USE_TILELANG_MHC_POST=0
    SGLANG_RUST_BUILD_MODE=never                  # Use installed Rust extension, no build.
    GLM53_R3_DIRECT_WORKERS=1                     # Preserve routed_experts at the gateway.
    GLM53_RAY_SIGINT_NONE_COMPAT=1                # Private Ray source compatibility.
    TORCH_DEVICE_BACKEND_AUTOLOAD=0               # SDK setup precedes torch_npu import.
    FLA_NPU_DISABLE_PTH=1                        # Explicit FLA vendor activation below.
    PYTHONNOUSERSITE=1                            # Avoid mixing user-site packages.
    PYTHONDONTWRITEBYTECODE=1                    # Avoid concurrent writes to NFS source.
    OMP_NUM_THREADS=4                            # Bound CPU threads per worker.
    TOKENIZERS_PARALLELISM=false                  # Avoid nested tokenizer thread pools.
    CUDA_DEVICE_MAX_CONNECTIONS=1                # Existing Megatron ordering setting.
)

glm53_eight_activate() {
    local item
    for item in "${GLM53_EIGHT_COMMON_ENV[@]}"; do export "$item"; done
    export GLOO_SOCKET_IFNAME="$FULL_COMM_IFNAME" HCCL_SOCKET_IFNAME="$FULL_COMM_IFNAME"
    export SLIME_HOST_IP="$node_ip"
    export no_proxy="localhost,127.0.0.1,$(IFS=,; echo "${NODE_IPS[*]}")"
    export NO_PROXY="$no_proxy"
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy RAY_ADDRESS
    unset SSH_CLIENT SSH_CONNECTION SSH_TTY
    unset GLM53_TMS_TEST_PAGEABLE GLM53_TMS_POST7_RELEASE_COMPAT
    unset GLM53_UPDATE_PARITY_PROBE GLM53_PARITY_ARM GLM53_TRACE_DIR
    unset PYTHONSTARTUP
    unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD ASCEND_CUSTOM_OPP_PATH ASCEND_OPP_PATH
    unset ASCEND_AICPU_PATH ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_CANN_PACKAGE_PATH
    unset ASCEND_HOME ASCEND_ATB_HOME ATB_HOME_PATH ASDSIP_HOME_PATH
    set +u
    # Activate the pinned release binaries, then point every source path to this delivery.
    source "$DEPS_ROOT/activate.sh"
    source "$CODE_ROOT/dependencies/batch-invariant/opp/vendors/batch_invariant/bin/set_env.bash"
    set -u
    export WORK_ROOT="$CODE_ROOT" SRC="$CODE_ROOT/src"
    export SLIME_ROOT="$SRC/slime-ascend/slime" MEGATRON_PATH="$SRC/Megatron-LM"
    export SLIME_ASCEND_DEPS_ROOT="$SRC" GLM53_NATIVE_ROOT="$SRC"
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
    export TORCH_EXTENSIONS_DIR="$node_log/torch-extensions" PIP_CACHE_DIR="$node_log/pip"
    export PIP_CONSTRAINT="$CODE_ROOT/constraints.txt"
    export SGLANG_NPU_HC_PRE_TAIL_GUARD=1 SLIME_DESTROY_WORLD_PROCESS_GROUP=0
    /usr/bin/python3 -I -S - "$CANN_ROOT" "$TMPDIR/cann" <<'PY_CANN'
import pathlib, shutil, sys
source, view = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2])
view.mkdir()
for entry in source.iterdir():
    (view / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
shutil.copyfile(source / 'aarch64-linux/ascend_toolkit_install.info', view / 'ascend_toolkit_install.info')
PY_CANN
    export ASCEND_HOME_PATH="$TMPDIR/cann"

    # Optional reuse of a previous run's completed C++ artifacts and SDK view.
    # The source/ABI/dependency checks remain PyTorch/Ninja's responsibility.
    if [[ -n ${GLM53_TORCH_EXTENSIONS_REUSE_RUN:-} ]]; then
        local reuse="$GLM53_TORCH_EXTENSIONS_REUSE_RUN" reuse_node reuse_tmp
        [[ $reuse =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$ ]] || { echo 'Invalid C++ cache source run ID' >&2; return 2; }
        reuse_node="$CODE_ROOT/all_logs/$reuse/$(basename -- "$node_log")"
        reuse_tmp=$(/usr/bin/python3 -I -S - "$reuse" <<'PY_REUSE'
import hashlib,os,sys
print("/tmp/g53e-"+str(os.getuid())+"/"+hashlib.sha256(sys.argv[1].encode()).hexdigest()[:12]+"/cann")
PY_REUSE
)
        if [[ -s "$CODE_ROOT/all_logs/$reuse/head-exit-code" ]] &&
           [[ $(cat "$CODE_ROOT/all_logs/$reuse/head-exit-code") == 0 ]] &&
           [[ -d "$reuse_node/torch-extensions" && -f "$reuse_tmp/ascend_toolkit_install.info" ]] &&
           [[ "$reuse_tmp/aarch64-linux" -ef "$CANN_ROOT/aarch64-linux" ]] &&
           [[ -z $(find "$reuse_node/torch-extensions" -name lock -print -quit) ]]; then
            export TORCH_EXTENSIONS_DIR="$reuse_node/torch-extensions"
            export ASCEND_HOME_PATH="$reuse_tmp"
            echo "[glm53-8node] Reusing validated C++ build cache: $TORCH_EXTENSIONS_DIR"
        else
            echo "[glm53-8node] C++ source cache unavailable; retaining this run's isolated cache"
        fi
    fi
    export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
    local dep="$CODE_ROOT/dependencies"
    # Complete source trees, not import overlays. SDK Python packages are retained.
    export PYTHONPATH="$CODE_ROOT:$SLIME_ROOT:$SRC/Megatron-LM:$SRC/MegatronAdaptor:$SRC/TransformerEngineNPU:$SRC/sglang/python:$dep/ray-runtime:$dep/torch-memory-saver:$dep/batch-invariant/python:$CONDA_ENV/lib/python3.12/site-packages:$CANN_ROOT/python/site-packages:$CANN_ROOT/opp/built-in/op_impl/ai_core/tbe"
    export GLM53_ZERO_MISMATCH=1 GLM53_RAY_SIGINT_NONE_COMPAT=1
    export HCCL_OP_EXPANSION_MODE=AICPU_TS
    unset GLM53_TMS_TEST_PAGEABLE SGLANG_RETURN_ORIGINAL_LOGPROB
}
