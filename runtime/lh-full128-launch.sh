#!/usr/bin/env bash
# Lifecycle for the current 8-container / 8-physical-host allocation.

glm53_full128_activate() {
    # Node-local values stay in the raylet environment, never in driver env_vars.
    export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DIES"
    export SLIME_ACCELERATOR=npu GLM53_NATIVE_ROOT="$DEPS_ROOT"
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0 FLA_NPU_DISABLE_PTH=1 PYTHONNOUSERSITE=1
    export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD ASCEND_CUSTOM_OPP_PATH ASCEND_OPP_PATH
    unset ASCEND_AICPU_PATH ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_CANN_PACKAGE_PATH
    unset ASCEND_HOME ASCEND_ATB_HOME ATB_HOME_PATH ASDSIP_HOME_PATH
    unset SSH_CLIENT SSH_CONNECTION SSH_TTY RAY_ADDRESS
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
    set +u
    source "$DEPS_ROOT/activate.sh"
    source "$CONDA_ENV/lib/python3.12/site-packages/fla_npu/opp/vendors/fla_npu_transformer/bin/set_env.bash"
    set -u
    # A shallow local SDK view avoids torch_npu recursively scanning CANN on NFS.
    local cann_source=${ASCEND_HOME_PATH:?CANN activation did not set ASCEND_HOME_PATH}
    export ASCEND_HOME_PATH
    ASCEND_HOME_PATH=$(/usr/bin/python3 -I -S - "$cann_source" "$TMPDIR/cann" <<'PY_CANN'
from pathlib import Path
import shutil, sys
source, view = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
view.mkdir()
for p in source.iterdir():
    (view/p.name).symlink_to(p, target_is_directory=p.is_dir())
for name in ('ascend_toolkit_install.info','ascend_all_cann_install.info'):
    if (source/'aarch64-linux'/name).is_file():
        if not (view/name).exists():
            shutil.copyfile(source/'aarch64-linux'/name,view/name)
        break
else:
    assert (view/'ascend_toolkit_install.info').is_file(), 'Missing CANN metadata'
print(view)
PY_CANN
)
    export PATH="$CONDA_ENV/bin:$PATH" CONDA_PREFIX="$CONDA_ENV"
    export LD_LIBRARY_PATH="$CONDA_ENV/lib:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$REPO_ROOT/scripts/lib/glm53/runtime:$SLIME_ROOT:$DEPS_ROOT/Megatron-LM:$DEPS_ROOT/MegatronAdaptor:$DEPS_ROOT/TransformerEngineNPU:$DEPS_ROOT/Megatron-Bridge/src:$DEPS_ROOT/sglang/python:${PYTHONPATH:-}"
    export GLOO_SOCKET_IFNAME="$COMM_IFNAME" HCCL_SOCKET_IFNAME="$COMM_IFNAME" SLIME_HOST_IP="$NODE_IP"
    local selected
    selected=$(/usr/bin/python3 -I -S -c 'import json,sys; print(",".join(n["ip"] for n in json.load(open(sys.argv[1]))["nodes"]))' "$FULL128_TOPOLOGY")
    export no_proxy="localhost,127.0.0.1,$selected" NO_PROXY="localhost,127.0.0.1,$selected"
    export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
    export CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
    export USE_LIBUV=${GLM53_TCPSTORE_USE_LIBUV:-1}
    [[ $USE_LIBUV == 0 || $USE_LIBUV == 1 ]] || return 2
    export HCCL_CONNECT_TIMEOUT=900 HCCL_EXEC_TIMEOUT=1800 ACL_DEVICE_SYNC_TIMEOUT=1800
    export HCCL_BUFFSIZE=256 DEEPEP_HCCL_BUFFSIZE=256
    export HCCL_HOST_SOCKET_PORT_RANGE=60000-60099 HCCL_NPU_SOCKET_PORT_RANGE=61000-61099
    export TASK_QUEUE_ENABLE=0 PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
    export TMS_HOOK_MODE=preload TMS_PRESERVE_PRELOAD_MODE=1
    export SGLANG_MEMORY_SAVER_CUDA_GRAPH=true SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1
    export SGLANG_ENABLE_JIT_DEEPGEMM=False SGLANG_RUST_BUILD_MODE=never
    export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0
    export ASCEND_USE_FIA=1 GLM53_R3_DIRECT_WORKERS=1 GLM53_RAY_SIGINT_NONE_COMPAT=1
    export GLM53_KPOOL_BACKEND=ascendc GLM53_MHC_BACKEND=ascendc
    export GLM53_SWIGLU_BACKEND=ascendc GLM53_CONV_BACKEND=ascendc GLM53_KDA_SEQUENCE_MODE=packed
    export GLM53_KDA_FWD_HEAD_CHUNK_SIZE=${GLM53_KDA_FWD_HEAD_CHUNK_SIZE:-1}
    export GLM53_KDA_BWD_HEAD_CHUNK_SIZE=${GLM53_KDA_BWD_HEAD_CHUNK_SIZE:-64}
    export GLM53_MOE_DISPATCH_CHUNK_SIZE=512 GLM53_EXPERT_TOKEN_CHUNK_SIZE=1024
    # The compatibility option is training-only, as in the supplied single-node recipe.
    unset GLM53_TMS_POST7_RELEASE_COMPAT
}

glm53_full128_usage() {
    cat <<'USAGE'
Usage: run this same entrypoint on ALL EIGHT containers with the SAME new run ID.
  bash lh_scripts/run_glm53_8host_full288.sh --check-config
  bash lh_scripts/run_glm53_8host_full288.sh --check
  bash lh_scripts/run_glm53_8host_full288.sh --check-env
  bash lh_scripts/run_glm53_8host_full288.sh --print-command
  bash lh_scripts/run_glm53_8host_full288.sh --run-id 20261105_full288_smoke1
Defaults: full 45L/288E, 128 dies, TP1/PP4/CP8/EP32, GB16, 16K context, 8K response,
one rollout/update; two TP64 text-only rollout engines. No checkpoints are written.
PROFILE=long selects GB128, 64K context, 32K response and 1000 rollouts, also full288.
--check-config validates topology, local devices and arguments without weight/data files.
--check is static and also requires complete weight/data files. --check-env additionally validates native framework CLI parsing;
neither starts Ray, model loading or training. An empty invocation prints this help.
The actual run verifies every selected Ray NPU ID before model initialization.
Stop: Ctrl-C on the head (.97); peers then stop their own run-tagged processes.
Never use ray stop globally. Logs: <project>/cache/work/glm53-full128/<run-id>.
USAGE
}

glm53_full128_submit() {
    local mode=run run_id='' arg
    (( $# )) || { glm53_full128_usage; return 0; }
    while (( $# )); do
        arg=$1; shift
        case "$arg" in
            --help|-h) glm53_full128_usage; return 0 ;;
            --check-config|--check|--check-env|--print-command)
                [[ $mode == run && -z $run_id ]] || return 2; mode=${arg#--} ;;
            --run-id) [[ $# -gt 0 && $mode == run ]] || return 2; run_id=$1; shift ;;
            *) echo "Unknown argument: $arg" >&2; return 2 ;;
        esac
    done
    if [[ $mode == print-command ]]; then
        printf '%q ' "$CONDA_ENV/bin/python" -u "$SLIME_ROOT/train.py" "${ARGS[@]}"
        printf '\n'; return 0
    fi
    if [[ $mode == check-config ]]; then
        /usr/bin/python3 -I -S "$FULL128_HELPER" check-config --topology "$FULL128_TOPOLOGY" -- "${ARGS[@]}"
        /usr/bin/python3 -I -S "$FULL128_HELPER" node --topology "$FULL128_TOPOLOGY"
        return
    fi
    /usr/bin/python3 -I -S "$FULL128_HELPER" check --topology "$FULL128_TOPOLOGY" \
        --checkpoint "$HF_CHECKPOINT" --data "$PROMPT_DATA" -- "${ARGS[@]}"
    [[ $mode != check ]] || return 0
    if [[ $mode == run ]]; then
        [[ $run_id =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$ ]] || { echo 'Supply a NEW --run-id NAME' >&2; return 2; }
        grep -Fq 'node_idx = address' "$SLIME_ROOT/slime/backends/sglang_utils/engine_group.py" || {
            echo 'Required mixed-container SGLang port fix is missing' >&2; return 2;
        }
    fi
    # Does not depend on the iproute2 package missing in these containers.
    local local_env
    local_env=$(/usr/bin/python3 -I -S "$FULL128_HELPER" node --topology "$FULL128_TOPOLOGY")
    eval "$local_env"
    export TMPDIR
    TMPDIR=$(mktemp -d /tmp/g53f128.XXXXXX)
    if [[ $mode == check-env ]]; then
        glm53_full128_activate
        cd "$SLIME_ROOT"
        "$CONDA_ENV/bin/python" "$FULL128_HELPER" native --topology "$FULL128_TOPOLOGY" -- "${ARGS[@]}"
        return
    fi
    export GLM53_FULL128_RUN_ID="$run_id"
    local cluster_dir="$LOG_ROOT/$run_id" node_log="$LOG_ROOT/$run_id/node-$NODE_IP"
    mkdir -p "$cluster_dir"
    mkdir "$node_log" || { echo 'This run ID was already used on this container' >&2; return 2; }
    exec 9>"/tmp/glm53-full128-$UID.lock"
    flock -n 9 || { echo 'Another full128 run owns this container' >&2; return 75; }
    exec > >(tee -a "$node_log/launcher.log") 2>&1
    # Trap captures values; cleanup can only signal matching run-token processes.
    trap 'rc=$?; trap - EXIT; if [[ '"$NODE_IP"' == '"$HEAD_IP"' ]]; then printf "%s\n" "$rc" > '"\"$cluster_dir/head-exit-code\""'; elif (( rc )); then printf "{\"exit_code\":%s}\n" "$rc" > '"\"$cluster_dir/failed-$NODE_IP.json\""'; fi; /usr/bin/python3 -I -S '"\"$FULL128_HELPER\""' cleanup --topology '"\"$FULL128_TOPOLOGY\""' --token '"\"$run_id\""' > '"\"$node_log/cleanup.json\""' 2>&1 || true; exit "$rc"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    [[ ! -e "$cluster_dir/head-exit-code" ]] || { echo 'Run ID has completed'; return 2; }
    /usr/bin/python3 -I -S "$FULL128_HELPER" register --topology "$FULL128_TOPOLOGY" --root "$cluster_dir" -- "${ARGS[@]}"
    glm53_full128_activate
    /usr/bin/python3 -I -S "$FULL128_HELPER" idle --topology "$FULL128_TOPOLOGY" --root "$node_log/npu-before.txt"
    ulimit -c 0
    ulimit -n 65535
    export RAY_TMPDIR="$TMPDIR/ray" TRITON_CACHE_DIR="$node_log/triton"
    export TORCH_EXTENSIONS_DIR="$node_log/torch-extensions" HF_MODULES_CACHE="$node_log/hf_modules"
    mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$HF_MODULES_CACHE"
    cd "$SLIME_ROOT"
    local gcs_port=25400 dashboard_port=25401 i ready=0
    local -a ray_args=(start --block --node-ip-address="$NODE_IP" --num-cpus=48
        --resources="{\"NPU\":$NODE_DIES}" --disable-usage-stats
        --object-manager-port=25402 --node-manager-port=25403
        --dashboard-agent-grpc-port=25405 --dashboard-agent-listen-port=25406
        --runtime-env-agent-port=25407 --metrics-export-port=25408
        --min-worker-port=28600 --max-worker-port=28999)
    if [[ $NODE_IP == "$HEAD_IP" ]]; then
        ray_args+=(--head --port="$gcs_port" --dashboard-host="$HEAD_IP"
            --dashboard-port="$dashboard_port" --ray-client-server-port=25404 --temp-dir="$RAY_TMPDIR")
    else
        for ((i=0;i<450;i++)); do
            [[ ! -e "$cluster_dir/head-exit-code" ]] || return 1
            if env -u LD_LIBRARY_PATH /usr/bin/curl --noproxy '*' --max-time 2 -fsS \
                "http://$HEAD_IP:$dashboard_port/api/jobs/" >/dev/null 2>&1; then ready=1; break; fi
            sleep 2
        done
        (( ready )) || { echo 'Head dashboard not ready'; return 1; }
        ray_args+=(--address="$HEAD_IP:$gcs_port")
    fi
    setsid "$CONDA_ENV/bin/ray" "${ray_args[@]}" > "$node_log/ray-start.log" 2>&1 &
    local ray_pid=$!
    printf '%s\n' "$ray_pid" > "$node_log/ray.pid"
    if [[ $NODE_IP != "$HEAD_IP" ]]; then
        while [[ ! -e "$cluster_dir/head-exit-code" ]]; do
            kill -0 "$ray_pid" 2>/dev/null || { tail -50 "$node_log/ray-start.log"; return 1; }
            sleep 3
        done
        return "$(cat "$cluster_dir/head-exit-code")"
    fi
    ready=0
    for ((i=0;i<300;i++)); do
        if env -u LD_LIBRARY_PATH /usr/bin/curl --noproxy '*' --max-time 2 -fsS \
            "http://$HEAD_IP:$dashboard_port/api/jobs/" >/dev/null 2>&1; then ready=1; break; fi
        kill -0 "$ray_pid" 2>/dev/null || { tail -50 "$node_log/ray-start.log"; return 1; }
        sleep 2
    done
    (( ready )) || return 1
    "$CONDA_ENV/bin/python" "$FULL128_HELPER" ready --topology "$FULL128_TOPOLOGY" \
        --address "$HEAD_IP:$gcs_port" --root "$cluster_dir"
    printf '%q ' "$CONDA_ENV/bin/python" -u "$SLIME_ROOT/train.py" "${ARGS[@]}" > "$cluster_dir/command.sh"
    printf '\n' >> "$cluster_dir/command.sh"
    # Inherit node-local visible IDs, IPs, SDK view and cache paths from each raylet.
    # Forwarding head env_vars here would bind every worker to .97's devices/IP.
    "$CONDA_ENV/bin/ray" job submit --submission-id="glm53-full128-$run_id" \
        --address="http://$HEAD_IP:$dashboard_port" \
        --runtime-env-json='{"worker_process_setup_hook":"glm53_ctor_setup_v5.install"}' \
        -- bash "$cluster_dir/command.sh" &
    local submit_pid=$!
    while kill -0 "$submit_pid" 2>/dev/null; do
        if compgen -G "$cluster_dir/failed-*.json" >/dev/null; then
            echo 'A peer launcher failed; ending this run'
            return 1
        fi
        sleep 3
    done
    wait "$submit_pid"
}
