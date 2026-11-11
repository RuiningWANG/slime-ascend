#!/usr/bin/env bash
# Defaults match the arguments in A5-1k-2k-geo3k.log (2026-11-01).
# Current allocation: eight hosts x sixteen visible Ascend dies = 128; full 45L/288E GLM.
# The log records 2048 prompt tokens despite the "1k" in its filename.
set -euo pipefail
REPO_ROOT="/mnt/share/l00937981/glm_53_flash_slime/slime-ascend"
DEPS_ROOT="$(dirname "$REPO_ROOT")"
CONDA_ENV=/mnt/share/l00937981/conda/envs/glm53-flash-slime
SLIME_ROOT="$REPO_ROOT/slime"
LOG_ROOT=${LOG_ROOT:-/mnt/share/l00937981/glm_53_flash_slime/cache/work/geo3k8-aligned-A5}
HF_CHECKPOINT=${HF_CHECKPOINT:-/mnt/share/l00804615/weight/GLM-5.3-Flash-BF16}
# Current local Geo3K training split uses problem/answer fields (2101 records).
PROMPT_DATA=${PROMPT_DATA:-/mnt/share/l00937981/glm_53_flash_slime/cache/datasets/geo3k_imgurl/train-absolute.jsonl}
export GLM53_NIGHT_OWNER=geo3k8_20261110
NUM_EXPERTS=288
CONTEXT_LENGTH=${CONTEXT_LENGTH:-4096}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-2048}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
NUM_ROLLOUT=${NUM_ROLLOUT:-1000}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE*N_SAMPLES_PER_PROMPT))}
EP_SIZE=${EP_SIZE:-32}
CP_SIZE=${CP_SIZE:-1}
INFER_TP=${INFER_TP:-16}
RUN_ID=inspection
for ((i=1; i<=$#; i++)); do
    if [[ ${!i} == --run-id ]]; then j=$((i+1)); RUN_ID=${!j:-}; fi
done
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
    --num-experts "$NUM_EXPERTS"
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

TRAIN_ENV_JSON='{"HCCL_HOST_SOCKET_PORT_RANGE":"9200-9299","HCCL_NPU_SOCKET_PORT_RANGE":"9300-9399","GLM53_MOE_DISPATCH_CHUNK_SIZE":"512","GLM53_EXPERT_TOKEN_CHUNK_SIZE":"1024","TASK_QUEUE_ENABLE":"0","GLM53_CONV_BACKEND":"ascendc","GLM53_MHC_BACKEND":"ascendc","GLM53_KPOOL_BACKEND":"ascendc","GLM53_SWIGLU_BACKEND":"ascendc","GLM53_KDA_SEQUENCE_MODE":"packed"}'
# Eight containers expose sixteen dies each; one TP16 engine runs on each host.
ARGS=(
    --actor-num-nodes 8 --actor-num-gpus-per-node 16 --num-gpus-per-node 16
    --rollout-num-gpus 128 --colocate --train-env-vars "$TRAIN_ENV_JSON"
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "$HF_CHECKPOINT" --load "$HF_CHECKPOINT"
    --prompt-data "$PROMPT_DATA" --input-key problem --label-key answer --apply-chat-template
    --multimodal-keys '{"image":"images"}'
    --apply-chat-template-kwargs '{}'
    --rm-type deepscaler --seed 522
    --num-rollout "$NUM_ROLLOUT"
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE" --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
    --global-batch-size "$GLOBAL_BATCH_SIZE" --micro-batch-size 1
    --rollout-max-prompt-len 2048 --rollout-max-response-len "$RESPONSE_LENGTH" --rollout-max-context-len "$CONTEXT_LENGTH"
    --rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1 --rollout-seed 42
    --advantage-estimator grpo --pg-loss-type reinforce
    --entropy-coef 0 --kl-coef 0 --kl-loss-coef 0 --eps-clip 0.2 --eps-clip-high 0.2
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 4
    --decoder-first-pipeline-num-layers 12 --decoder-last-pipeline-num-layers 11
    --context-parallel-size "$CP_SIZE" --expert-model-parallel-size "$EP_SIZE"
    --expert-tensor-parallel-size 1 --moe-token-dispatcher-type alltoall
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --seq-length "$CONTEXT_LENGTH" --max-position-embeddings 65536
    --use-dynamic-batch-size --max-tokens-per-gpu 4096 --data-pad-size-multiplier 128
    --log-probs-max-tokens-per-gpu 4096 --log-probs-chunk-size 256
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
    --adam-beta1 0.9 --adam-beta2 0.98 --adam-eps 1e-8 --clip-grad 1.0
    --no-load-optim --no-load-rng --no-save-optim --freeze-params-name-list 'model.visual.*'
    --optimizer-cpu-offload --optimizer-offload-fraction 1.0 --use-precision-aware-optimizer
    --use-distributed-optimizer --use-layer-wise-param-layout
    --disable-param-buffers-cpu-backup --disable-grad-buffers-cpu-backup
    --overlap-cpu-optimizer-d2h-h2d --no-pin-cpu-grads --no-pin-cpu-params
    --rollout-num-gpus-per-engine "$INFER_TP"
    --sglang-mm-enable-dp-encoder --sglang-device npu --sglang-attention-backend ascend
    --sglang-tp-size "$INFER_TP" --sglang-dp-size 1 --sglang-ep-size 1
    --sglang-disable-shared-experts-fusion
    --sglang-weight-loader-prefetch-checkpoints --sglang-weight-loader-prefetch-num-threads 1
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.75}" --sglang-context-length "$CONTEXT_LENGTH"
    --sglang-max-running-requests 64 --sglang-chunked-prefill-size 4096
    --sglang-disable-overlap-schedule --sglang-cuda-graph-bs 1 2 4 8 16 24 32 64
    --sglang-page-size 64 --sglang-kv-cache-dtype bfloat16 --sglang-watchdog-timeout 1200
    --sglang-server-concurrency 512
    --attention-dropout 0 --hidden-dropout 0 --bf16
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
    --rollout-health-check-first-wait 600 --rollout-health-check-interval 600 --rollout-health-check-timeout 600
    --distributed-timeout-minutes 60 --update-weight-buffer-size "${UPDATE_WEIGHT_BUFFER_SIZE:-268435456}"
    --update-weight-mode full --update-weight-transport nccl --check-weight-update-equal
    --rollout-data-transport object-store
    --use-rollout-routing-replay --routing-replay-prefetch-microbatches 1
    # TIS is disabled in the reference log; retain its inactive clip defaults.
    --tis-clip-low 0 --tis-clip 2.0
)
glm53_dashboard_ready() {
    /usr/bin/python3 -I -S - "$1" "$2" <<'PY_READY'
import os, sys, urllib.request
from pathlib import Path
try:
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    token=Path(os.environ['RAY_AUTH_TOKEN_PATH']).read_text().strip()
    req=urllib.request.Request('http://'+sys.argv[1]+':'+sys.argv[2]+'/api/jobs/',headers={'Authorization':'Bearer '+token})
    with opener.open(req,timeout=2) as r:
        assert r.status == 200
except Exception:
    sys.exit(1)
PY_READY
}

#!/usr/bin/env bash
# Lifecycle for the current 8-container / 8-physical-host allocation.

glm53_full128_activate() {
    # Node-local values stay in the raylet environment, never in driver env_vars.
    export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DIES"
    export TORCH_GLOO_LAZY_INIT=1 USE_LIBUV=0
    export SLIME_ACCELERATOR=npu GLM53_NATIVE_ROOT="$DEPS_ROOT"
    export RAY_AUTH_MODE=token RAY_AUTH_TOKEN_PATH="$REPO_ROOT/runtime/geo3k8-ray-auth-token"
    # The shared filesystem has ample absolute free space despite high utilization.
    export RAY_local_fs_capacity_threshold=${RAY_local_fs_capacity_threshold:-0.999}
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
    selected=80.5.27.199,80.5.27.200,80.5.27.202,80.5.27.203,80.5.27.204,80.5.27.205,80.5.27.206,80.5.27.207
    export no_proxy="localhost,127.0.0.1,$selected" NO_PROXY="localhost,127.0.0.1,$selected"
    export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
    export CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
    export USE_LIBUV=${GLM53_TCPSTORE_USE_LIBUV:-0}
    export SLIME_DESTROY_WORLD_PROCESS_GROUP=0
    [[ $USE_LIBUV == 0 || $USE_LIBUV == 1 ]] || return 2
    export HCCL_CONNECT_TIMEOUT=900 HCCL_EXEC_TIMEOUT=1800 ACL_DEVICE_SYNC_TIMEOUT=1800
    export HCCL_BUFFSIZE=256 DEEPEP_HCCL_BUFFSIZE=256
    export HCCL_HOST_SOCKET_PORT_RANGE=60000-60099 HCCL_NPU_SOCKET_PORT_RANGE=61000-61099
    export TASK_QUEUE_ENABLE=0 PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
    export TMS_HOOK_MODE=preload TMS_PRESERVE_PRELOAD_MODE=1
    export SGLANG_MEMORY_SAVER_CUDA_GRAPH=true SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1
    export SGLANG_NPU_PRE_OFFLOAD_FENCE=1
    export SGLANG_ENABLE_JIT_DEEPGEMM=False SGLANG_RUST_BUILD_MODE=never
    export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0
    export ASCEND_USE_FIA=1 GLM53_R3_DIRECT_WORKERS=1 GLM53_RAY_SIGINT_NONE_COMPAT=1
    export GLM53_KPOOL_BACKEND=ascendc GLM53_MHC_BACKEND=ascendc
    export GLM53_SWIGLU_BACKEND=ascendc GLM53_CONV_BACKEND=ascendc GLM53_KDA_SEQUENCE_MODE=packed
    export GLM53_KDA_FWD_HEAD_CHUNK_SIZE=${GLM53_KDA_FWD_HEAD_CHUNK_SIZE:-1}
    export GLM53_KDA_BWD_HEAD_CHUNK_SIZE=${GLM53_KDA_BWD_HEAD_CHUNK_SIZE:-64}
    export GLM53_MOE_DISPATCH_CHUNK_SIZE=512 GLM53_EXPERT_TOKEN_CHUNK_SIZE=1024
    # This compatibility override is absent from the reference training environment.
    unset GLM53_TMS_POST7_RELEASE_COMPAT
}


geo3k_cleanup() {
    /usr/bin/python3 -I -S - "$1" <<'PY_CLEANUP'
import json,os,signal,sys,time
from pathlib import Path
def cleanup(token):
    marker=('GLM53_FULL128_RUN_ID='+token).encode()
    protected={os.getpid()}
    pid=os.getppid()
    while pid>1:
        protected.add(pid)
        try: pid=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[1])
        except (OSError,ValueError,IndexError): break
    def scan():
        result={}
        for p in Path('/proc').iterdir():
            if not p.name.isdigit() or int(p.name) in protected: continue
            try:
                if marker in (p/'environ').read_bytes().split(b'\0'):
                    st=(p/'stat').read_text().rsplit(')',1)[1].split()
                    if st[0]!='Z': result[int(p.name)]=st[19]
            except (OSError,IndexError): pass
        return result
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for pid,ticks in scan().items():
            try:
                now=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
                if now[19]==ticks: os.kill(pid,sig)
            except (OSError,IndexError): pass
        time.sleep(3)
    left=scan()
    print(json.dumps({'cleanup_run_id':token,'remaining_owned_processes':left}))
    assert not left, 'Some task-owned processes remain'
cleanup(sys.argv[1])
PY_CLEANUP
}

geo3k_wait_peer() {
    /usr/bin/python3 -I -S - "$1" "$2" "$RAY_AUTH_TOKEN_PATH" "$HEAD_IP" <<'PY_PEER'
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

run_id, pid = sys.argv[1], int(sys.argv[2])
assert run_id.startswith('geo3k8_')
stat = Path(f'/proc/{pid}/stat')
ticks = stat.read_text().rsplit(')', 1)[1].split()[19]
token = Path(sys.argv[3]).read_text().strip()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
url = 'http://' + sys.argv[4] + ':25401/api/jobs/glm53-full128-' + run_id
last_contact = started = time.monotonic()
seen = False
while True:
    try:
        st = stat.read_text().rsplit(')', 1)[1].split()
        assert st[19] == ticks and st[0] != 'Z'
    except (OSError, AssertionError):
        sys.exit('This run peer Ray process exited')
    try:
        request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + token})
        with opener.open(request, timeout=4) as response:
            job = json.load(response)
        last_contact = time.monotonic()
        assert job.get('submission_id', 'glm53-full128-' + run_id) == 'glm53-full128-' + run_id
        status = job['status']
        seen = True
        if status in ('SUCCEEDED', 'FAILED', 'STOPPED'):
            print(json.dumps({'run_id': run_id, 'head_job_terminal_status': status}), flush=True)
            sys.exit(0 if status == 'SUCCEEDED' else 1)
        assert status in ('PENDING', 'RUNNING'), status
    except urllib.error.HTTPError as error:
        if error.code == 404 and not seen:
            last_contact = time.monotonic()
        else:
            print('Head job status temporarily unavailable: HTTP ' + str(error.code), flush=True)
    except (OSError, ValueError) as error:
        print('Head job status temporarily unavailable: ' + type(error).__name__, flush=True)
    if time.monotonic() - last_contact > 120:
        sys.exit('Head dashboard unavailable for 120 seconds; releasing only this run peer')
    if not seen and time.monotonic() - started > 1800:
        sys.exit('This run job was not submitted within 1800 seconds')
    time.sleep(3)

PY_PEER
}

geo3k_main() {
    local run_id='' mode=run arg
    if (( $# == 0 )); then mode=help; fi
    while (( $# )); do
        arg=$1; shift
        case "$arg" in
            --help|-h) mode=help ;;
            --print-command) mode=print ;;
            --run-id) [[ $# -gt 0 ]] || return 2; run_id=$1; shift ;;
            *) echo "Unknown argument: $arg" >&2; return 2 ;;
        esac
    done
    if [[ $mode == help ]]; then
        cat <<'USAGE'
Usage: bash lh_scripts/run_glm53_8host_geo3k_aligned_A5.sh --run-id NEW_RUN_ID
       bash lh_scripts/run_glm53_8host_geo3k_aligned_A5.sh --print-command
Full45L/288E/top8 BF16, 8x16 Ascend dies, TP1/PP4/CP1/EP32, GBS128.
Eight TP16/DP16/EP16 multimodal engines; local Geo3K train split, 4K context/2K prompt/2K response.
TIS disabled; rollout routing replay enabled. A5 recipe adapted to the current 128-die allocation.
Default1000 updates. No checkpoint writes. Run the same new ID on all8 containers.
This entry contains the launch lifecycle; there are no Geo3K probe/check helpers.
Logs/caches/spill use shared LOG_ROOT with a separate directory per node.
USAGE
        return 0
    fi
    if [[ $mode == print ]]; then
        printf '%q ' "$CONDA_ENV/bin/python" -u "$SLIME_ROOT/train.py" "${ARGS[@]}"
        printf '
'; return 0
    fi
    [[ $run_id =~ ^geo3k8_[a-zA-Z0-9_-]{1,55}$ ]] || { echo 'Supply a new geo3k8_ run ID' >&2; return 2; }
    [[ -d $HF_CHECKPOINT ]] || { echo "Missing HF checkpoint: $HF_CHECKPOINT" >&2; return 2; }
    [[ -f $PROMPT_DATA ]] || { echo "Missing reference dataset: $PROMPT_DATA (expected prompt/label fields)" >&2; return 2; }
    local NODE_IP=${NODE_IP:-} HEAD_IP=80.5.27.199 COMM_IFNAME=eth0 NODE_DIES=16
    local VISIBLE_DIES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    if [[ -z $NODE_IP ]]; then
        NODE_IP=$(/usr/bin/python3 -I -S - <<'PY_IP'
import fcntl,socket,struct
with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as s:
    print(socket.inet_ntoa(fcntl.ioctl(s,0x8915,struct.pack('256s',b'eth0'))[20:24]))
PY_IP
)
    fi
    case "$NODE_IP" in
        80.5.27.199|80.5.27.200|80.5.27.202|80.5.27.203|80.5.27.204|80.5.27.205|80.5.27.206|80.5.27.207) ;;
        *) echo 'This host is outside the assigned allocation' >&2; return 2 ;;
    esac
    export NODE_IP HEAD_IP COMM_IFNAME NODE_DIES VISIBLE_DIES
    export GLM53_FULL128_RUN_ID="$run_id" GLM53_NIGHT_OWNER=geo3k8_20261110
    export TMPDIR; TMPDIR=$(mktemp -d /tmp/g53f128.XXXXXX)
    local cluster_dir="$LOG_ROOT/$run_id" node_log="$LOG_ROOT/$run_id/node-$NODE_IP"
    mkdir -p "$cluster_dir"
    mkdir "$node_log" || { echo 'Run ID already used on this host' >&2; return 2; }
    exec 9>"/tmp/glm53-full128-$UID.lock"
    flock -n 9 || { echo 'Another full128 run owns this container' >&2; return 75; }
    exec > >(tee -a "$node_log/launcher.log") 2>&1
    local stop_command
    printf -v stop_command 'rc=$?; trap - EXIT; if [[ %q == %q ]]; then printf "%%s\n" "$rc" > %q; fi; geo3k_cleanup %q > %q 2>&1 || true; exit "$rc"' "$NODE_IP" "$HEAD_IP" "$cluster_dir/head-exit-code" "$run_id" "$node_log/cleanup.json"
    trap "$stop_command" EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    glm53_full128_activate
    ulimit -c 0; ulimit -n 65535
    local triton_node="$node_log" torch_node="$node_log"
    if [[ -n ${GLM53_TRITON_CACHE_RUN_ID:-} ]]; then
        [[ $GLM53_TRITON_CACHE_RUN_ID =~ ^geo3k8_[A-Za-z0-9_-]+$ ]] || return 2
        triton_node="$LOG_ROOT/$GLM53_TRITON_CACHE_RUN_ID/node-$NODE_IP"
    fi
    if [[ -n ${GLM53_TORCH_CACHE_RUN_ID:-} ]]; then
        [[ $GLM53_TORCH_CACHE_RUN_ID =~ ^geo3k8_[A-Za-z0-9_-]+$ ]] || return 2
        torch_node="$LOG_ROOT/$GLM53_TORCH_CACHE_RUN_ID/node-$NODE_IP"
    fi
    export RAY_TMPDIR="$TMPDIR/ray" TRITON_CACHE_DIR="$triton_node/triton"
    export TORCH_EXTENSIONS_DIR="$torch_node/torch-extensions" HF_MODULES_CACHE="$node_log/hf_modules"
    mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$HF_MODULES_CACHE"
    cd "$SLIME_ROOT"
    local gcs_port=25400 dashboard_port=25401 i ready=0
    local -a ray_args=(start --block --node-ip-address="$NODE_IP" --num-cpus=48
        --resources="{\"NPU\":$NODE_DIES}" --disable-usage-stats --object-store-memory=85899345920 --object-spilling-directory="$node_log/object-spill"
        --object-manager-port=25402 --node-manager-port=25403
        --dashboard-agent-grpc-port=25405 --dashboard-agent-listen-port=25406
        --runtime-env-agent-port=25407 --metrics-export-port=25408
        --min-worker-port=28600 --max-worker-port=28999)
    if [[ $NODE_IP == $HEAD_IP ]]; then
        ray_args+=(--head --port="$gcs_port" --dashboard-host="$HEAD_IP" --dashboard-port="$dashboard_port" --ray-client-server-port=25404 --temp-dir="$RAY_TMPDIR")
    else
        for ((i=0;i<450;i++)); do
            if glm53_dashboard_ready "$HEAD_IP" "$dashboard_port" >/dev/null 2>&1; then ready=1; break; fi
            sleep 2
        done
        (( ready )) || return 1
        ray_args+=(--address="$HEAD_IP:$gcs_port")
    fi
    setsid "$CONDA_ENV/bin/ray" "${ray_args[@]}" > "$node_log/ray-start.log" 2>&1 &
    local ray_pid=$!
    printf '%s
' "$ray_pid" > "$node_log/ray.pid"
    if [[ $NODE_IP != $HEAD_IP ]]; then geo3k_wait_peer "$run_id" "$ray_pid"; return $?; fi
    ready=0
    for ((i=0;i<300;i++)); do
        if glm53_dashboard_ready "$HEAD_IP" "$dashboard_port" >/dev/null 2>&1; then ready=1; break; fi
        kill -0 "$ray_pid" 2>/dev/null || return 1
        sleep 2
    done
    (( ready )) || return 1
    printf '%q ' "$CONDA_ENV/bin/python" -u "$SLIME_ROOT/train.py" "${ARGS[@]}" > "$cluster_dir/command.sh"
    printf '
' >> "$cluster_dir/command.sh"
    "$CONDA_ENV/bin/ray" job submit --submission-id="glm53-full128-$run_id"         --address="http://$HEAD_IP:$dashboard_port"         --runtime-env-json='{"worker_process_setup_hook":"glm53_ctor_setup_v5.install"}'         -- bash "$cluster_dir/command.sh"
}

geo3k_main "$@"
