#!/usr/bin/env bash
# Ascend/platform initialization only. Training parameters live in the public recipe.
set -euo pipefail
GLM53_RUNTIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT=${LOG_ROOT:-/mnt/share/l00937981/glm_53_flash_slime/cache/work/glm53-single-node}
SLIME_ACCELERATOR=npu
USE_LIBUV=0
SGLANG_MEMORY_SAVER_CUDA_GRAPH=true
GLM53_RAY_SIGINT_NONE_COMPAT=1
SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1
TMS_HOOK_MODE=preload
TMS_PRESERVE_PRELOAD_MODE=1
SLIME_HOST_IP=127.0.0.1
RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
# Some hosts enable Ray's dashboard token auth by default; the launcher's
# readiness probe (/api/jobs/) and `ray job submit` send no token and would get
# 401 forever (the probe then spins for its full 10-minute budget). Opting out
# here keeps every recipe working on those hosts.
export RAY_AUTH_MODE=${RAY_AUTH_MODE:-disabled}
HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-300}
HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-300}
ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-300}
SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
SGLANG_OPT_USE_TILELANG_MHC_PRE=0
SGLANG_OPT_USE_TILELANG_MHC_POST=0
SGLANG_ENABLE_JIT_DEEPGEMM=False
PYTHONDONTWRITEBYTECODE=1
OMP_NUM_THREADS=4
TOKENIZERS_PARALLELISM=false
TORCH_DEVICE_BACKEND_AUTOLOAD=0
FLA_NPU_DISABLE_PTH=1
PYTHONNOUSERSITE=1
GLM53_NUM_NPUS=${GLM53_NUM_NPUS:-8}
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
DEPS_ROOT=/mnt/share/l00937981/glm_53_flash_slime
CONDA_ENV=/mnt/share/l00937981/conda/envs/glm53-flash-slime
SLIME_ROOT=/mnt/share/l00937981/glm_53_flash_slime/slime-ascend/slime
CANN_ROOT=/mnt/share/l00804615/cann/9.2.0/cann-9.2.0

# Paths, interfaces, ports and explicit local overrides come from this machine.
source "${GLM53_RUNTIME_DIR}/host.env"
# Training-only TMS compatibility; never applied to the rollout engine.
if [[ -z ${TRAIN_ENV_JSON+x} ]]; then
    TRAIN_ENV_JSON="{\"HCCL_HOST_SOCKET_PORT_RANGE\":\"$TRAIN_HCCL_HOST_RANGE\",\"HCCL_NPU_SOCKET_PORT_RANGE\":\"$TRAIN_HCCL_NPU_RANGE\",\"GLM53_TMS_POST7_RELEASE_COMPAT\":\"1\"}"
fi


# Reuse the local launch flow; all tunable runtime values above mirror host 23.
source "${GLM53_RUNTIME_DIR}/lh-single-node-launch.sh"
