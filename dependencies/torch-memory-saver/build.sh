#!/usr/bin/env bash
set -euo pipefail
tms_source="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${CANN_ROOT:=/mnt/share/l00804615/glm5.3flash_release/cann/9.2.0/cann-9.2.0}"
g++ -O2 -fPIC -shared -std=c++17 -DPy_LIMITED_API=0x03090000 -DTMS_BACKEND_ASCEND=1 -DTMS_HOOK_MODE_PRELOAD=1 \
  -I "$CANN_ROOT/include" "$tms_source/csrc/api_forwarder.cpp" "$tms_source/csrc/core.cpp" "$tms_source/csrc/entrypoint.cpp" \
  -L "$CANN_ROOT/lib64" -lascendcl -ldl "-Wl,-rpath,$CANN_ROOT/lib64" \
  -o "$tms_source/torch_memory_saver_hook_mode_preload.abi3.so"
