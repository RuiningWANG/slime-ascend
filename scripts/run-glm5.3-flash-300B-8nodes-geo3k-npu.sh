#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$here/../script/run-glm5.3-flash-300B-8nodes-geo3k-npu.sh" "$@"
