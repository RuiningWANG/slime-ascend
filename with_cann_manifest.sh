#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
manifest="$PWD/runtime/ascend_package_load_no_hybm.ini"
shared=/mnt/share/l00804615/cann/9.2.0/cann-9.2.0/aarch64-linux/conf/ascend_package_load.ini
exec unshare --mount --propagation private bash -euc '
    mount --bind "$1" "$2"
    mount -o remount,bind,ro "$2"
    shift 2
    exec bash "$@"
' -- "$manifest" "$shared" "$@"

