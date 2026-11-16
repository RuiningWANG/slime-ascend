#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export GLM53_PHASE=${GLM53_PHASE:-short-01} GLM53_BI_KDA_MODE=per-sequence
export RAY_NODE_IP=${RAY_NODE_IP:-197.53.23.1}
mkdir -p "$GLM53_PHASE"

probe() (
    unset PYTHONPATH LD_PRELOAD
    set +u
    source "$PWD/runtime/activate.sh" || exit
    set -u
    rc=0
    for physical in 0 1; do
        log="$GLM53_PHASE/acl-$1-card-$physical.log"
        ASCEND_RT_VISIBLE_DEVICES="$physical" /usr/bin/python3 -I -S probe_acl.py > "$log" 2>&1 || rc=1
        cat "$log"
    done
    exit "$rc"
)

if [[ ${1:-} == private ]]; then
    probe private
    exec bash start.sh
fi
if probe default; then
    exec bash start.sh
fi
if grep -q 'E30009.*cann-hybm-compat' "$GLM53_PHASE"/acl-default-card-*.log; then
    exec bash with_cann_manifest.sh run-node.sh private
fi
exit 1

