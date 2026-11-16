#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
phase=${GLM53_PHASE:-short-01}
[[ $phase =~ ^[a-zA-Z0-9_-]+$ ]]
export GLM53_PHASE_DIR="$PWD/$phase"
mkdir -p "$GLM53_PHASE_DIR"
export LOG_ROOT="$GLM53_PHASE_DIR/jobs"
export GLM53_BATCH_REPORT="$GLM53_PHASE_DIR/batch-invariance.json"
export GLM53_BI_DEPS=/mnt/share/w00876333/glm5.3/2026-11-14/11-50-31-batch-invariance/deps
export SAVE_DEBUG_TRAIN_DATA="$GLM53_PHASE_DIR/train-{rollout_id}-{rank}.pt"
until [[ -f checkpoint.exit ]]; do sleep 5; done
[[ $(cat checkpoint.exit) == 0 ]]
exec bash lh_scripts/run_glm53_single_node_geo3k.sh "$@"

