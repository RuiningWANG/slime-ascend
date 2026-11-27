#!/usr/bin/env bash
set -euo pipefail
export NUM_ROLLOUT=3
export GLM53_SAVE_TRAIN_DATA=1
deadline=$(date -d '2026-11-27 09:00:00 +0800' +%s)
remaining=$((deadline-$(date +%s)))
(( remaining > 0 )) || { echo 'Allocated resource window has ended' >&2; exit 2; }
exec timeout --signal=TERM --kill-after=30s "$remaining" bash '/mnt/share/w00876333/glm5.3/full_laryer_zero_mismatch_1126/script/run-glm5.3-flash-300B-8nodes-geo3k-npu.sh' "$@"
