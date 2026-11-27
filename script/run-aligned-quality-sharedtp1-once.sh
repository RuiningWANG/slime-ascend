#!/usr/bin/env bash
set -euo pipefail
export LOG_ROOT="/mnt/share/w00876333/glm5.3/full_laryer_zero_mismatch_1126/all_logs"
deadline=$(date -d '2026-11-27 09:00:00 +0800' +%s)
remaining=$((deadline-$(date +%s)))
(( remaining > 0 )) || exit 2
exec timeout --signal=TERM --kill-after=30s "$remaining" bash "/mnt/share/w00876333/glm5.3/full_laryer_zero_mismatch_1126/script/run-glm53-eval-aligned-sharedtp1.sh" "$@"
