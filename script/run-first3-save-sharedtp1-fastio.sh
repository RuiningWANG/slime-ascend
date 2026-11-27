#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NUM_ROLLOUT=3 GLM53_CHECKPOINT_INTERVAL=3 GLM53_SAVE_TRAIN_DATA=1 GLM53_PROFILE_ONE_STEP=0
exec /usr/bin/python3 -I -S "$SCRIPT_DIR/run_with_fast_checkpoint.py" --entry "$SCRIPT_DIR/run-glm5.3-flash-300B-8nodes-geo3k-long150-sharedtp1-fixedckpt.sh" "$@"
