#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NUM_ROLLOUT=150 GLM53_CHECKPOINT_INTERVAL=10 GLM53_SAVE_TRAIN_DATA=0 GLM53_PROFILE_ONE_STEP=1
exec /usr/bin/python3 -I -S "$SCRIPT_DIR/run_with_fast_checkpoint.py" --entry "$SCRIPT_DIR/run-glm5.3-flash-300B-8nodes-geo3k-long150-sharedtp1-fixedckpt.sh" "$@"
