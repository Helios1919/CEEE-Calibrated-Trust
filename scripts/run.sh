#!/usr/bin/env bash
# Canonical Stage-1 pipeline. Use --force-build/--force-extract to rebuild caches.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
DATA=${DATA:-popqa}
BOOTSTRAP=${BOOTSTRAP:-2000}

"$PYTHON" scripts/run_experiment.py --data "$DATA" "$@"
"$PYTHON" scripts/analyze.py --data "$DATA"
"$PYTHON" scripts/stage1_falsification.py --data "$DATA" --bootstrap "$BOOTSTRAP"
"$PYTHON" scripts/label_sensitivity.py --data "$DATA" --bootstrap "$BOOTSTRAP"
"$PYTHON" scripts/evaluate_complete_answers.py --data "$DATA" --split all
"$PYTHON" scripts/evaluate_decisions.py --data "$DATA" --bootstrap "$BOOTSTRAP"
