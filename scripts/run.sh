#!/usr/bin/env bash
# Run the full experiment on an A100 with a single command. Usage:
#   bash scripts/run.sh                          # main comparison (PopQA)
#   bash scripts/run.sh --data facts             # offline smoke test
set -euo pipefail

cd "$(dirname "$0")/.."

# Model cache dir (point to a mounted data disk to avoid re-downloading)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DOWNLOAD_TIMEOUT=600

echo "== environment check =="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "== run main experiment =="
python scripts/run_experiment.py "$@"

echo "== metrics =="
python scripts/analyze.py "$@"
