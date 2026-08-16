#!/usr/bin/env bash
# 在 A100 上一条命令跑完整实验。用法：
#   bash run.sh                          # 正式对比（PopQA）
#   bash run.sh --data facts             # 离线冒烟
#   bash run.sh --skip-ablation --skip-generalization   # 快速版
set -euo pipefail

cd "$(dirname "$0")"

# 模型缓存目录（可改到你挂载的大盘，避免重下）
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DOWNLOAD_TIMEOUT=600

echo "== 环境检查 =="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "== 跑主实验 =="
python run_experiment.py "$@"

echo "== 结果汇总 =="
python results.py
