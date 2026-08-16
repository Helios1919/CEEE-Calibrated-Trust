#!/usr/bin/env bash
# 完整重跑：popqa + counterfact 两个数据源，最后打印汇总。
# 用法：nohup bash run_all.sh > logs/run_all.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DOWNLOAD_TIMEOUT=600

echo "==== [popqa] start $(date) ===="
python run_experiment.py --data popqa 2>&1
echo "==== [popqa] end $(date) ===="

echo "==== [counterfact] start $(date) ===="
python run_experiment.py --data counterfact 2>&1
echo "==== [counterfact] end $(date) ===="

echo "==== [results] start $(date) ===="
python results.py 2>&1
echo "==== ALL DONE $(date) ===="
