#!/usr/bin/env bash
#SBATCH --job-name=credence
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Run the main experiment (PopQA comparison)
python run_experiment.py --data popqa
python results.py
