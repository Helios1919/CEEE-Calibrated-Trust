"""Global configuration (runnable remotely: all paths are repo-relative, model/data
can be overridden via environment variables).

Four-state encoding (c* = whether the context fact is correct, m* = whether the
model's parametric memory is correct):
    state = c* * 2 + m*
    0 = double_wrong  1 = resistance  2 = correction  3 = agreement

Derived credibility:
    p_c = P(state in {correction, agreement}) = P(context is correct)
    p_m = P(state in {resistance, agreement}) = P(memory is correct)
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # project root (config.py lives under src/)

STATE_NAMES = {0: "double_wrong", 1: "resistance", 2: "correction", 3: "agreement"}
STATE2ID = {v: k for k, v in STATE_NAMES.items()}

# ---------------------------------------------------------------- model / data source
MODEL_NAME = os.environ.get("CRED_MODEL", "Qwen/Qwen2.5-7B")
# Data source: facts = built-in 80 facts (offline smoke); popqa = real PopQA (main comparison)
DATA_SOURCE = os.environ.get("CRED_DATA", "popqa")
POPQA_SPLIT = "test"          # PopQA's official split only has object labels on "test"
POPQA_N = int(os.environ.get("CRED_N", "-1"))     # number of items to sample (-1 = all)
POPQA_MAX_PER_PROP = int(os.environ.get("CRED_MAX_PER_PROP", "-1"))   # -1 = no per-relation cap
# CounterFact sample size (knowledge-conflict: true fact vs counterfact)
COUNTERFACT_N = int(os.environ.get("CRED_CF_N", "-1"))

# Note: torch<2.1 has a bfloat16 torch.triu bug on CUDA; fall back to float16 then.
TORCH_DTYPE = os.environ.get("CRED_DTYPE", "bfloat16")
MAX_NEW_TOKENS = 12           # max tokens for closed-book m* label generation
TOP_K = 100                   # cache top-K logits of z_pri/z_ctx for downstream decoding

# ---------------------------------------------------------------- paths
ARTIFACT_DIR = ROOT / "artifacts"   # per-dataset run artifacts (data/features/estimator/results)
DATA_PATH = ARTIFACT_DIR / "data.jsonl"
FEATURE_PATH = ARTIFACT_DIR / "features.npz"
TOPK_PATH = ARTIFACT_DIR / "topk_logits.pkl"
ESTIMATOR_PATH = ARTIFACT_DIR / "estimator.pt"
LOG_DIR = ROOT / "logs"

# ---------------------------------------------------------------- split / training
SEED = int(os.environ.get("CRED_SEED", "0"))
SEEDS = [0, 1, 2]              # multiple estimator seeds (report mean±std)
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2               # all methods share the same test set

HIDDEN_DIM = 256            # estimator hidden width
HIDDEN_LAYERS = 2           # number of hidden layers (d -> h x L -> 4)
DROPOUT = 0.3               # dropout probability between hidden layers
WEIGHT_DECAY = 1e-4         # AdamW weight decay (regularization)
EPOCHS = 300                # max epochs (early stopping may stop earlier)
LR = 1e-3
EARLY_STOP_PATIENCE = 30    # stop if val accuracy does not improve for this many epochs

# Temperature-scaling candidate grid (selected on validation by NLL)
T_GRID = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 3.0]
