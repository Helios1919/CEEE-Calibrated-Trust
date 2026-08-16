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

ROOT = Path(__file__).resolve().parent

STATE_NAMES = {0: "double_wrong", 1: "resistance", 2: "correction", 3: "agreement"}
STATE2ID = {v: k for k, v in STATE_NAMES.items()}

# ---------------------------------------------------------------- model / data source
MODEL_NAME = os.environ.get("CRED_MODEL", "Qwen/Qwen2.5-7B")
# Data source: facts = built-in 80 facts (offline smoke); popqa = real PopQA (main comparison)
DATA_SOURCE = os.environ.get("CRED_DATA", "popqa")
POPQA_SPLIT = "test"          # PopQA's official split only has object labels on "test"
POPQA_N = int(os.environ.get("CRED_N", "3000"))     # number of items to sample (-1 = all)
POPQA_MAX_PER_PROP = int(os.environ.get("CRED_MAX_PER_PROP", "300"))
# CounterFact sample size (knowledge-conflict: true fact vs counterfact)
COUNTERFACT_N = int(os.environ.get("CRED_CF_N", "3000"))

# Note: torch<2.1 has a bfloat16 torch.triu bug on CUDA; fall back to float16 then.
TORCH_DTYPE = os.environ.get("CRED_DTYPE", "bfloat16")
MAX_NEW_TOKENS = 12           # max tokens for closed-book m* label generation
TOP_K = 100                   # cache top-K logits of z_pri/z_ctx for downstream decoding

# ---------------------------------------------------------------- paths
DATA_PATH = ROOT / "data.jsonl"
FEATURE_PATH = ROOT / "features.npz"
TOPK_PATH = ROOT / "topk_logits.pkl"
ESTIMATOR_PATH = ROOT / "estimator.pt"
RESULT_PATH = ROOT / "results.json"
LOG_DIR = ROOT / "logs"

# ---------------------------------------------------------------- split / training
SEED = int(os.environ.get("CRED_SEED", "0"))
SEEDS = [0, 1, 2]              # multiple estimator seeds (report mean±std)
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2               # all baselines/estimators share the same test set

HIDDEN_DIM = 64
EPOCHS = 300
LR = 1e-3

# Temperature-scaling candidate grid (selected on validation by NLL)
T_GRID = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 3.0]
# Abstention (selective prediction): abstain score = P(double-wrong), refuse above threshold
ABSTAIN_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

# ---------------------------------------------------------------- decoder hyperparams (from the papers)
CAD_ALPHA = 1.0               # CAD: q = (1+alpha) z_ctx - alpha z_pri
ADACAD_THETA = 0.7            # AdaCAD: JSD>theta is treated as conflict
ADACAD_GAMMA = 1.0            # AdaCAD: alpha = (1-JSD)^gamma

# Ablation categories (see CATEGORIES in features/extract.py)
# ---------------------------------------------------------------- generalization
HELD_OUT_RELATIONS = ["author"]   # held out for facts source; popqa holds out by prop
CROSS_MODEL = os.environ.get("CRED_CROSS_MODEL", "Qwen/Qwen2.5-7B-Instruct")
