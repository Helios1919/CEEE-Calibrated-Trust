"""全局配置（远程可直接跑：所有路径相对仓库根，模型/数据源可用环境变量覆盖）。

状态编码（四态，c* = 上下文事实是否正确, m* = 模型记忆是否正确）：
    state = c* * 2 + m*
    0 = double_wrong(双错)  1 = resistance(抵抗)  2 = correction(纠正)  3 = agreement(一致)

派生可信度：
    p_c = P(state ∈ {correction, agreement}) = P(上下文正确)
    p_m = P(state ∈ {resistance, agreement}) = P(记忆正确)
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STATE_NAMES = {0: "double_wrong", 1: "resistance", 2: "correction", 3: "agreement"}
STATE2ID = {v: k for k, v in STATE_NAMES.items()}

# ---------------------------------------------------------------- 模型 / 数据源
MODEL_NAME = os.environ.get("CRED_MODEL", "Qwen/Qwen2.5-7B")
# 数据源：facts = 内置 80 条事实（离线冒烟）；popqa = 真实 PopQA（正式对比）
DATA_SOURCE = os.environ.get("CRED_DATA", "popqa")
POPQA_SPLIT = "test"          # PopQA 官方只有 test 带 obj 标签
POPQA_N = int(os.environ.get("CRED_N", "3000"))     # 采样条数（-1 = 全量）
POPQA_MAX_PER_PROP = int(os.environ.get("CRED_MAX_PER_PROP", "300"))
# CounterFact（知识冲突专用：真实事实 vs 反事实）采样条数
COUNTERFACT_N = int(os.environ.get("CRED_CF_N", "3000"))

# 注意：torch<2.1 在 CUDA 上对 bfloat16 的 torch.triu 有 bug，此时用 float16 兜底。
TORCH_DTYPE = os.environ.get("CRED_DTYPE", "bfloat16")
MAX_NEW_TOKENS = 12           # 封闭式 m* 标签生成的最大 token 数
TOP_K = 100                   # 缓存 z_pri/z_ctx 的 top-K logits（下游解码用，近似不失真）

# ---------------------------------------------------------------- 路径
DATA_PATH = ROOT / "data.jsonl"
FEATURE_PATH = ROOT / "features.npz"
TOPK_PATH = ROOT / "topk_logits.pkl"
ESTIMATOR_PATH = ROOT / "estimator.pt"
RESULT_PATH = ROOT / "results.json"
LOG_DIR = ROOT / "logs"

# ---------------------------------------------------------------- 切分 / 训练
SEED = int(os.environ.get("CRED_SEED", "0"))
SEEDS = [0, 1, 2]              # 估计器多随机种子（报告 mean±std）
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2               # 所有基线/估计器共用同一 test 集，公平对比

HIDDEN_DIM = 64
EPOCHS = 300
LR = 1e-3

# ---------------------------------------------------------------- 下游解码方法超参（照原论文）
CAD_ALPHA = 1.0               # CAD: q = (1+α) z_ctx − α z_pri
ADACAD_THETA = 0.7            # AdaCAD: JSD>θ 视为冲突
ADACAD_GAMMA = 1.0            # AdaCAD: α = (1−JSD)^γ

# 消融类别（来自 features/extract.py 的 CATEGORIES，见下）
# ---------------------------------------------------------------- 泛化
HELD_OUT_RELATIONS = ["author"]   # facts 源时留出；popqa 源时按 prop 留出
CROSS_MODEL = os.environ.get("CRED_CROSS_MODEL", "Qwen/Qwen2.5-7B-Instruct")
