"""基线：单信号判别（学习 vs 手工的头对头）+ 下游解码方法（学习门控 vs 手工门控）。

一、判别层（c* / m* 二分类，AUROC 阈值无关）：
   每个"手工信号" = 一个特征列。方向在 train 上确定，test 上报 AUROC（无泄漏）。
   "最佳单信号"在 train 上选取、test 报值（避免 winner's curse）。
   另加 logistic（线性、全特征）基线，隔离"非线性 MLP"的贡献。

二、解码层（单 token EM）：
   方法统一在缓存的 top-K logits 上解码，与 gold 首 token 比 EM。
   说明：CAD/ARR/CoRect/CRED-hard 共享同一套"power-family 解码模板" τ=1+(2d-1)s，
   唯一区别是"信谁"的门控 d——这正是 Credence 要检验的变量（把手工门控换成学习门控）。
   AdaCAD 是门控 + 对比解码（contrastive）。
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import config
from features.extract import FEATURE_NAMES

# 手工信号 → 特征列（供可读性）
SIGNAL_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _auroc(y_bin, score):
    """标准 AUROC：score 越大越倾向 1。"""
    y = np.asarray(y_bin).astype(int)
    s = np.asarray(score, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _fit_sign_auroc_both(s_tr, y_tr, s_te, y_te):
    """方向在 train 上确定，返回 (train_auroc, test_auroc)。"""
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return float("nan"), float("nan")
    sign = 1.0 if roc_auc_score(y_tr, s_tr) >= roc_auc_score(y_tr, -s_tr) else -1.0
    return (float(roc_auc_score(y_tr, sign * s_tr)),
            float(roc_auc_score(y_te, sign * s_te)))


# --------------------------------------------------------------------------- #
# 一、单信号判别（学习 vs 手工的头对头）
# --------------------------------------------------------------------------- #
def single_signal_aurocs(Xtr, Xte, c_tr, c_te, m_tr, m_te):
    """每个手工信号对 c*/m* 的判别力。

    返回 test_aurocs_dict：{name: {auroc_c, auroc_m, auroc_c_train, auroc_m_train}}。
    train 值供 best_single_signal 做无泄漏选取。
    """
    out = {}
    for name, i in SIGNAL_IDX.items():
        ctr, cte = _fit_sign_auroc_both(Xtr[:, i], c_tr, Xte[:, i], c_te)
        mtr, mte = _fit_sign_auroc_both(Xtr[:, i], m_tr, Xte[:, i], m_te)
        out[name] = {"auroc_c": cte, "auroc_m": mte,
                     "auroc_c_train": ctr, "auroc_m_train": mtr}
    return out


def _key_or(x):
    return x if (x is not None and x == x) else -1.0


def best_single_signal(aurocs):
    """用 train AUROC 选取最佳信号，返回 (best_c_name, best_c_test, best_m_name, best_m_test)。
    在 train 上选、test 上报值，避免 winner's curse。"""
    bc = max(aurocs.items(), key=lambda kv: _key_or(kv[1]["auroc_c_train"]))
    bm = max(aurocs.items(), key=lambda kv: _key_or(kv[1]["auroc_m_train"]))
    return bc[0], bc[1]["auroc_c"], bm[0], bm[1]["auroc_m"]


def logistic_auc(Xtr, Xte, ytr_bin, yte_bin):
    """线性（全特征）判别：logistic 在 train 拟合、test 报 AUROC（方向由模型学）。"""
    if len(np.unique(ytr_bin)) < 2 or len(np.unique(yte_bin)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr_bin)
    return _auroc(yte_bin, clf.predict_proba(Xte)[:, 1])


# --------------------------------------------------------------------------- #
# 二、下游解码
# --------------------------------------------------------------------------- #
def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _argmax_token(tk, q):
    return int(tk["tokens"][np.argmax(q)])


def dec_greedy_pri(tk, f, pc, pm):
    return int(tk["tokens"][np.argmax(tk["z_pri"])])


def dec_greedy_ctx(tk, f, pc, pm):
    return int(tk["tokens"][np.argmax(tk["z_ctx"])])


def dec_cad(tk, f, pc, pm):
    a = config.CAD_ALPHA
    q = (1 + a) * tk["z_ctx"] - a * tk["z_pri"]
    return _argmax_token(tk, q)


def dec_arr(tk, f, pc, pm):
    s = f[2]                          # jsd
    d = 1 if f[0] > 0 else 0          # max_prob_gap（手工门控）
    tau = 1.0 + (2 * d - 1) * s
    q = (1 - tau) * tk["z_pri"] + tau * tk["z_ctx"]
    return _argmax_token(tk, q)


def dec_adacad(tk, f, pc, pm):
    jsd = f[2]
    if jsd <= config.ADACAD_THETA:
        return dec_greedy_ctx(tk, f, pc, pm)   # 无冲突 → 信上下文
    alpha = (1 - jsd) ** config.ADACAD_GAMMA   # 冲突越强，对比越弱：α=(1−JSD)^γ
    q = (1 + alpha) * tk["z_ctx"] - alpha * tk["z_pri"]
    return _argmax_token(tk, q)


def dec_corect(tk, f, pc, pm, thresh):
    supp = f[9]                       # 参数抑制（手工门控）
    d = 0 if supp > thresh else 1     # 被抑制 → 信记忆；否则信上下文
    s = f[2]
    tau = 1.0 + (2 * d - 1) * s
    q = (1 - tau) * tk["z_pri"] + tau * tk["z_ctx"]
    return _argmax_token(tk, q)


def dec_cred_hard(tk, f, pc, pm):
    d = 1 if pc > pm else 0           # 学习门控
    s = f[2]
    tau = 1.0 + (2 * d - 1) * s
    q = (1 - tau) * tk["z_pri"] + tau * tk["z_ctx"]
    return _argmax_token(tk, q)


def dec_cred_mix(tk, f, pc, pm):
    # 软贝叶斯混合：按后验信念加权两路分布。pc/pm 同乘正数不改变 argmax，故不归一也正确。
    q = pc * _softmax(tk["z_ctx"]) + pm * _softmax(tk["z_pri"])
    return _argmax_token(tk, q)


def _mean(x):
    x = list(x)
    return float(np.mean(x)) if x else float("nan")


def run_decoders(topks, feats, c_star, m_star, p_c=None, p_m=None, corect_thresh=None):
    """在给定样本上跑全部解码方法，返回 {method: {em, em_correction, em_resistance,
    em_agreement, em_double_wrong}}。

    topks: list of topk_cache；feats: [N,17]；c_star/m_star 用于按四态分组。
    p_c/p_m: 学习到的方法才需要（None 时 CRED-* 自动跳过）。
    corect_thresh: CoRect 门控阈值，应在 train 上估计（避免 test 泄漏）。
    """
    n = len(topks)
    if corect_thresh is None:
        raise ValueError(
            "corect_thresh 必须在训练集上估计后传入（如 np.median(X_tr[:, 9])），"
            "禁止在测试集上取中位数，否则 CoRect 基线会对测试分布产生泄漏。")

    methods = [
        ("greedy-pri", dec_greedy_pri),
        ("greedy-ctx", dec_greedy_ctx),
        ("CAD", dec_cad),
        ("ARR", dec_arr),
        ("AdaCAD", dec_adacad),
        ("CoRect", lambda tk, f, pc, pm: dec_corect(tk, f, pc, pm, corect_thresh)),
        ("CRED-hard", dec_cred_hard),
        ("CRED-mix", dec_cred_mix),
    ]

    res = {}
    for name, dec in methods:
        if name.startswith("CRED") and (p_c is None or p_m is None):
            continue
        preds = []
        for i in range(n):
            pc = p_c[i] if p_c is not None else 0.0
            pm = p_m[i] if p_m is not None else 0.0
            preds.append(dec(topks[i], feats[i], pc, pm))
        em = [preds[i] == topks[i]["gold_tok"] for i in range(n)]
        res[name] = {
            "em": _mean(em),
            "em_correction": _mean(em[i] for i in range(n) if c_star[i] == 1 and m_star[i] == 0),
            "em_resistance": _mean(em[i] for i in range(n) if c_star[i] == 0 and m_star[i] == 1),
            "em_agreement": _mean(em[i] for i in range(n) if c_star[i] == 1 and m_star[i] == 1),
            "em_double_wrong": _mean(em[i] for i in range(n) if c_star[i] == 0 and m_star[i] == 0),
        }
    return res
