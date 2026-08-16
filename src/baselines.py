"""Baselines: single-signal discriminators (learned vs hand-crafted head-to-head)
plus downstream decoding methods (learned gating vs hand-crafted gating).

1. Discrimination layer (c* / m* binary classification, threshold-free AUROC):
   each "hand-crafted signal" is one feature column. The direction is determined on
   train and AUROC is reported on test (no leakage). The "best single signal" is
   selected on train and reported on test (avoids winner's curse). A logistic
   (linear, all-features) baseline isolates the contribution of the nonlinear MLP.

2. Decoding layer (single-token EM):
   every method decodes over the cached top-K logits and is scored against the gold
   first token. CAD/ARR/CoRect/CRED-hard share the same "power-family decoding
   template" tau = 1 + (2d - 1)s; the only difference is the "who to trust" gate d,
   which is exactly the variable Credence tests (hand gate -> learned gate).
   AdaCAD adds contrastive decoding on top of its gate.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import config
from features.extract import FEATURE_NAMES

# hand-crafted signal -> feature column (for readability)
SIGNAL_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _auroc(y_bin, score):
    """Standard AUROC: larger score -> more likely class 1."""
    y = np.asarray(y_bin).astype(int)
    s = np.asarray(score, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _fit_sign_auroc_both(s_tr, y_tr, s_te, y_te):
    """Direction determined on train; returns (train_auroc, test_auroc)."""
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return float("nan"), float("nan")
    sign = 1.0 if roc_auc_score(y_tr, s_tr) >= roc_auc_score(y_tr, -s_tr) else -1.0
    return (float(roc_auc_score(y_tr, sign * s_tr)),
            float(roc_auc_score(y_te, sign * s_te)))


# --------------------------------------------------------------------------- #
# 1. single-signal discrimination (learned vs hand-crafted head-to-head)
# --------------------------------------------------------------------------- #
def single_signal_aurocs(Xtr, Xte, c_tr, c_te, m_tr, m_te):
    """Discriminative power of each hand-crafted signal for c*/m*.

    Returns test_aurocs_dict: {name: {auroc_c, auroc_m, auroc_c_train, auroc_m_train}}.
    train values let best_single_signal select without leakage.
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
    """Select the best signal by train AUROC, return (best_c_name, best_c_test, best_m_name, best_m_test).
    Selected on train, reported on test, avoiding winner's curse."""
    bc = max(aurocs.items(), key=lambda kv: _key_or(kv[1]["auroc_c_train"]))
    bm = max(aurocs.items(), key=lambda kv: _key_or(kv[1]["auroc_m_train"]))
    return bc[0], bc[1]["auroc_c"], bm[0], bm[1]["auroc_m"]


def logistic_auc(Xtr, Xte, ytr_bin, yte_bin):
    """Linear (all-feature) discriminator: logistic fit on train, AUROC on test (direction learned)."""
    if len(np.unique(ytr_bin)) < 2 or len(np.unique(yte_bin)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr_bin)
    return _auroc(yte_bin, clf.predict_proba(Xte)[:, 1])


# --------------------------------------------------------------------------- #
# 2. downstream decoding
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


def dec_cad(tk, f, pc, pm, alpha=None):
    a = config.CAD_ALPHA if alpha is None else alpha
    q = (1 + a) * tk["z_ctx"] - a * tk["z_pri"]
    return _argmax_token(tk, q)


def dec_arr(tk, f, pc, pm):
    s = f[2]                          # jsd
    d = 1 if f[0] > 0 else 0          # max_prob_gap (hand gate)
    tau = 1.0 + (2 * d - 1) * s
    q = (1 - tau) * tk["z_pri"] + tau * tk["z_ctx"]
    return _argmax_token(tk, q)


def dec_adacad(tk, f, pc, pm, theta=None, gamma=None):
    theta = config.ADACAD_THETA if theta is None else theta
    gamma = config.ADACAD_GAMMA if gamma is None else gamma
    jsd = f[2]
    if jsd <= theta:
        return dec_greedy_ctx(tk, f, pc, pm)   # no conflict -> trust context
    alpha = (1 - jsd) ** gamma   # stronger conflict -> weaker contrast: alpha=(1-JSD)^gamma
    q = (1 + alpha) * tk["z_ctx"] - alpha * tk["z_pri"]
    return _argmax_token(tk, q)


def dec_corect(tk, f, pc, pm, thresh):
    supp = f[9]                       # parametric suppression (hand gate)
    d = 0 if supp > thresh else 1     # suppressed -> trust memory; else trust context
    s = f[2]
    tau = 1.0 + (2 * d - 1) * s
    q = (1 - tau) * tk["z_pri"] + tau * tk["z_ctx"]
    return _argmax_token(tk, q)


def dec_cred_hard(tk, f, pc, pm):
    d = 1 if pc > pm else 0           # learned gate
    s = f[2]
    tau = 1.0 + (2 * d - 1) * s
    q = (1 - tau) * tk["z_pri"] + tau * tk["z_ctx"]
    return _argmax_token(tk, q)


def dec_cred_mix(tk, f, pc, pm):
    # Soft Bayesian mix: weight the two distributions by posterior belief. Scaling pc/pm
    # by a common positive factor does not change argmax, so normalization is unnecessary.
    q = pc * _softmax(tk["z_ctx"]) + pm * _softmax(tk["z_pri"])
    return _argmax_token(tk, q)


def dec_uniform_mix(tk, f, pc, pm):
    # Zero-learning control: uniform soft mix (pc=pm=0.5). Separates the contribution of
    # the soft-mix mechanism itself from the learned estimator's contribution.
    q = 0.5 * _softmax(tk["z_ctx"]) + 0.5 * _softmax(tk["z_pri"])
    return _argmax_token(tk, q)


def dec_conf_mix(tk, f, pc, pm):
    # Hand-crafted soft-mix control: weight by the single-signal confidences conf_ctx/conf_pri
    # (non-learned version). Differs from CRED-mix only in where the weights come from,
    # isolating the net gain of the learned estimator.
    w_ctx = float(f[6])   # conf_ctx
    w_pri = float(f[5])   # conf_pri
    q = w_ctx * _softmax(tk["z_ctx"]) + w_pri * _softmax(tk["z_pri"])
    return _argmax_token(tk, q)


def _mean(x):
    x = list(x)
    return float(np.mean(x)) if x else float("nan")


def _em_at(topks, feats, dec):
    """Average single-token EM of a decoder over a sample set (for validation tuning)."""
    n = len(topks)
    if n == 0:
        return 0.0
    hit = 0.0
    for i in range(n):
        if dec(topks[i], feats[i], 0.0, 0.0) == topks[i]["gold_tok"]:
            hit += 1.0
    return hit / n


def tune_baseline_params(topks_va, feats_va):
    """Search baseline decoder hyperparams on the validation set, matching the learned
    side's tuning budget (temperature scaling / multi-seed selection).

    Returns {"cad_alpha", "adacad_theta", "adacad_gamma", "corect_thresh"} (all float).
    The CoRect threshold is also estimated on validation only, never touching test,
    to avoid leaking the test distribution.
    """
    # CAD alpha: contrast strength
    cad = max(((a, _em_at(topks_va, feats_va,
                          lambda tk, f, pc, pm, a=a: dec_cad(tk, f, pc, pm, alpha=a)))
               for a in np.linspace(0.0, 2.0, 9)), key=lambda t: t[1])

    # AdaCAD: theta (conflict threshold) and gamma (contrast decay)
    ada = None
    for th in np.linspace(0.0, 1.0, 11):
        for gm in (0.5, 1.0, 1.5, 2.0):
            em = _em_at(topks_va, feats_va,
                        lambda tk, f, pc, pm, th=th, gm=gm:
                        dec_adacad(tk, f, pc, pm, theta=th, gamma=gm))
            if ada is None or em > ada[2]:
                ada = (th, gm, em)

    # CoRect: parametric-suppression threshold
    supp = feats_va[:, 9]
    corect = None
    for t in np.percentile(supp, [10, 25, 40, 50, 60, 75, 90]):
        em = _em_at(topks_va, feats_va,
                    lambda tk, f, pc, pm, t=t: dec_corect(tk, f, pc, pm, t))
        if corect is None or em > corect[1]:
            corect = (float(t), em)

    return {"cad_alpha": float(cad[0]),
            "adacad_theta": float(ada[0]),
            "adacad_gamma": float(ada[1]),
            "corect_thresh": float(corect[0])}


def run_decoders(topks, feats, c_star, m_star, p_c=None, p_m=None,
                 cad_alpha=None, adacad_theta=None, adacad_gamma=None,
                 corect_thresh=None):
    """Run every decoding method on the given samples, returning {method: {em,
    em_correction, em_resistance, em_agreement, em_double_wrong}}.

    topks: list of topk_cache; feats: [N,17]; c_star/m_star group by the four states.
    p_c/p_m: only needed by learned methods (CRED-* is skipped when None).
    Baseline hyperparams (cad_alpha/adacad_theta/adacad_gamma/corect_thresh) should be
    tuned on validation before being passed in (see tune_baseline_params) to avoid
    leaking the test distribution; None falls back to config defaults.
    Also includes two soft-mix controls (uniform-mix / conf-mix) to separate the
    soft-mix mechanism's contribution from the learned estimator's (avoid mis-attribution).
    """
    n = len(topks)
    if corect_thresh is None:
        corect_thresh = float(np.median(feats[:, 9]))

    methods = [
        ("greedy-pri", dec_greedy_pri),
        ("greedy-ctx", dec_greedy_ctx),
        ("CAD", lambda tk, f, pc, pm: dec_cad(tk, f, pc, pm, alpha=cad_alpha)),
        ("ARR", dec_arr),
        ("AdaCAD", lambda tk, f, pc, pm: dec_adacad(
            tk, f, pc, pm, theta=adacad_theta, gamma=adacad_gamma)),
        ("CoRect", lambda tk, f, pc, pm: dec_corect(tk, f, pc, pm, corect_thresh)),
        ("CRED-hard", dec_cred_hard),
        ("CRED-mix", dec_cred_mix),
        ("uniform-mix", dec_uniform_mix),
        ("conf-mix", dec_conf_mix),
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


def _aurc(scores, hits):
    """Area under the risk-coverage curve (AURC, lower is better). Larger scores tend to abstain."""
    order = np.argsort(-np.asarray(scores, dtype=float))
    h = np.asarray(hits, dtype=float)[order]
    total = h.sum()
    risk_sum = 0.0
    cum_rej = 0.0
    for k in range(len(h) - 1):
        n_acc = len(h) - k
        risk_sum += 1.0 - (total - cum_rej) / n_acc
        cum_rej += h[k]
    return float(risk_sum / len(h))


def evaluate_abstention(topks, probs, thresholds=None):
    """Selective-prediction (abstention) evaluation: abstain score = P(double-wrong) = probs[:, 0].

    Refuse when score >= threshold, otherwise answer with CRED-mix soft routing. Returns
    (coverage, EM) at the key thresholds plus AURC. coverage = fraction answered;
    EM = single-token exact match over the answered subset.
    """
    if thresholds is None:
        thresholds = config.ABSTAIN_THRESHOLDS
    n = len(topks)
    p_c = probs[:, 2] + probs[:, 3]
    p_m = probs[:, 1] + probs[:, 3]
    score = probs[:, 0]                       # P(double-wrong)
    hit = np.array([
        dec_cred_mix(topks[i], None, p_c[i], p_m[i]) == topks[i]["gold_tok"]
        for i in range(n)], dtype=float)

    points = []
    for th in thresholds:
        keep = score < th
        cov = float(keep.mean())
        em = float(hit[keep].mean()) if keep.sum() else float("nan")
        points.append((float(th), cov, em))

    return {
        "thresholds": [p[0] for p in points],
        "coverage": [p[1] for p in points],
        "em": [p[2] for p in points],
        "aurc": _aurc(score, hit),
    }
