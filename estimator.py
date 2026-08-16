"""Credibility estimator: 4-way MLP (D->64->4) + temperature scaling + calibration
/discrimination metrics.

Outputs P(state) over {double_wrong, resistance, correction, agreement}, deriving:
    p_c = P(correction) + P(agreement)   # probability the context is correct
    p_m = P(resistance) + P(agreement)   # probability the memory is correct
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report

from config import STATE_NAMES, T_GRID


def ece(y_true, probs, n_bins=10):
    """Expected calibration error. y_true: int, probs: [N,4] softmax."""
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = (confs > bins[b]) & (confs <= bins[b + 1])
        if mask.sum() == 0:
            continue
        e += (mask.sum() / n) * abs((preds[mask] == y_true[mask]).mean() - confs[mask].mean())
    return float(e)


def _auroc(y_bin, score):
    """Binary AUROC; y_bin must contain both classes, else NaN. Larger score -> class 1."""
    y = np.asarray(y_bin)
    s = np.asarray(score)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def p_cm_from_probs(probs):
    """4-way probs -> p_c, p_m. probs: [...,4]"""
    p_c = probs[..., 2] + probs[..., 3]
    p_m = probs[..., 1] + probs[..., 3]
    return p_c, p_m


class Estimator(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 4))

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- single fit
def _fit_once(model, Xtr_t, ytr_t, Xva_t, yva_t, epochs, lr, seed, device):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        va_logits = model(Xva_t)
        best_T, best_nll = 1.0, float("inf")
        for T in T_GRID:
            nll = F.cross_entropy(va_logits / T, yva_t).item()
            if nll < best_nll:
                best_nll, best_T = nll, T
    return model, float(best_T)


# ---------------------------------------------------------------- metrics
def evaluate_probs(probs, y, c_star, m_star):
    """Return a full metric dict given [N,4] probs and labels. c_star/m_star are binary ground truth."""
    pred = probs.argmax(1)
    p_c, p_m = p_cm_from_probs(probs)
    return {
        "acc4": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "ece": ece(y, probs),
        "auroc_c": _auroc(c_star, p_c),
        "auroc_m": _auroc(m_star, p_m),
        "report": classification_report(
            y, pred, target_names=[STATE_NAMES[i] for i in range(4)],
            zero_division=0, output_dict=True),
    }


# ---------------------------------------------------------------- multi-seed training + testing
def train_estimator(X, y, c_star, m_star, splits, epochs=300, lr=1e-3,
                    seeds=(0, 1, 2), device="cuda"):
    """Train on a fixed split, repeat over seeds, return (best_model, T, norm, mean report, per-seed).

    splits = {"tr": idx, "va": idx, "te": idx} (fixed, so every method shares the same test set)
    """
    tr, va, te = splits["tr"], splits["va"], splits["te"]
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y[tr], y[va], y[te]

    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    Xtr_s, Xva_s, Xte_s = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd

    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32).to(device)
    ytr_t = torch.tensor(ytr, dtype=torch.long).to(device)
    Xva_t = torch.tensor(Xva_s, dtype=torch.float32).to(device)
    yva_t = torch.tensor(yva, dtype=torch.long).to(device)
    Xte_t = torch.tensor(Xte_s, dtype=torch.float32).to(device)

    per_seed = []
    best = None
    best_val_acc = -1.0
    for sd_i in seeds:
        m = Estimator(X.shape[1]).to(device)
        m, T = _fit_once(m, Xtr_t, ytr_t, Xva_t, yva_t, epochs, lr, sd_i, device)
        with torch.no_grad():
            va_prob = F.softmax(m(Xva_t) / T, -1).cpu().numpy()
        val_acc = float(accuracy_score(yva, va_prob.argmax(1)))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best = (m, T)

    # report on test with the model best on validation
    m, T = best
    with torch.no_grad():
        te_prob = F.softmax(m(Xte_t) / T, -1).cpu().numpy()
    rep = evaluate_probs(te_prob, yte, c_star[te], m_star[te])
    rep["n_test"] = int(len(yte))

    # per-seed variance (also on test)
    per_seed = []
    for sd_i in seeds:
        mm = Estimator(X.shape[1]).to(device)
        mm, TT = _fit_once(mm, Xtr_t, ytr_t, Xva_t, yva_t, epochs, lr, sd_i, device)
        with torch.no_grad():
            pp = F.softmax(mm(Xte_t) / TT, -1).cpu().numpy()
        per_seed.append(evaluate_probs(pp, yte, c_star[te], m_star[te]))

    return m, T, (mu, sd), rep, per_seed


def predict_probs(model, X, T, norm_params, device="cuda"):
    """Return the full [N,4] probs (for normalized input)."""
    mu, sd = norm_params
    X_s = (X - mu) / sd
    X_t = torch.tensor(X_s, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        return F.softmax(model(X_t) / T, -1).cpu().numpy()
