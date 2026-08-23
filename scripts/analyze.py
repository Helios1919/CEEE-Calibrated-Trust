"""Re-focused evaluation for Credence.

Credence is a *calibrated joint 4-state credibility estimator*, not a decoder switch.
The right metrics are therefore organized in four layers:

  A. discrimination / classification quality  (the "classification" answer)
       - 4-way macro-F1, balanced accuracy, per-state F1
       - per-state one-vs-rest AUROC (+ macro-AUROC)
       - hand-crafted (single-signal) c* classifier (threshold fit on train, report on test)
       - linear logistic 4-way (isolates the nonlinear MLP gain)
  B. calibration quality  (a scalar signal has no probability to calibrate)
       - ECE, Brier (multiclass), NLL
  C. selective-prediction utility  (the payoff of calibration)
       - AURC + coverage-accuracy: our P(double-wrong) vs best single signal
  D. downstream diagnosis  (the EM ceiling is structural, not a method failure)
       - oracle upper bound given the four-state distribution

Reads artifacts produced by run_experiment.py — no LM reload, no retrain.

Usage:
  python scripts/analyze.py --data popqa
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

import config
from config import STATE_NAMES
from estimator import Estimator, ece, p_cm_from_probs, predict_probs
from features.extract import FEATURE_NAMES


SIGNAL_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _aurc(scores, hits):
    """Area under the risk-coverage curve; lower scores are retained first."""
    order = np.argsort(np.asarray(scores, dtype=float))
    errors = 1.0 - np.asarray(hits, dtype=float)[order]
    risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(risk.mean())


# --------------------------------------------------------------------------- #
# split reproduction (must match run_experiment.py exactly)
# --------------------------------------------------------------------------- #
def load_samples(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _split(a, y_a, test_size, seed):
    try:
        return train_test_split(a, test_size=test_size, random_state=seed,
                                stratify=y_a)
    except ValueError:
        return train_test_split(a, test_size=test_size, random_state=seed)


def group_ids(samples):
    keys = {}
    ids = []
    for s in samples:
        key = s["item_id"]
        if key not in keys:
            keys[key] = len(keys)
        ids.append(keys[key])
    return np.array(ids)


def make_group_split(groups, group_label, seed=0):
    uniq_g, first_idx = np.unique(groups, return_index=True)
    g_lab = np.asarray(group_label)[first_idx]
    g_arr = np.arange(len(uniq_g))
    g_tr, g_rest = _split(g_arr, g_lab, 1 - config.TRAIN_FRAC, seed)
    g_va, g_te = _split(g_rest, g_lab[g_rest],
                        config.TEST_FRAC / (config.VAL_FRAC + config.TEST_FRAC),
                        seed)

    def _expand(g_sel):
        return np.where(np.isin(groups, uniq_g[g_sel]))[0]

    return {"tr": _expand(g_tr), "va": _expand(g_va), "te": _expand(g_te)}


# --------------------------------------------------------------------------- #
# layer A: discrimination / classification
# --------------------------------------------------------------------------- #
def multiclass_brier(y, probs):
    y_oh = np.zeros_like(probs)
    y_oh[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - y_oh) ** 2, axis=1)))


def per_state_auroc(y, probs):
    out = {}
    for k in range(4):
        ybin = (y == k).astype(int)
        s = probs[:, k]
        out[STATE_NAMES[k]] = (float(roc_auc_score(ybin, s))
                               if len(np.unique(ybin)) == 2 else float("nan"))
    out["macro"] = float(np.nanmean(list(out.values())))
    return out


def single_signal_classifier(Xtr, c_tr, Xte, c_te):
    """Threshold every single signal into a c* binary classifier (fit on train,
    report on test). Returns {name: {auroc, acc, macro_f1, balanced_acc}}."""
    out = {}
    for name, i in SIGNAL_IDX.items():
        s_tr, s_te = Xtr[:, i], Xte[:, i]
        if len(np.unique(c_tr)) < 2 or len(np.unique(c_te)) < 2:
            out[name] = {"auroc": float("nan"), "acc": float("nan"),
                         "macro_f1": float("nan"), "balanced_acc": float("nan")}
            continue
        sign = 1.0 if roc_auc_score(c_tr, s_tr) >= roc_auc_score(c_tr, -s_tr) else -1.0
        s_tr, s_te = sign * s_tr, sign * s_te
        best_th, best_ba = None, -1.0
        for th in np.unique(s_tr):
            ba = balanced_accuracy_score(c_tr, (s_tr >= th).astype(int))
            if ba > best_ba:
                best_ba, best_th = ba, th
        pred = (s_te >= best_th).astype(int)
        out[name] = {
            "auroc": float(roc_auc_score(c_te, s_te)),
            "acc": float(accuracy_score(c_te, pred)),
            "macro_f1": float(f1_score(c_te, pred, average="macro")),
            "balanced_acc": float(balanced_accuracy_score(c_te, pred)),
        }
    return out


def logistic_4way(Xtr, ytr, Xte, yte, c_te, m_te):
    clf = LogisticRegression(multi_class="multinomial", max_iter=5000)
    clf.fit(Xtr, ytr)
    probs = clf.predict_proba(Xte)
    pred = probs.argmax(1)
    p_c = probs[:, 2] + probs[:, 3]
    p_m = probs[:, 1] + probs[:, 3]
    return {
        "acc4": float(accuracy_score(yte, pred)),
        "f1_macro": float(f1_score(yte, pred, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(yte, pred)),
        "ece": ece(yte, probs),
        "brier": multiclass_brier(yte, probs),
        "nll": float(-np.mean(np.log(probs[np.arange(len(yte)), yte] + 1e-12))),
        "auroc_c": float(roc_auc_score(c_te, p_c)),
        "auroc_m": float(roc_auc_score(m_te, p_m)),
    }


# --------------------------------------------------------------------------- #
# layer C: selective prediction (unified answer source = greedy-ctx)
# --------------------------------------------------------------------------- #
def selective_prediction(score, hits, thresholds=(0.30, 0.40, 0.50, 0.60, 0.70)):
    """score: larger -> abstain. hits: 1 if greedy-ctx correct. Returns coverage/EM points + AURC."""
    score = np.asarray(score, dtype=float)
    hits = np.asarray(hits, dtype=float)
    points = []
    for th in thresholds:
        keep = score < th
        cov = float(keep.mean())
        em = float(hits[keep].mean()) if keep.sum() else float("nan")
        points.append((th, cov, em))
    return {"points": points, "aurc": _aurc(score, hits)}


def _ctx_hit(tk):
    return int(tk["tokens"][int(np.argmax(tk["z_ctx"]))] == tk["gold_tok"])


def _pri_hit(tk):
    return int(tk["tokens"][int(np.argmax(tk["z_pri"]))] == tk["gold_tok"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="popqa")
    args, _ = ap.parse_known_args()

    tag = args.data
    adir = config.ARTIFACT_DIR / tag

    samples = load_samples(adir / "data.jsonl")
    d = np.load(adir / "features.npz")
    X = d["X"]
    y = d["y"]
    c_star = d["c_star"]
    m_star = d["m_star"]
    with open(adir / "topk_logits.pkl", "rb") as f:
        topks = pickle.load(f)

    # Use the persisted split when available; legacy artifacts reconstruct it.
    split_path = adir / "split.npz"
    if split_path.exists():
        saved_split = np.load(split_path)
        splits = {name: saved_split[name] for name in ["tr", "va", "te"]}
    else:
        gid = group_ids(samples)
        splits = make_group_split(gid, m_star, seed=config.SEED)
    tr, va, te = splits["tr"], splits["va"], splits["te"]
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    # rebuild the trained estimator from checkpoint (no LM, no retrain)
    ckpt = torch.load(adir / "estimator.pt", map_location="cpu", weights_only=False)
    model = Estimator(ckpt["feat_dim"])
    model.load_state_dict(ckpt["state_dict"])
    T, norm = ckpt["T"], ckpt["norm"]
    probs_te = predict_probs(model, Xte, T, norm, "cpu")
    pred_te = probs_te.argmax(1)
    p_c_te, p_m_te = p_cm_from_probs(probs_te)

    # standardize identically to the trained MLP for a fair logistic comparison
    mu, sd = norm
    Xtr_s = (Xtr - mu) / sd
    Xte_s = (Xte - mu) / sd

    print("=" * 72)
    print(f"# Credence re-focused evaluation  |  data={tag}  |  n_test={len(yte)}")
    print(f"  (split reproduced identically to run_experiment.py; test n={len(yte)})")

    # ---------------- layer A ----------------
    print("\n## A. discrimination / classification quality")
    print("  (the 'classification' answer; hand-crafted signals are 1-D, so they")
    print("   can only be thresholded into a c* binary classifier — not 4-way)")
    print(f"  {'method':<22}{'4way-F1':>9}{'balAcc':>9}{'AUROCc':>9}{'AUROCm':>9}")
    print(f"  {'MLP (ours)':<22}{f1_score(yte, pred_te, average='macro'):>9.3f}"
          f"{balanced_accuracy_score(yte, pred_te):>9.3f}"
          f"{roc_auc_score(c_star[te], p_c_te):>9.3f}"
          f"{roc_auc_score(m_star[te], p_m_te):>9.3f}")
    lr = logistic_4way(Xtr_s, ytr, Xte_s, yte, c_star[te], m_star[te])
    print(f"  {'logistic (linear)':<22}{lr['f1_macro']:>9.3f}{lr['balanced_acc']:>9.3f}"
          f"{lr['auroc_c']:>9.3f}{lr['auroc_m']:>9.3f}")

    # per-state one-vs-rest AUROC
    print("\n  per-state one-vs-rest AUROC (ours):")
    ps = per_state_auroc(yte, probs_te)
    for k in STATE_NAMES.values():
        print(f"    {k:<14} {ps[k]:.3f}")
    print(f"    {'macro':<14} {ps['macro']:.3f}")

    # hand-crafted single-signal c* classifier (fair: threshold fit on train)
    ssc = single_signal_classifier(Xtr, c_star[tr], Xte, c_star[te])
    best = max(ssc.items(), key=lambda kv: kv[1]["auroc"] if kv[1]["auroc"] == kv[1]["auroc"] else -1)
    print(f"\n  hand-crafted single-signal -> c* binary classifier (best by AUROC):")
    print(f"    best = {best[0]}:  AUROC={best[1]['auroc']:.3f}  "
          f"macro-F1={best[1]['macro_f1']:.3f}  balAcc={best[1]['balanced_acc']:.3f}")
    print(f"    (top-5 by AUROC:", end="")
    top5 = sorted(ssc.items(), key=lambda kv: -(kv[1]["auroc"] if kv[1]["auroc"] == kv[1]["auroc"] else -1))[:5]
    print("  ".join(f"{n}={v['auroc']:.3f}" for n, v in top5) + ")")

    # ---------------- layer B ----------------
    print("\n## B. calibration quality  (scalar signals have no probability -> n/a)")
    print(f"  {'method':<22}{'ECE':>9}{'Brier':>9}{'NLL':>9}")
    print(f"  {'MLP (ours)':<22}{ece(yte, probs_te):>9.3f}"
          f"{multiclass_brier(yte, probs_te):>9.3f}"
          f"{-np.mean(np.log(probs_te[np.arange(len(yte)), yte] + 1e-12)):>9.3f}")
    print(f"  {'logistic (linear)':<22}{lr['ece']:>9.3f}{lr['brier']:>9.3f}{lr['nll']:>9.3f}")
    print(f"  (random 4-way baseline: Brier=0.750, NLL=1.386 — lower is better)")

    # ---------------- layer C ----------------
    print("\n## C. selective prediction (reject the likely-wrong; answer source = greedy-ctx)")
    hits = np.array([_ctx_hit(tk) for tk in topks])[te]
    ours = selective_prediction(probs_te[:, 0], hits)          # P(double-wrong)
    # hand-crafted: conf_ctx (best context-confidence signal); low conf -> abstain
    conf_ctx = Xte[:, SIGNAL_IDX["conf_ctx"]]
    hand = selective_prediction(1.0 - conf_ctx, hits)
    print(f"  {'threshold':>10}{'cov(ours)':>12}{'EM(ours)':>10}{'cov(hand)':>12}{'EM(hand)':>10}")
    for (th, co, eo), (_, ch, eh) in zip(ours["points"], hand["points"]):
        print(f"  {th:>10.2f}{co:>12.3f}{eo:>10.3f}{ch:>12.3f}{eh:>10.3f}")
    print(f"  AURC (lower better):  ours(P(dw))={ours['aurc']:.3f}   "
          f"hand-crafted(1-conf_ctx)={hand['aurc']:.3f}")

    # ---------------- layer D ----------------
    print("\n## D. fixed-candidate hard-routing diagnostic")
    hit_ctx = np.array([_ctx_hit(tk) for tk in topks])[te]
    hit_pri = np.array([_pri_hit(tk) for tk in topks])[te]
    oracle = 0.0
    for k in range(4):
        mask = yte == k
        if mask.sum() == 0:
            continue
        frac = mask.sum() / len(yte)
        best_k = max(hit_ctx[mask].mean(), hit_pri[mask].mean())
        oracle += best_k * frac
        print(f"  {STATE_NAMES[k]:<14} share={frac:>5.2f}  best(ctx,pri) EM="
              f"{hit_ctx[mask].mean():.3f}/{hit_pri[mask].mean():.3f} -> {best_k:.3f}")
    print(f"  fixed-candidate oracle EM = {oracle:.3f}")
    print("  (upper bound only for hard selection between the cached context and closed-book candidates)")
    print("=" * 72)


if __name__ == "__main__":
    main()
