#!/usr/bin/env python3
"""Stage-1 falsification analysis on a frozen Credence artifact.

This script asks whether a joint four-state model adds information beyond two
independent marginal models. It does not reload the language model.
"""

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import sklearn
import torch
import transformers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, log_loss, roc_auc_score)

import config
from estimator import Estimator, predict_probs
from analyze import group_ids, load_samples, make_group_split, multiclass_brier
from features.extract import FEATURE_NAMES


STATE_NAMES = ["neither_available", "memory_only", "context_only", "both_available"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_value(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def state_metrics(y, probs, c, m):
    pred = probs.argmax(1)
    pc = probs[:, 2] + probs[:, 3]
    pm = probs[:, 1] + probs[:, 3]
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "nll": float(log_loss(y, probs, labels=[0, 1, 2, 3])),
        "brier": multiclass_brier(y, probs),
        "auroc_context": float(roc_auc_score(c, pc)),
        "auprc_context": float(average_precision_score(c, pc)),
        "auroc_memory": float(roc_auc_score(m, pm)),
        "auprc_memory": float(average_precision_score(m, pm)),
        "auroc_neither": float(roc_auc_score(y == 0, probs[:, 0])),
        "auprc_neither": float(average_precision_score(y == 0, probs[:, 0])),
    }


def independent_probs(pc, pm):
    return np.column_stack([
        (1 - pc) * (1 - pm),
        (1 - pc) * pm,
        pc * (1 - pm),
        pc * pm,
    ])


def fit_logistic(Xtr, ytr, Xte):
    clf = LogisticRegression(max_iter=5000, class_weight=None)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)


def scalar_baselines(Xtr, ytr, Xte, yte):
    """Calibrate each individual feature into a four-state probability model."""
    results = {}
    for index, name in enumerate(FEATURE_NAMES):
        probs = fit_logistic(Xtr[:, [index]], ytr, Xte[:, [index]])
        results[name] = {
            "probs": probs,
            "nll": float(log_loss(yte, probs, labels=[0, 1, 2, 3])),
        }
    return results


def action_losses(probs, y, cost_matrix):
    """Choose the minimum posterior-risk action and return its realized losses."""
    expected = probs @ cost_matrix.T
    actions = expected.argmin(1)
    return cost_matrix[actions, y], actions


def bootstrap_action_delta(y, p_joint, alternatives, groups, cost_matrices,
                           n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    out = {}
    for cost_name, matrix in cost_matrices.items():
        joint_loss, _ = action_losses(p_joint, y, matrix)
        alt_losses = {name: action_losses(probs, y, matrix)[0]
                      for name, probs in alternatives.items()}
        draws = {name: [] for name in alternatives}
        for _ in range(n_boot):
            selected = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([np.where(groups == group)[0] for group in selected])
            for name, losses in alt_losses.items():
                draws[name].append(float(losses[idx].mean() - joint_loss[idx].mean()))
        out[cost_name] = {
            name: {
                "joint_advantage_mean": float(np.mean(values)),
                "ci95": [float(np.quantile(values, 0.025)),
                         float(np.quantile(values, 0.975))],
            }
            for name, values in draws.items()
        }
    return out


def bootstrap_delta(y, c, m, p_joint, p_ind, groups, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    deltas = {"macro_f1": [], "nll": [], "brier": [], "auroc_neither": []}
    for _ in range(n_boot):
        selected = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in selected])
        # Resampling a group more than once is intentional; indices are concatenated.
        if len(np.unique(y[idx])) < 2 or len(np.unique((y[idx] == 0).astype(int))) < 2:
            continue
        a = state_metrics(y[idx], p_joint[idx], c[idx], m[idx])
        b = state_metrics(y[idx], p_ind[idx], c[idx], m[idx])
        for key in deltas:
            # Positive means joint is better for F1/AUROC; lower is better for NLL/Brier.
            value = a[key] - b[key] if key in {"macro_f1", "auroc_neither"} else b[key] - a[key]
            deltas[key].append(value)
    out = {}
    for key, values in deltas.items():
        arr = np.asarray(values)
        out[key] = {
            "joint_advantage_mean": float(arr.mean()),
            "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="popqa")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--output")
    args = ap.parse_args()

    adir = ROOT / "artifacts" / args.data
    samples = load_samples(adir / "data.jsonl")
    data = np.load(adir / "features.npz")
    X, y = data["X"], data["y"]
    c, m = data["c_star"], data["m_star"]
    groups = group_ids(samples)
    split_path = adir / "split.npz"
    if split_path.exists():
        saved_split = np.load(split_path)
        splits = {name: saved_split[name] for name in ["tr", "va", "te"]}
        if "group_id" in saved_split:
            groups = saved_split["group_id"]
    else:
        splits = make_group_split(groups, m, config.SEED)
    tr, te = splits["tr"], splits["te"]

    ckpt = torch.load(adir / "estimator.pt", map_location="cpu", weights_only=False)
    model = Estimator(ckpt["feat_dim"])
    model.load_state_dict(ckpt["state_dict"])
    p_mlp = predict_probs(model, X[te], ckpt["T"], ckpt["norm"], "cpu")

    mu, sd = ckpt["norm"]
    Xtr = (X[tr] - mu) / sd
    Xte = (X[te] - mu) / sd
    p_joint_linear = fit_logistic(Xtr, y[tr], Xte)
    p_c = fit_logistic(Xtr, c[tr], Xte)[:, 1]
    p_m = fit_logistic(Xtr, m[tr], Xte)[:, 1]
    p_ind = independent_probs(p_c, p_m)
    p_neither = fit_logistic(Xtr, (y[tr] == 0).astype(int), Xte)[:, 1]
    scalar = scalar_baselines(Xtr, y[tr], Xte, y[te])
    best_scalar_name = min(scalar, key=lambda name: scalar[name]["nll"])
    p_scalar = scalar[best_scalar_name]["probs"]

    # Rows are actions [use_context, use_memory, abstain]; columns are the four states.
    # These matrices are fixed before observing results and encode three operational regimes.
    cost_matrices = {
        "error_averse": np.asarray([[1.0, 1.0, 0.0, 0.0],
                                    [1.0, 0.0, 1.0, 0.0],
                                    [0.20, 0.20, 0.20, 0.20]]),
        "coverage_averse": np.asarray([[1.0, 1.0, 0.0, 0.0],
                                       [1.0, 0.0, 1.0, 0.0],
                                       [0.45, 0.45, 0.45, 0.45]]),
        "context_harm_averse": np.asarray([[1.0, 1.50, 0.0, 0.0],
                                           [1.0, 0.0, 1.0, 0.0],
                                           [0.30, 0.30, 0.30, 0.30]]),
    }

    metrics = {
        "joint_mlp": state_metrics(y[te], p_mlp, c[te], m[te]),
        "joint_linear": state_metrics(y[te], p_joint_linear, c[te], m[te]),
        "independent_binary_product": state_metrics(y[te], p_ind, c[te], m[te]),
        "direct_neither_binary": {
            "auroc_neither": float(roc_auc_score(y[te] == 0, p_neither)),
            "auprc_neither": float(average_precision_score(y[te] == 0, p_neither)),
        },
        "best_scalar_4way": {
            "feature": best_scalar_name,
            **state_metrics(y[te], p_scalar, c[te], m[te]),
        },
    }

    boot = bootstrap_delta(y[te], c[te], m[te], p_mlp, p_ind, groups[te], args.bootstrap)
    action_boot = bootstrap_action_delta(
        y[te], p_mlp,
        {"independent_product": p_ind, "best_scalar": p_scalar,
         "joint_linear": p_joint_linear},
        groups[te], cost_matrices, args.bootstrap, config.SEED)

    files = [adir / n for n in ["data.jsonl", "features.npz", "topk_logits.pkl",
                                "estimator.pt", "split.npz"]]
    manifest = {
        "schema_version": 1,
        "status": "controlled-stage1-baseline",
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "dataset": args.data,
        "model": config.MODEL_NAME,
        "sample_count": int(len(y)),
        "test_count": int(len(te)),
        "split": "persisted item_id groups, stratified by closed-book label",
        "state_names": {str(i): name for i, name in enumerate(STATE_NAMES)},
        "legacy_state_names": {str(k): v for k, v in config.STATE_NAMES.items()},
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "artifact_sha256": {p.name: sha256(p) for p in files},
        "limitations": [
            "Context labels are assigned by synthetic construction.",
            "Closed-book labels come from one greedy generation.",
            "Top-k first-token correctness is not complete-answer correctness.",
            "Action costs are preregistered diagnostic matrices, not user-study estimates.",
        ],
    }

    result = {
        "manifest": manifest,
        "metrics": metrics,
        "bootstrap_joint_mlp_vs_independent_product": boot,
        "action_cost_matrices": {name: matrix.tolist() for name, matrix in cost_matrices.items()},
        "bootstrap_action_loss_joint_mlp_vs_baselines": action_boot,
        "interpretation_rule": "A positive bootstrap joint advantage favors the joint MLP. A CI crossing zero is inconclusive.",
    }
    out = Path(args.output) if args.output else adir / "stage1_falsification.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (adir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print("\nBootstrap joint MLP advantage over independent product:")
    print(json.dumps(boot, indent=2))
    print("\nBootstrap action-loss advantage:")
    print(json.dumps(action_boot, indent=2))
    print(f"\nWrote {out} and {adir / 'manifest.json'}")


if __name__ == "__main__":
    main()
