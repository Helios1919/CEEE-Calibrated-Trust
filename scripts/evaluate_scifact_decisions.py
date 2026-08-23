#!/usr/bin/env python3
"""Outcome-grounded Gate-C decisions with empirical deeper retrieval."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import config
from estimator import Estimator, predict_probs


ACTIONS = ["use_context", "use_memory", "fuse", "abstain", "retrieve_again"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fit_binary(Xtr, ytr, Xte):
    ytr = np.asarray(ytr)
    if len(np.unique(ytr)) == 1:
        return np.full(len(Xte), float(ytr[0]))
    model = LogisticRegression(max_iter=5000).fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def independent_probs(pc, pm):
    return np.column_stack([
        (1 - pc) * (1 - pm),
        (1 - pc) * pm,
        pc * (1 - pm),
        pc * pm,
    ])


def aligned_probs(model, X):
    probs = np.zeros((len(X), 4), dtype=float)
    probs[:, model.classes_.astype(int)] = model.predict_proba(X)
    return probs


def best_scalar_probs(Xtr, ytr, Xva, yva, Xte):
    best = None
    for feature in range(Xtr.shape[1]):
        model = LogisticRegression(max_iter=5000).fit(
            Xtr[:, [feature]], ytr
        )
        pva = aligned_probs(model, Xva[:, [feature]])
        score = log_loss(yva, pva, labels=np.arange(4))
        if best is None or score < best[0]:
            best = (score, feature, model)
    score, feature, model = best
    return aligned_probs(model, Xte[:, [feature]]), feature, float(score)


def retrieval_features(retrieval_rows):
    features = []
    for row in retrieval_rows:
        scores = np.asarray([passage["score"] for passage in row["passages"]])
        features.append([
            scores[0],
            scores[0] - scores[1],
            scores.mean(),
            scores.std(),
            np.log1p(len(row["claim"].split())),
            np.mean([len(passage["text"].split()) for passage in row["passages"]]),
        ])
    return np.asarray(features, dtype=float)


def regimes():
    return {
        "safety_first": {
            "answer_error": 1.0,
            "unsupported": 0.35,
            "abstain": 0.25,
            "retrieve_cost": 0.10,
        },
        "balanced": {
            "answer_error": 1.0,
            "unsupported": 0.20,
            "abstain": 0.45,
            "retrieve_cost": 0.20,
        },
        "coverage_first": {
            "answer_error": 1.0,
            "unsupported": 0.10,
            "abstain": 0.70,
            "retrieve_cost": 0.35,
        },
    }


def action_costs(rows5, rows10, regime):
    context_wrong = 1.0 - np.asarray([
        row["context_conditioned"]["correct"] for row in rows5
    ])
    memory_wrong = 1.0 - np.asarray([
        row["closedbook"]["correct"] for row in rows5
    ])
    fused_wrong = 1.0 - np.asarray([row["fused"]["correct"] for row in rows5])
    retrieve_wrong = 1.0 - np.asarray([
        row["context_conditioned"]["correct"] for row in rows10
    ])
    unresolved5 = 1.0 - np.asarray([row["c_star"] for row in rows5])
    unresolved10 = 1.0 - np.asarray([row["c_star"] for row in rows10])

    costs = np.empty((len(rows5), len(ACTIONS)), dtype=float)
    costs[:, 0] = (regime["answer_error"] * context_wrong
                   + regime["unsupported"] * unresolved5)
    costs[:, 1] = regime["answer_error"] * memory_wrong
    costs[:, 2] = (regime["answer_error"] * fused_wrong
                   + regime["unsupported"] * unresolved5)
    costs[:, 3] = regime["abstain"]
    costs[:, 4] = (regime["retrieve_cost"]
                   + regime["answer_error"] * retrieve_wrong
                   + regime["unsupported"] * unresolved10)
    return costs


def state_conditional_costs(ytr, costs):
    matrix = np.empty((len(ACTIONS), 4), dtype=float)
    for state in range(4):
        matrix[:, state] = costs[ytr == state].mean(axis=0)
    return matrix


def select_state(probs, conditional):
    expected = probs @ conditional.T
    return expected.argmin(1), expected


def direct_expected_costs(Xtr, Xte, rows5, rows10, tr, regime):
    context_wrong = [1 - rows5[i]["context_conditioned"]["correct"] for i in tr]
    memory_wrong = [1 - rows5[i]["closedbook"]["correct"] for i in tr]
    fused_wrong = [1 - rows5[i]["fused"]["correct"] for i in tr]
    retrieval_wrong = [1 - rows10[i]["context_conditioned"]["correct"] for i in tr]
    unresolved5 = [1 - rows5[i]["c_star"] for i in tr]
    unresolved10 = [1 - rows10[i]["c_star"] for i in tr]
    return np.column_stack([
        regime["answer_error"] * fit_binary(Xtr, context_wrong, Xte)
        + regime["unsupported"] * fit_binary(Xtr, unresolved5, Xte),
        regime["answer_error"] * fit_binary(Xtr, memory_wrong, Xte),
        regime["answer_error"] * fit_binary(Xtr, fused_wrong, Xte)
        + regime["unsupported"] * fit_binary(Xtr, unresolved5, Xte),
        np.full(len(Xte), regime["abstain"]),
        regime["retrieve_cost"]
        + regime["answer_error"] * fit_binary(Xtr, retrieval_wrong, Xte)
        + regime["unsupported"] * fit_binary(Xtr, unresolved10, Xte),
    ])


def summarize(actions, costs, expected=None):
    chosen = costs[np.arange(len(costs)), actions]
    oracle = costs.min(axis=1)
    counts = np.bincount(actions, minlength=len(ACTIONS))
    result = {
        "mean_cost": float(chosen.mean()),
        "oracle_cost": float(oracle.mean()),
        "regret_to_oracle": float((chosen - oracle).mean()),
        "answer_rate": float(np.isin(actions, [0, 1, 2, 4]).mean()),
        "fusion_rate": float((actions == 2).mean()),
        "abstention_rate": float((actions == 3).mean()),
        "retrieve_again_rate": float((actions == 4).mean()),
        "action_counts": {name: int(counts[i]) for i, name in enumerate(ACTIONS)},
    }
    if expected is not None:
        predicted = expected[np.arange(len(costs)), actions]
        result["mean_predicted_cost"] = float(predicted.mean())
        result["calibration_gap"] = float(chosen.mean() - predicted.mean())
    return result, chosen


def bootstrap_delta(joint_loss, alternatives, n_boot, seed):
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in alternatives}
    for _ in range(n_boot):
        idx = rng.integers(0, len(joint_loss), len(joint_loss))
        for name, losses in alternatives.items():
            draws[name].append(float(losses[idx].mean() - joint_loss[idx].mean()))
    return {
        name: {
            "joint_advantage_mean": float(np.mean(values)),
            "ci95": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
        }
        for name, values in draws.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="scifact")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output")
    args = parser.parse_args()

    adir = ROOT / "artifacts" / args.data
    rows5 = load_jsonl(adir / "complete_verdicts.jsonl")
    rows10 = load_jsonl(adir / "complete_verdicts_top10.jsonl")
    retrieval5 = load_jsonl(adir / "retrieval_pilot.jsonl")
    if not (len(rows5) == len(rows10) == len(retrieval5)):
        raise ValueError("SciFact artifacts have inconsistent claim counts")
    if any(a["id"] != b["id"] for a, b in zip(rows5, rows10)):
        raise ValueError("Top-5 and top-10 verdict artifacts are not aligned")

    features = np.load(adir / "features.npz")
    X, y = features["X"], features["y"]
    split = np.load(adir / "split.npz")
    tr, va, te = split["tr"], split["va"], split["te"]

    checkpoint = torch.load(adir / "estimator.pt", map_location="cpu", weights_only=False)
    estimator = Estimator(checkpoint["feat_dim"])
    estimator.load_state_dict(checkpoint["state_dict"])
    p_joint = predict_probs(
        estimator, X[te], checkpoint["T"], checkpoint["norm"], "cpu"
    )

    mu, sd = checkpoint["norm"]
    Xtr, Xva, Xte = (X[tr] - mu) / sd, (X[va] - mu) / sd, (X[te] - mu) / sd
    linear = LogisticRegression(max_iter=5000).fit(Xtr, y[tr])
    p_linear = aligned_probs(linear, Xte)
    p_scalar, scalar_feature, scalar_validation_nll = best_scalar_probs(
        Xtr, y[tr], Xva, y[va], Xte
    )
    pc = fit_binary(Xtr, features["c_star"][tr], Xte)
    pm = fit_binary(Xtr, features["m_star"][tr], Xte)
    p_independent = independent_probs(pc, pm)

    R = retrieval_features(retrieval5)
    r_mu, r_sd = R[tr].mean(0), R[tr].std(0) + 1e-6
    Rtr, Rte = (R[tr] - r_mu) / r_sd, (R[te] - r_mu) / r_sd

    result = {
        "schema_version": 1,
        "purpose": "Gate-C outcome-grounded decisions with empirical deeper retrieval",
        "actions": ACTIONS,
        "claim_count": len(rows5),
        "test_count": len(te),
        "retrieve_again_protocol": (
            "Extend deterministic TF-IDF retrieval from top five to top ten and run a "
            "new context-conditioned complete verdict. Its realized errors and unresolved "
            "evidence are charged in addition to a fixed retrieval cost."
        ),
        "support_policy": (
            "An answer is conservatively unsupported when no matching annotated gold-evidence "
            "document is retrieved. This does not assert that unannotated abstracts are neutral."
        ),
        "regimes": {},
    }

    for name, regime in regimes().items():
        costs = action_costs(rows5, rows10, regime)
        conditional = state_conditional_costs(y[tr], costs[tr])
        joint_actions, joint_expected = select_state(p_joint, conditional)
        linear_actions, linear_expected = select_state(p_linear, conditional)
        scalar_actions, scalar_expected = select_state(p_scalar, conditional)
        independent_actions, independent_expected = select_state(p_independent, conditional)

        direct_expected = direct_expected_costs(
            Xtr, Xte, rows5, rows10, tr, regime
        )
        direct_actions = direct_expected.argmin(1)
        retrieval_expected = direct_expected_costs(
            Rtr, Rte, rows5, rows10, tr, regime
        )
        retrieval_actions = retrieval_expected.argmin(1)

        policies = {}
        losses = {}
        candidates = [
            ("joint_state_mlp", joint_actions, joint_expected),
            ("joint_state_linear", linear_actions, linear_expected),
            ("best_scalar_state", scalar_actions, scalar_expected),
            ("independent_state_product", independent_actions, independent_expected),
            ("direct_answer_risk", direct_actions, direct_expected),
            ("retrieval_only_risk", retrieval_actions, retrieval_expected),
        ]
        for policy, actions, expected in candidates:
            policies[policy], losses[policy] = summarize(actions, costs[te], expected)
        for action_index, action_name in enumerate(ACTIONS):
            fixed = np.full(len(te), action_index, dtype=int)
            key = f"always_{action_name}"
            policies[key], losses[key] = summarize(fixed, costs[te])

        alternatives = {
            key: value for key, value in losses.items() if key != "joint_state_mlp"
        }
        result["regimes"][name] = {
            "costs": regime,
            "best_scalar_feature_index": scalar_feature,
            "best_scalar_validation_nll": scalar_validation_nll,
            "state_conditional_action_costs": conditional.tolist(),
            "policies": policies,
            "bootstrap_joint_state_mlp_advantage": bootstrap_delta(
                losses["joint_state_mlp"], alternatives, args.bootstrap, config.SEED
            ),
        }

    output = Path(args.output) if args.output else adir / "gate_c_decision_evaluation.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
