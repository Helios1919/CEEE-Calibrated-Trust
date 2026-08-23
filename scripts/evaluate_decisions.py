#!/usr/bin/env python3
"""Outcome-grounded cost-sensitive decisions for the controlled PopQA artifact.

Policies choose among cached context, memory, and fused complete answers,
abstention, and a simulated perfect retrieve-again action. The latter is a
controlled value-of-information diagnostic, not an implemented retriever.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import config
from analyze import load_samples, multiclass_brier
from estimator import Estimator, predict_probs


ACTIONS = ["use_context", "use_memory", "fuse", "abstain", "retrieve_again"]


def load_outcomes(path):
    with open(path, encoding="utf-8") as handle:
        return {row["id"]: row for row in map(json.loads, handle) if row}


def action_costs(rows, regime):
    """Return realized [N,5] costs for the five available actions."""
    ctx_wrong = 1.0 - np.asarray([
        row["context_conditioned"]["entity_exact_match"] for row in rows])
    mem_wrong = 1.0 - np.asarray([
        row["closedbook"]["entity_exact_match"] for row in rows])
    unsupported = 1.0 - np.asarray([
        row["context_conditioned"]["supported_by_context"] for row in rows])

    fuse_wrong = 1.0 - np.asarray([
        row["fused"]["entity_exact_match"] for row in rows])
    fuse_unsupported = 1.0 - np.asarray([
        row["fused"]["supported_by_context"] for row in rows])

    costs = np.empty((len(rows), len(ACTIONS)), dtype=float)
    costs[:, 0] = regime["answer_error"] * ctx_wrong + regime["unsupported"] * unsupported
    costs[:, 1] = regime["answer_error"] * mem_wrong
    costs[:, 2] = (regime["answer_error"] * fuse_wrong
                   + regime["unsupported"] * fuse_unsupported)
    costs[:, 3] = regime["abstain"]
    costs[:, 4] = regime["retrieve_again"]
    return costs


def regimes():
    return {
        "safety_first": {
            "answer_error": 1.0, "unsupported": 0.35,
            "abstain": 0.25, "retrieve_again": 0.15,
        },
        "balanced": {
            "answer_error": 1.0, "unsupported": 0.20,
            "abstain": 0.45, "retrieve_again": 0.30,
        },
        "coverage_first": {
            "answer_error": 1.0, "unsupported": 0.10,
            "abstain": 0.70, "retrieve_again": 0.55,
        },
    }


def fit_action_model(Xtr, rows_tr, Xte):
    """Direct baseline: predict realized answer errors without the state label."""
    ctx_wrong = np.asarray([
        1 - row["context_conditioned"]["entity_exact_match"] for row in rows_tr])
    mem_wrong = np.asarray([
        1 - row["closedbook"]["entity_exact_match"] for row in rows_tr])
    unsupported = np.asarray([
        1 - row["context_conditioned"]["supported_by_context"] for row in rows_tr])
    fuse_wrong = np.asarray([
        1 - row["fused"]["entity_exact_match"] for row in rows_tr])
    fuse_unsupported = np.asarray([
        1 - row["fused"]["supported_by_context"] for row in rows_tr])
    return (
        fit_binary(Xtr, ctx_wrong, Xte),
        fit_binary(Xtr, mem_wrong, Xte),
        fit_binary(Xtr, unsupported, Xte),
        fit_binary(Xtr, fuse_wrong, Xte),
        fit_binary(Xtr, fuse_unsupported, Xte),
    )


def fit_binary(Xtr, ytr, Xte):
    model = LogisticRegression(max_iter=5000)
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def state_conditional_costs(ytr, costs_tr):
    matrix = np.empty((len(ACTIONS), 4), dtype=float)
    for state in range(4):
        matrix[:, state] = costs_tr[ytr == state].mean(axis=0)
    return matrix


def select_from_state(probs, state_costs):
    expected = probs @ state_costs.T
    return expected.argmin(1), expected


def summarize_policy(actions, costs, expected=None):
    chosen = costs[np.arange(len(costs)), actions]
    oracle = costs.min(axis=1)
    counts = np.bincount(actions, minlength=len(ACTIONS))
    result = {
        "mean_cost": float(chosen.mean()),
        "oracle_cost": float(oracle.mean()),
        "regret_to_oracle": float((chosen - oracle).mean()),
        "answer_rate": float(np.isin(actions, [0, 1, 2]).mean()),
        "fusion_rate": float((actions == 2).mean()),
        "abstention_rate": float((actions == 3).mean()),
        "retrieval_rate": float((actions == 4).mean()),
        "action_counts": {name: int(counts[i]) for i, name in enumerate(ACTIONS)},
    }
    if expected is not None:
        predicted = expected[np.arange(len(costs)), actions]
        result["mean_predicted_cost"] = float(predicted.mean())
        result["calibration_gap"] = float(chosen.mean() - predicted.mean())
    return result, chosen


def bootstrap_delta(joint_loss, alternatives, groups, n_boot, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    draws = {name: [] for name in alternatives}
    for _ in range(n_boot):
        selected = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.where(groups == group)[0] for group in selected])
        for name, loss in alternatives.items():
            draws[name].append(float(loss[idx].mean() - joint_loss[idx].mean()))
    return {
        name: {
            "joint_advantage_mean": float(np.mean(values)),
            "ci95": [float(np.quantile(values, 0.025)),
                     float(np.quantile(values, 0.975))],
        }
        for name, values in draws.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="popqa")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output")
    args = parser.parse_args()

    adir = ROOT / "artifacts" / args.data
    samples = load_samples(adir / "data.jsonl")
    outcomes = load_outcomes(adir / "complete_answers_all.jsonl")
    if len(outcomes) != len(samples):
        raise ValueError(
            f"Complete-answer cache has {len(outcomes)} rows for {len(samples)} samples")
    rows = [outcomes[sample["id"]] for sample in samples]

    features = np.load(adir / "features.npz")
    X, y = features["X"], features["y"]
    saved = np.load(adir / "split.npz")
    tr, te = saved["tr"], saved["te"]
    groups = saved["group_id"]

    checkpoint = torch.load(adir / "estimator.pt", map_location="cpu", weights_only=False)
    estimator = Estimator(checkpoint["feat_dim"])
    estimator.load_state_dict(checkpoint["state_dict"])
    p_joint = predict_probs(
        estimator, X[te], checkpoint["T"], checkpoint["norm"], "cpu")

    mu, sd = checkpoint["norm"]
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    joint_linear = LogisticRegression(max_iter=5000)
    joint_linear.fit(Xtr, y[tr])
    p_linear = joint_linear.predict_proba(Xte)

    c_model = LogisticRegression(max_iter=5000).fit(Xtr, features["c_star"][tr])
    m_model = LogisticRegression(max_iter=5000).fit(Xtr, features["m_star"][tr])
    pc, pm = c_model.predict_proba(Xte)[:, 1], m_model.predict_proba(Xte)[:, 1]
    p_independent = np.column_stack([
        (1 - pc) * (1 - pm), (1 - pc) * pm, pc * (1 - pm), pc * pm])

    result = {
        "schema_version": 1,
        "purpose": "Outcome-grounded cost-sensitive action evaluation",
        "actions": ACTIONS,
        "test_count": int(len(te)),
        "retrieve_again_assumption": (
            "Simulated perfect second retrieval returns a correct supported answer at fixed cost. "
            "This is a value-of-information diagnostic, not an implemented retriever."),
        "regimes": {},
    }

    for name, regime in regimes().items():
        all_costs = action_costs(rows, regime)
        costs_tr, costs_te = all_costs[tr], all_costs[te]
        conditional = state_conditional_costs(y[tr], costs_tr)

        joint_actions, joint_expected = select_from_state(p_joint, conditional)
        linear_actions, linear_expected = select_from_state(p_linear, conditional)
        ind_actions, ind_expected = select_from_state(p_independent, conditional)

        (p_ctx_wrong, p_mem_wrong, p_unsupported,
         p_fuse_wrong, p_fuse_unsupported) = fit_action_model(
            Xtr, [rows[int(i)] for i in tr], Xte)
        direct_expected = np.column_stack([
            regime["answer_error"] * p_ctx_wrong
            + regime["unsupported"] * p_unsupported,
            regime["answer_error"] * p_mem_wrong,
            regime["answer_error"] * p_fuse_wrong
            + regime["unsupported"] * p_fuse_unsupported,
            np.full(len(te), regime["abstain"]),
            np.full(len(te), regime["retrieve_again"]),
        ])
        direct_actions = direct_expected.argmin(1)

        policies = {}
        losses = {}
        for policy, actions, expected in [
            ("joint_state_mlp", joint_actions, joint_expected),
            ("joint_state_linear", linear_actions, linear_expected),
            ("independent_state_product", ind_actions, ind_expected),
            ("direct_answer_risk", direct_actions, direct_expected),
        ]:
            policies[policy], losses[policy] = summarize_policy(
                actions, costs_te, expected)

        fixed = {
            "always_context": np.zeros(len(te), dtype=int),
            "always_memory": np.ones(len(te), dtype=int),
            "always_fuse": np.full(len(te), 2, dtype=int),
            "always_abstain": np.full(len(te), 3, dtype=int),
            "always_retrieve_again": np.full(len(te), 4, dtype=int),
        }
        for policy, actions in fixed.items():
            policies[policy], losses[policy] = summarize_policy(actions, costs_te)

        alternatives = {key: value for key, value in losses.items()
                        if key != "joint_state_mlp"}
        result["regimes"][name] = {
            "costs": regime,
            "state_conditional_action_costs": conditional.tolist(),
            "policies": policies,
            "bootstrap_joint_state_mlp_advantage": bootstrap_delta(
                losses["joint_state_mlp"], alternatives, groups[te],
                args.bootstrap, config.SEED),
        }

    output = Path(args.output) if args.output else adir / "decision_evaluation.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
