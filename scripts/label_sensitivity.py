#!/usr/bin/env python3
"""Audit closed-book labels and test Stage-1 conclusions across label policies.

The strict policy requires the complete gold answer. The conservative policy
adds only registered canonical surface forms and relation-scoped media-type
suffixes. The permissive policy reproduces the legacy bidirectional substring
rule as an intentionally optimistic upper envelope.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import config
from analyze import group_ids, load_samples, multiclass_brier
from data.popqa import make_items
from estimator import predict_probs, train_estimator
from stage1_falsification import (action_losses, bootstrap_action_delta,
                                  fit_logistic, independent_probs,
                                  state_metrics)


CANONICAL_EQUIVALENTS = {
    ("unitedstates", "unitedstatesofamerica"),
    ("football", "associationfootball"),
    ("china", "peoplesrepublicofchina"),
}
MEDIA_SUFFIXES = ("film", "music", "literature", "televisionseries", "videogame")


def normalize(text):
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def strict_match(generated, gold, aliases=None):
    generated = normalize(generated)
    answer = normalize(gold)
    return bool(generated) and bool(answer) and generated.startswith(answer)


def conservative_match(sample):
    generated = normalize(sample["closedbook_answer"])
    answers = [normalize(sample["gold"]),
               *(normalize(alias) for alias in sample.get("aliases", []))]
    matched = bool(generated) and any(answer and generated.startswith(answer)
                                      for answer in answers)
    if not matched:
        return False, "unmatched"
    if generated.startswith(normalize(sample["gold"])):
        return True, "gold_prefix"
    return True, "official_alias_prefix"


def alias_contains_match(sample):
    generated = normalize(sample["closedbook_answer"])
    answers = [normalize(sample["gold"]),
               *(normalize(alias) for alias in sample.get("aliases", []))]
    return bool(generated) and any(answer and answer in generated for answer in answers)


def permissive_match(sample):
    generated = normalize(sample["closedbook_answer"])
    gold = normalize(sample["gold"])
    return bool(generated) and bool(gold) and (gold in generated or generated in gold)


def make_labels(samples, policy):
    item_labels = {}
    reasons = Counter()
    audit_rows = []
    for sample in samples:
        item_id = sample["item_id"]
        if item_id in item_labels:
            continue
        if policy == "strict":
            matched = strict_match(sample["closedbook_answer"], sample["gold"])
            reason = "gold_prefix" if matched else "unmatched"
        elif policy == "conservative":
            matched, reason = conservative_match(sample)
        elif policy == "alias_contains":
            matched = alias_contains_match(sample)
            reason = "official_answer_anywhere" if matched else "unmatched"
        elif policy == "permissive":
            matched = permissive_match(sample)
            reason = "legacy_bidirectional_substring" if matched else "unmatched"
        else:
            raise ValueError(policy)
        item_labels[item_id] = int(matched)
        reasons[reason] += 1
        audit_rows.append({
            "item_id": item_id,
            "relation": sample["relation"],
            "question": sample["question"],
            "gold": sample["gold"],
            "closedbook_answer": sample["closedbook_answer"],
            "label": int(matched),
            "reason": reason,
        })
    m = np.asarray([item_labels[sample["item_id"]] for sample in samples])
    c = np.asarray([sample["c_star"] for sample in samples])
    return c * 2 + m, m, reasons, audit_rows


def load_split(adir):
    saved = np.load(adir / "split.npz")
    return {name: saved[name] for name in ["tr", "va", "te"]}


def cost_matrices():
    return {
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


def fit_policy(policy, X, c, y, m, splits, samples, adir, args):
    tr, te = splits["tr"], splits["te"]
    model, temperature, norm, _, _ = train_estimator(
        X, y, c, m, splits, epochs=config.EPOCHS, lr=config.LR,
        seeds=config.SEEDS, device=args.device)
    p_joint = predict_probs(model, X[te], temperature, norm, args.device)
    mu, sd = norm
    training = "retrained"

    Xtr = (X[tr] - mu) / sd
    Xte = (X[te] - mu) / sd
    p_linear = fit_logistic(Xtr, y[tr], Xte)
    p_c = fit_logistic(Xtr, c[tr], Xte)[:, 1]
    p_m = fit_logistic(Xtr, m[tr], Xte)[:, 1]
    p_independent = independent_probs(p_c, p_m)
    groups = group_ids(samples)[te]
    alternatives = {
        "independent_product": p_independent,
        "joint_linear": p_linear,
    }
    action_boot = bootstrap_action_delta(
        y[te], p_joint, alternatives, groups, cost_matrices(),
        args.bootstrap, config.SEED)
    return {
        "training": training,
        "positive_item_count": int(m.sum() // 2),
        "state_counts": {str(state): int((y == state).sum()) for state in range(4)},
        "metrics": {
            "joint_mlp": state_metrics(y[te], p_joint, c[te], m[te]),
            "joint_linear": state_metrics(y[te], p_linear, c[te], m[te]),
            "independent_product": state_metrics(y[te], p_independent, c[te], m[te]),
        },
        "action_loss_bootstrap": action_boot,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="popqa")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    args = parser.parse_args()

    adir = ROOT / "artifacts" / args.data
    samples = load_samples(adir / "data.jsonl")
    features = np.load(adir / "features.npz")
    X = features["X"]
    c = features["c_star"]
    splits = load_split(adir)

    policies = ["strict", "conservative", "alias_contains", "permissive"]
    item_aliases = {item["item_id"]: item.get("aliases", [])
                    for item in make_items(split=config.POPQA_SPLIT, n=config.POPQA_N,
                                           max_per_prop=config.POPQA_MAX_PER_PROP,
                                           seed=config.SEED)}
    enriched_samples = [{**sample, "aliases": item_aliases.get(sample["item_id"], [])}
                        for sample in samples]
    labels = {}
    audit_by_policy = {}
    reasons = {}
    for policy in policies:
        y, m, policy_reasons, audit_rows = make_labels(enriched_samples, policy)
        labels[policy] = (y, m)
        reasons[policy] = dict(policy_reasons)
        audit_by_policy[policy] = audit_rows

    policy_items = {
        policy: {row["item_id"]: row for row in audit_by_policy[policy]}
        for policy in policies
    }
    disagreements = []
    for item_id in policy_items["strict"]:
        rows = {policy: policy_items[policy][item_id] for policy in policies}
        if len({row["label"] for row in rows.values()}) > 1:
            disagreements.append({
                **rows["strict"],
                **{f"{policy}_label": row["label"] for policy, row in rows.items()},
                **{f"{policy}_reason": row["reason"] for policy, row in rows.items()},
            })

    results = {}
    for policy in policies:
        print(f"Fitting policy: {policy}", flush=True)
        y, m = labels[policy]
        results[policy] = fit_policy(
            policy, X, c, y, m, splits, enriched_samples, adir, args)

    output = {
        "schema_version": 1,
        "purpose": "Gate-B closed-book label-policy sensitivity",
        "policies": {
            "strict": "Normalized gold must occur at the start of the short-answer generation.",
            "conservative": "Normalized gold or an official PopQA object alias must occur at the start.",
            "alias_contains": "Gold or an official PopQA object alias may occur anywhere; used as a format-sensitivity envelope.",
            "permissive": "Legacy bidirectional substring match without aliases; treated only as an optimistic lexical envelope.",
        },
        "reason_counts": reasons,
        "disagreement_count": len(disagreements),
        "results": results,
        "gate_rule": (
            "Gate B passes this lexical-policy test only if the joint MLP retains positive "
            "action-loss advantage over both alternatives under every policy and cost matrix."
        ),
    }
    out = Path(args.output) if args.output else adir / "label_sensitivity.json"
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    audit_path = adir / "label_disagreements.jsonl"
    audit_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                  for row in disagreements), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Wrote {out} and {audit_path}")


if __name__ == "__main__":
    main()
