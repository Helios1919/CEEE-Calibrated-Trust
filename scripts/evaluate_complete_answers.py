#!/usr/bin/env python3
"""Generate and evaluate complete closed-book and context-conditioned answers.

Generation is cached separately from estimator features so evaluation policies
can be changed without another model forward pass. By default only the persisted
test split is generated; use --split all for a full artifact.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import config
from data.build import gen_closedbook_batch, load_model
from evaluation import atomic_answer_outcome
from analyze import load_samples


def load_indices(adir, split):
    if split == "all":
        return None
    saved = np.load(adir / "split.npz")
    return saved[split]


def generate_unique(model, tok, prompts, batch_size):
    unique = list(dict.fromkeys(prompts))
    generated = gen_closedbook_batch(model, tok, unique, batch_size)
    lookup = dict(zip(unique, generated))
    return [lookup[prompt] for prompt in prompts]


def generate_answers(model, tok, samples, batch_size):
    pri_prompts = [sample["pri_prompt"] for sample in samples]
    ctx_prompts = [sample["ctx_prompt"] for sample in samples]
    pri_answers = generate_unique(model, tok, pri_prompts, batch_size)
    ctx_answers = generate_unique(model, tok, ctx_prompts, batch_size)
    fuse_prompts = [
        (
            f"Evidence: {sample['context']}\n"
            f"Closed-book candidate: {pri_answer}\n\n"
            "Resolve any conflict and give only the best short answer.\n"
            f"{sample['question']}\nAnswer:"
        )
        for sample, pri_answer in zip(samples, pri_answers)
    ]
    fused_answers = generate_unique(model, tok, fuse_prompts, batch_size)
    return pri_answers, ctx_answers, fused_answers


def evaluate_rows(samples, pri_answers, ctx_answers):
    rows = []
    for sample, pri_answer, ctx_answer in zip(samples, pri_answers, ctx_answers):
        rows.append({
            "id": sample["id"],
            "item_id": sample["item_id"],
            "context_variant": sample["context_variant"],
            "state": sample["state"],
            "c_star": sample["c_star"],
            "m_star_cached": sample["m_star"],
            "closedbook_answer": pri_answer,
            "context_answer": ctx_answer,
            "closedbook": atomic_answer_outcome(pri_answer, sample),
            "context_conditioned": atomic_answer_outcome(ctx_answer, sample),
        })
    return rows


def mean(rows, source, metric):
    return float(np.mean([row[source][metric] for row in rows])) if rows else float("nan")


def summarize(rows, split, protocol):
    by_state = defaultdict(list)
    for row in rows:
        by_state[row["state"]].append(row)

    first_token = {}
    summary = {
        "schema_version": 2,
        "purpose": "Complete-answer and atomic-claim outcome evaluation",
        "split": split,
        "sample_count": len(rows),
        "generation_protocol": protocol,
        "metrics": {
            "closedbook_entity_em": mean(rows, "closedbook", "entity_exact_match"),
            "closedbook_token_f1": mean(rows, "closedbook", "token_f1"),
            "context_entity_em": mean(rows, "context_conditioned", "entity_exact_match"),
            "context_token_f1": mean(rows, "context_conditioned", "token_f1"),
            "context_support_rate": mean(rows, "context_conditioned", "supported_by_context"),
            "fused_entity_em": mean(rows, "fused", "entity_exact_match"),
            "fused_token_f1": mean(rows, "fused", "token_f1"),
            "fused_context_support_rate": mean(rows, "fused", "supported_by_context"),
            "wrong_context_copy_rate": float(np.mean([
                row["context_conditioned"]["copies_wrong_context"]
                for row in rows if not row["c_star"]
            ])),
            "correct_but_context_unsupported_rate": mean(
                rows, "context_conditioned", "correct_but_unsupported"),
        },
        "by_state": {},
        "closedbook_label_audit": {
            "cached_positive_samples": int(sum(row["m_star_cached"] for row in rows)),
            "regenerated_answer_initial_positive_samples": int(sum(
                row["closedbook"]["answer_initial_match"] for row in rows)),
            "transition_counts": dict(Counter(
                f"{row['m_star_cached']}->{row['closedbook']['answer_initial_match']}"
                for row in rows
            )),
        },
        "interpretation": (
            "PopQA contributes one atomic entity-valued claim per sample. Entity exact match "
            "uses gold plus official object aliases; token F1 is descriptive only. Context "
            "support is equality with the entity asserted by the synthetic context and is not "
            "a general natural-language entailment judgment."
        ),
    }
    for state, state_rows in sorted(by_state.items()):
        summary["by_state"][str(state)] = {
            "n": len(state_rows),
            "context_entity_em": mean(
                state_rows, "context_conditioned", "entity_exact_match"),
            "context_support_rate": mean(
                state_rows, "context_conditioned", "supported_by_context"),
            "fused_entity_em": mean(state_rows, "fused", "entity_exact_match"),
            "fused_context_support_rate": mean(
                state_rows, "fused", "supported_by_context"),
            "closedbook_entity_em": mean(
                state_rows, "closedbook", "entity_exact_match"),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="popqa")
    parser.add_argument("--model", default=config.MODEL_NAME)
    parser.add_argument("--split", choices=["tr", "va", "te", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    adir = ROOT / "artifacts" / args.data
    samples = load_samples(adir / "data.jsonl")
    indices = load_indices(adir, args.split)
    if indices is not None:
        samples = [samples[int(i)] for i in indices]

    cache_path = adir / f"complete_answers_{args.split}.jsonl"
    required_fields = {"fused_answer", "fused"}
    use_cache = cache_path.exists() and not args.force_generate
    if use_cache:
        with open(cache_path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if (len(rows) != len(samples)
                or any(not required_fields.issubset(row) for row in rows)):
            use_cache = False
            print(f"Ignoring incomplete cache {cache_path}; fused outcomes are required")
    if use_cache:
        pass
    else:
        model, tok = load_model(args.model)
        pri_answers, ctx_answers, fused_answers = generate_answers(
            model, tok, samples, args.batch_size)
        rows = evaluate_rows(samples, pri_answers, ctx_answers)
        for row, fused_answer, sample in zip(rows, fused_answers, samples):
            row["fused_answer"] = fused_answer
            row["fused"] = atomic_answer_outcome(fused_answer, sample)
        with open(cache_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    protocol = {
        "model": args.model,
        "decoding": "greedy",
        "max_new_tokens": config.MAX_NEW_TOKENS,
        "answer_schema": "one atomic entity-valued claim",
    }
    report = summarize(rows, args.split, protocol)
    output = Path(args.output) if args.output else adir / "complete_answer_evaluation.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {cache_path} and {output}")


if __name__ == "__main__":
    main()
