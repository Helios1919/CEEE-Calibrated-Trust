#!/usr/bin/env python3
"""Generate complete SciFact verdicts for the real-retrieval Gate-C pilot."""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

import config
from data.build import gen_closedbook_batch, load_model


VERDICTS = {"SUPPORTED", "CONTRADICTED"}
_VERDICT = re.compile(r"\b(SUPPORTED|CONTRADICTED)\b", re.IGNORECASE)


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gold_verdict(row):
    labels = {
        label
        for doc_labels in row["gold_evidence"].values()
        for label in doc_labels
    }
    if labels == {"supports"}:
        return "SUPPORTED"
    if labels == {"contradicts"}:
        return "CONTRADICTED"
    raise ValueError(f"Claim {row['id']} has unsupported label set: {sorted(labels)}")


def normalize_verdict(answer):
    match = _VERDICT.search(answer or "")
    return match.group(1).upper() if match else None


def truncate_words(text, max_words):
    words = text.split()
    return " ".join(words[:max_words])


def format_evidence(passages, max_passage_words):
    blocks = []
    for passage in passages:
        text = truncate_words(passage["text"].strip(), max_passage_words)
        blocks.append(
            f"[{passage['rank']}] {passage['title']}\n{text}"
        )
    return "\n\n".join(blocks)


def closedbook_prompt(row):
    return (
        "Classify the scientific claim as SUPPORTED or CONTRADICTED. "
        "Return exactly one label and no explanation.\n\n"
        f"Claim: {row['claim']}\nVerdict:"
    )


def context_prompt(row, max_passage_words):
    return (
        "Using only the scientific abstracts below, classify the claim as "
        "SUPPORTED or CONTRADICTED. Return exactly one label and no explanation.\n\n"
        f"Evidence:\n{format_evidence(row['passages'], max_passage_words)}\n\n"
        f"Claim: {row['claim']}\nVerdict:"
    )


def fusion_prompt(row, closedbook_answer, max_passage_words):
    return (
        "Resolve any conflict between the closed-book candidate and the scientific "
        "abstracts. Classify the claim as SUPPORTED or CONTRADICTED. Return exactly "
        "one label and no explanation.\n\n"
        f"Closed-book candidate: {closedbook_answer.strip()}\n\n"
        f"Evidence:\n{format_evidence(row['passages'], max_passage_words)}\n\n"
        f"Claim: {row['claim']}\nVerdict:"
    )


def generate_unique(model, tokenizer, prompts, batch_size):
    unique = list(dict.fromkeys(prompts))
    generated = gen_closedbook_batch(model, tokenizer, unique, batch_size)
    lookup = dict(zip(unique, generated))
    return [lookup[prompt] for prompt in prompts]


def evaluate(answer, gold):
    verdict = normalize_verdict(answer)
    return {
        "verdict": verdict,
        "valid": int(verdict in VERDICTS),
        "correct": int(verdict == gold),
    }


def build_rows(source, model, tokenizer, batch_size, max_passage_words):
    prior_prompts = [closedbook_prompt(row) for row in source]
    evidence_prompts = [context_prompt(row, max_passage_words) for row in source]
    prior_answers = generate_unique(model, tokenizer, prior_prompts, batch_size)
    evidence_answers = generate_unique(model, tokenizer, evidence_prompts, batch_size)
    fusion_prompts = [
        fusion_prompt(row, answer, max_passage_words)
        for row, answer in zip(source, prior_answers)
    ]
    fusion_answers = generate_unique(model, tokenizer, fusion_prompts, batch_size)

    rows = []
    for row, prior_prompt, evidence_prompt, fusion_p, prior, evidence, fused in zip(
            source, prior_prompts, evidence_prompts, fusion_prompts,
            prior_answers, evidence_answers, fusion_answers):
        gold = gold_verdict(row)
        # Retrieval usability is defined by the matching annotated evidence label.
        expected = "supports" if gold == "SUPPORTED" else "contradicts"
        c_star = int(any(
            expected in passage["gold_evidence_labels"]
            for passage in row["passages"]
        ))
        prior_outcome = evaluate(prior, gold)
        m_star = prior_outcome["correct"]
        rows.append({
            "id": row["id"],
            "claim_id": row["claim_id"],
            "source_split": row["source_split"],
            "claim": row["claim"],
            "gold_verdict": gold,
            "c_star": c_star,
            "m_star": m_star,
            "state": 2 * c_star + m_star,
            "evidence_condition": (
                "gold_evidence_retrieved" if c_star else "unresolved_retrieval"
            ),
            "first_gold_rank": row["first_gold_rank"],
            "passage_ids": [passage["corpus_id"] for passage in row["passages"]],
            "prompts": {
                "closedbook": prior_prompt,
                "context_conditioned": evidence_prompt,
                "fused": fusion_p,
            },
            "closedbook_answer": prior,
            "context_answer": evidence,
            "fused_answer": fused,
            "closedbook": prior_outcome,
            "context_conditioned": evaluate(evidence, gold),
            "fused": evaluate(fused, gold),
        })
    return rows


def mean(rows, source, metric):
    return float(np.mean([row[source][metric] for row in rows])) if rows else None


def summarize(rows, model, top_k, max_new_tokens, max_passage_words):
    by_state = defaultdict(list)
    for row in rows:
        by_state[row["state"]].append(row)
    return {
        "schema_version": 1,
        "purpose": "Complete-verdict outcomes for the SciFact Gate-C pilot",
        "claim_count": len(rows),
        "generation_protocol": {
            "model": model,
            "decoding": "greedy",
            "max_new_tokens": max_new_tokens,
            "labels": sorted(VERDICTS),
            "retrieval_top_k": top_k,
            "max_passage_words": max_passage_words,
        },
        "state_encoding": "state = 2 * gold_evidence_retrieved + closedbook_correct",
        "state_counts": dict(Counter(str(row["state"]) for row in rows)),
        "metrics": {
            "closedbook_accuracy": mean(rows, "closedbook", "correct"),
            "context_accuracy": mean(rows, "context_conditioned", "correct"),
            "fused_accuracy": mean(rows, "fused", "correct"),
            "closedbook_valid_rate": mean(rows, "closedbook", "valid"),
            "context_valid_rate": mean(rows, "context_conditioned", "valid"),
            "fused_valid_rate": mean(rows, "fused", "valid"),
        },
        "by_state": {
            str(state): {
                "n": len(state_rows),
                "closedbook_accuracy": mean(state_rows, "closedbook", "correct"),
                "context_accuracy": mean(state_rows, "context_conditioned", "correct"),
                "fused_accuracy": mean(state_rows, "fused", "correct"),
            }
            for state, state_rows in sorted(by_state.items())
        },
        "label_scope": (
            "C=1 only when the retrieved set contains the claim's annotated gold-evidence "
            "document. C=0 means unresolved retrieval, not that every retrieved abstract is "
            "irrelevant or neutral."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/scifact/retrieval_pilot.jsonl")
    parser.add_argument("--output", default="artifacts/scifact/complete_verdicts.jsonl")
    parser.add_argument("--model", default=config.MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--max-passage-words", type=int, default=160)
    parser.add_argument("--force-generate", action="store_true")
    args = parser.parse_args()

    source = load_jsonl(ROOT / args.input)
    output = ROOT / args.output
    use_cache = output.exists() and not args.force_generate
    if use_cache:
        rows = load_jsonl(output)
        required = {"closedbook", "context_conditioned", "fused", "prompts"}
        if len(rows) != len(source) or any(not required.issubset(row) for row in rows):
            use_cache = False
            print(f"Ignoring incomplete cache {output}")

    old_max_new_tokens = config.MAX_NEW_TOKENS
    config.MAX_NEW_TOKENS = args.max_new_tokens
    try:
        if not use_cache:
            model, tokenizer = load_model(args.model)
            rows = build_rows(
                source, model, tokenizer, args.batch_size, args.max_passage_words
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        config.MAX_NEW_TOKENS = old_max_new_tokens

    top_k = source[0]["retrieval"]["top_k"] if source else None
    report = summarize(
        rows, args.model, top_k, args.max_new_tokens, args.max_passage_words
    )
    report_path = output.with_suffix(".summary.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {output} and {report_path}")


if __name__ == "__main__":
    main()
