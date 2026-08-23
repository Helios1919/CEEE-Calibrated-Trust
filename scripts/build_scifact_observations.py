#!/usr/bin/env python3
"""Build SciFact four-state samples and extract the Gate-C observation bank."""

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import config
from data.build import load_model
from features.extract import FEATURE_NAMES, Forwarder, extract_all


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_samples(retrieval_rows, verdict_rows):
    retrieval = {row["id"]: row for row in retrieval_rows}
    samples = []
    for verdict in verdict_rows:
        source = retrieval[verdict["id"]]
        evidence_prompt = verdict["prompts"]["context_conditioned"]
        marker = "Evidence:\n"
        end_marker = "\n\nClaim:"
        context = evidence_prompt.split(marker, 1)[1].rsplit(end_marker, 1)[0]
        samples.append({
            "id": verdict["id"],
            "item_id": verdict["id"],
            "claim_id": verdict["claim_id"],
            "source_split": verdict["source_split"],
            "question": verdict["claim"],
            "claim": verdict["claim"],
            "gold": verdict["gold_verdict"],
            "context": context,
            "pri_prompt": verdict["prompts"]["closedbook"],
            "ctx_prompt": evidence_prompt,
            "c_star": verdict["c_star"],
            "m_star": verdict["m_star"],
            "state": verdict["state"],
            "evidence_condition": verdict["evidence_condition"],
            "first_gold_rank": verdict["first_gold_rank"],
            "passages": source["passages"],
            "closedbook_answer": verdict["closedbook_answer"],
            "context_answer": verdict["context_answer"],
            "fused_answer": verdict["fused_answer"],
            "closedbook": verdict["closedbook"],
            "context_conditioned": verdict["context_conditioned"],
            "fused": verdict["fused"],
        })
    return samples


def stratified_split(y, seed):
    indices = np.arange(len(y))
    train, rest = train_test_split(
        indices,
        test_size=1.0 - config.TRAIN_FRAC,
        random_state=seed,
        stratify=y,
    )
    valid, test = train_test_split(
        rest,
        test_size=config.TEST_FRAC / (config.VAL_FRAC + config.TEST_FRAC),
        random_state=seed,
        stratify=y[rest],
    )
    return {"tr": train, "va": valid, "te": test}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", default="artifacts/scifact/retrieval_pilot.jsonl")
    parser.add_argument("--verdicts", default="artifacts/scifact/complete_verdicts.jsonl")
    parser.add_argument("--output-dir", default="artifacts/scifact")
    parser.add_argument("--model", default=config.MODEL_NAME)
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_rows = load_jsonl(ROOT / args.retrieval)
    verdict_rows = load_jsonl(ROOT / args.verdicts)
    samples = build_samples(retrieval_rows, verdict_rows)
    if len(samples) != len(retrieval_rows):
        raise ValueError("Retrieval and verdict artifacts have different claim counts")

    data_path = output_dir / "data.jsonl"
    feature_path = output_dir / "features.npz"
    topk_path = output_dir / "topk_logits.pkl"
    split_path = output_dir / "split.npz"
    write_jsonl(data_path, samples)

    y = np.asarray([row["state"] for row in samples], dtype=int)
    c_star = np.asarray([row["c_star"] for row in samples], dtype=int)
    m_star = np.asarray([row["m_star"] for row in samples], dtype=int)
    splits = stratified_split(y, config.SEED)
    group_id = np.arange(len(samples), dtype=int)
    np.savez(split_path, **splits, group_id=group_id)

    if args.force_extract or not (feature_path.exists() and topk_path.exists()):
        model, tokenizer = load_model(args.model)
        forwarder = Forwarder(model, tokenizer)
        X, metas, topks = extract_all(forwarder, samples, top_k=config.TOP_K)
        forwarder.clear()
        np.savez(
            feature_path,
            X=X,
            y=y,
            c_star=c_star,
            m_star=m_star,
            claim_id=np.asarray([row["claim_id"] for row in samples]),
        )
        with topk_path.open("wb") as handle:
            pickle.dump(topks, handle)
        del model, tokenizer
        torch.cuda.empty_cache()
    else:
        X = np.load(feature_path)["X"]

    summary = {
        "schema_version": 1,
        "claim_count": len(samples),
        "feature_count": len(FEATURE_NAMES),
        "feature_shape": list(X.shape),
        "state_counts": {str(state): int((y == state).sum()) for state in range(4)},
        "split_counts": {name: len(indices) for name, indices in splits.items()},
        "split_unit": "claim",
        "model": args.model,
        "artifacts": {
            "data": str(data_path),
            "features": str(feature_path),
            "topk_logits": str(topk_path),
            "split": str(split_path),
        },
    }
    summary_path = output_dir / "observation_bank.summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
