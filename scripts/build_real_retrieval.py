#!/usr/bin/env python3
"""Build an auditable real-Wikipedia retrieval pilot for persisted PopQA items."""

import argparse
import json
import sys
import urllib.error
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from analyze import load_samples
from data.real_retrieval import fetch_wikipedia_extract, passage_chunks, write_jsonl
from evaluation import answer_forms, normalize_answer


def unique_items(samples):
    items = {}
    for sample in samples:
        items.setdefault(sample["item_id"], sample)
    return list(items.values())


def retrieve_passages(question, passages, top_k):
    if not passages:
        return []
    corpus = [question] + [passage["text"] for passage in passages]
    matrix = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(corpus)
    scores = (matrix[1:] @ matrix[0].T).toarray().ravel()
    order = np.argsort(-scores)[:top_k]
    return [{**passages[int(i)], "rank": rank, "score": float(scores[int(i)])}
            for rank, i in enumerate(order, 1)]


def contains_form(text, forms):
    text = normalize_answer(text)
    return int(any(form in text for form in forms))


def label_passage(passage, sample):
    gold_forms = answer_forms(sample["gold"], sample.get("aliases", []))
    distractor_forms = answer_forms(sample["distractor"])
    supports_gold = contains_form(passage["text"], gold_forms)
    supports_distractor = contains_form(passage["text"], distractor_forms)
    return {
        **passage,
        "mentions_subject": contains_form(passage["text"], answer_forms(sample["subject"])),
        "supports_gold_lexically": supports_gold,
        "supports_distractor_lexically": supports_distractor,
        "conflict_label": (
            "both" if supports_gold and supports_distractor else
            "supports_gold" if supports_gold else
            "supports_distractor" if supports_distractor else
            "neither"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="popqa")
    parser.add_argument("--split", choices=["va", "te"], default="te")
    parser.add_argument("--items", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=120)
    parser.add_argument("--output")
    args = parser.parse_args()

    adir = ROOT / "artifacts" / args.data
    samples = load_samples(adir / "data.jsonl")
    saved = np.load(adir / "split.npz")
    selected = unique_items([samples[int(i)] for i in saved[args.split]])
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(selected))[:min(args.items, len(selected))]

    rows = []
    for index in order:
        sample = selected[int(index)]
        try:
            page = fetch_wikipedia_extract(sample["subject"])
        except (OSError, urllib.error.URLError) as exc:
            rows.append({
                "item_id": sample["item_id"],
                "question": sample["question"],
                "subject": sample["subject"],
                "gold": sample["gold"],
                "aliases": sample.get("aliases", []),
                "status": "fetch_error",
                "error_type": type(exc).__name__,
                "passages": [],
            })
            continue
        if page is None:
            rows.append({
                "item_id": sample["item_id"],
                "question": sample["question"],
                "subject": sample["subject"],
                "gold": sample["gold"],
                "aliases": sample.get("aliases", []),
                "status": "page_missing",
                "passages": [],
            })
            continue
        ranked = retrieve_passages(
            sample["question"], passage_chunks(page, args.max_words), args.top_k)
        labeled = [label_passage(passage, sample) for passage in ranked]
        rows.append({
            "item_id": sample["item_id"],
            "question": sample["question"],
            "subject": sample["subject"],
            "relation": sample["relation"],
            "gold": sample["gold"],
            "aliases": sample.get("aliases", []),
            "distractor": sample["distractor"],
            "status": "ok",
            "retriever": {
                "corpus": "live English Wikipedia subject pages",
                "ranking": "per-page TF-IDF unigram-bigram cosine",
                "top_k": args.top_k,
                "max_passage_words": args.max_words,
            },
            "passages": labeled,
            "top_k_supports_gold": int(any(
                passage["supports_gold_lexically"] for passage in labeled)),
            "top_1_supports_gold": int(bool(labeled) and labeled[0]["supports_gold_lexically"]),
            "top_k_conflict": int(any(
                passage["supports_distractor_lexically"] for passage in labeled)),
        })

    output = (Path(args.output) if args.output else
              adir / f"real_retrieval_{args.split}_{len(rows)}.jsonl")
    write_jsonl(output, rows)
    ok = [row for row in rows if row["status"] == "ok"]
    summary = {
        "schema_version": 1,
        "purpose": "Auditable real-Wikipedia retrieval pilot",
        "split": args.split,
        "requested_items": args.items,
        "sample_count": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "top_1_gold_support_rate": float(np.mean([
            row["top_1_supports_gold"] for row in ok])) if ok else None,
        "top_k_gold_support_rate": float(np.mean([
            row["top_k_supports_gold"] for row in ok])) if ok else None,
        "top_k_conflict_rate": float(np.mean([
            row["top_k_conflict"] for row in ok])) if ok else None,
        "label_scope": (
            "Gold and distractor support are lexical entity-presence labels. "
            "Relevance and sufficiency require manual or semantic adjudication."
        ),
        "output": str(output),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
