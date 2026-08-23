#!/usr/bin/env python3
"""Build a reproducible real-retrieval pilot from SciFact claims and abstracts."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_label(label):
    return {"SUPPORT": "supports", "CONTRADICT": "contradicts"}[label]


def evidence_labels(claim):
    labels = {}
    rationales = {}
    for doc_id, groups in claim.get("evidence", {}).items():
        doc_labels = {normalize_label(group["label"]) for group in groups}
        labels[str(doc_id)] = sorted(doc_labels)
        rationales[str(doc_id)] = sorted({
            int(sentence)
            for group in groups
            for sentence in group.get("sentences", [])
        })
    return labels, rationales


def rank_claims(claims, documents, top_k):
    doc_ids = list(documents)
    doc_text = [
        documents[doc_id]["title"] + " " + " ".join(documents[doc_id]["abstract"])
        for doc_id in doc_ids
    ]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=100000)
    doc_matrix = vectorizer.fit_transform(doc_text)
    query_matrix = vectorizer.transform([claim["claim"] for claim in claims])
    scores = query_matrix @ doc_matrix.T

    rows = []
    for query_index, claim in enumerate(claims):
        labels, rationales = evidence_labels(claim)
        row_scores = scores.getrow(query_index).toarray().ravel()
        order = np.argsort(-row_scores)[:top_k]
        passages = []
        for rank, index in enumerate(order, 1):
            doc_id = doc_ids[int(index)]
            document = documents[doc_id]
            passages.append({
                "corpus_id": doc_id,
                "rank": rank,
                "score": float(row_scores[int(index)]),
                "title": document["title"],
                "text": " ".join(document["abstract"]),
                "sentences": document["abstract"],
                "gold_evidence_labels": labels.get(doc_id, []),
                "gold_rationale_sentences": rationales.get(doc_id, []),
            })
        gold_docs = set(labels)
        retrieved = [passage["corpus_id"] for passage in passages]
        first_gold_rank = next(
            (rank for rank, doc_id in enumerate(retrieved, 1) if doc_id in gold_docs), None)
        rows.append({
            "id": f"scifact-{claim['id']}",
            "claim_id": int(claim["id"]),
            "claim": claim["claim"],
            "source_split": claim["source_split"],
            "gold_evidence": labels,
            "gold_rationales": rationales,
            "retrieval": {
                "corpus": "SciFact 5,183-abstract corpus",
                "method": "TF-IDF unigram-bigram cosine",
                "top_k": top_k,
            },
            "passages": passages,
            "first_gold_rank": first_gold_rank,
            "gold_recall_at_k": int(any(doc_id in gold_docs for doc_id in retrieved)),
            "retrieved_support": int(any(
                "supports" in passage["gold_evidence_labels"] for passage in passages)),
            "retrieved_contradiction": int(any(
                "contradicts" in passage["gold_evidence_labels"] for passage in passages)),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="artifacts/scifact/raw/data")
    parser.add_argument("--splits", nargs="+", choices=["train", "dev"], default=["train", "dev"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="artifacts/scifact/retrieval_pilot.jsonl")
    args = parser.parse_args()

    raw = Path(args.raw)
    documents = {
        str(row["doc_id"]): row for row in load_jsonl(raw / "corpus.jsonl")
    }
    claims = []
    for split in args.splits:
        for claim in load_jsonl(raw / f"claims_{split}.jsonl"):
            if claim.get("evidence"):
                claims.append({**claim, "source_split": split})

    rows = rank_claims(claims, documents, args.top_k)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    ranks = [row["first_gold_rank"] for row in rows if row["first_gold_rank"]]
    summary = {
        "schema_version": 1,
        "purpose": "Reproducible real-retrieval pilot with evidence provenance",
        "corpus_documents": len(documents),
        "claim_count": len(rows),
        "source_split_counts": dict(Counter(row["source_split"] for row in rows)),
        "top_k": args.top_k,
        "gold_recall_at_k": float(np.mean([
            row["gold_recall_at_k"] for row in rows])),
        "gold_mean_reciprocal_rank": float(np.mean([
            1.0 / row["first_gold_rank"] if row["first_gold_rank"] else 0.0
            for row in rows
        ])),
        "retrieved_support_rate": float(np.mean([
            row["retrieved_support"] for row in rows])),
        "retrieved_contradiction_rate": float(np.mean([
            row["retrieved_contradiction"] for row in rows])),
        "retrieved_gold_rank_counts": dict(Counter(ranks)),
        "output": str(output),
        "limitations": (
            "TF-IDF is a deterministic sparse baseline. Evidence labels apply to annotated "
            "documents and rationale sentences; unannotated retrieved abstracts are not assumed "
            "irrelevant or neutral."
        ),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
