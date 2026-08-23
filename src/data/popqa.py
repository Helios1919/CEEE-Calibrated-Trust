"""Real PopQA data -> ITEM list (main comparison).

PopQA (https://huggingface.co/datasets/akariasai/PopQA) is a 14k-item, 16-relation
real open-domain fact QA dataset. We only use its (subj, prop, obj, question) and
construct correct/wrong contexts via corpus substitution — the NQ-Swap recipe — so
the distractor is same-relation, same-type, and equally natural as the gold.

Note: this is a "real question distribution + synthetic injected context". To use real
retrieved passages, wire in a retriever (noted in the README as a next step); the
interface only depends on the ITEM schema, so it is a drop-in swap.
"""

import hashlib
import json
import random


def _col(row, *names):
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return None


def make_items(split="test", n=-1, max_per_prop=-1, seed=0):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "PopQA needs the `datasets` package: pip install datasets. Or use --data facts for a smoke test.") from e

    ds = load_dataset("akariasai/PopQA", split=split)

    # Normalization: column names differ slightly across versions; defensive column picking.
    by_rel = {}
    for row in ds:
        # Use the readable (_ln) form as answer/subject to avoid Wikipedia title underscores (e.g. Michael_Jackson)
        subj = _col(row, "subj_ln", "subj", "subject")
        gold = _col(row, "obj_ln", "obj", "answer")
        prop = _col(row, "prop", "prop_ln", "relation")   # prop is the bare relation name (e.g. capital)
        question = _col(row, "question")
        aliases_raw = _col(row, "o_aliases", "aliases")
        if isinstance(aliases_raw, str):
            try:
                aliases = json.loads(aliases_raw)
            except json.JSONDecodeError:
                aliases = [aliases_raw]
        else:
            aliases = list(aliases_raw or [])
        aliases = [str(alias) for alias in aliases if str(alias).strip() and str(alias) != str(gold)]
        if not (subj and gold and prop):
            continue
        prop = str(prop)
        if question is None:
            question = f"What is the {prop} of {subj}?"
        by_rel.setdefault(prop, []).append((str(subj), str(gold), str(question), aliases))

    # Sampling: at most max_per_prop items per relation (-1 = no cap), then shuffle
    random.seed(seed)
    pool = []
    for rel, rows in by_rel.items():
        if max_per_prop and max_per_prop > 0:
            rows = rows[:max_per_prop]
        for subj, gold, q, aliases in rows:
            pool.append({"relation": rel, "subject": subj, "gold": gold,
                         "question": q, "aliases": aliases})
    random.shuffle(pool)

    # PopQA can repeat an exact QA tuple. Remove only exact duplicates: rows with
    # different gold objects remain distinct facts even when relation and subject match.
    deduped = []
    seen = set()
    for it in pool:
        key = (it["relation"], it["subject"], it["question"], it["gold"])
        if key not in seen:
            seen.add(key)
            deduped.append(it)
    pool = deduped
    if n > 0:
        pool = pool[:n]

    # Distractor: randomly drawn from the same relation
    golds_by_rel = {}
    for it in pool:
        golds_by_rel.setdefault(it["relation"], []).append(it["gold"])
    items = []
    for it in pool:
        rel = it["relation"]
        cands = [g for g in golds_by_rel[rel] if g != it["gold"]]
        dist = random.choice(cands) if cands else it["gold"] + " (wrong)"
        identity = "\x1f".join([rel, it["subject"], it["question"], it["gold"]])
        item_id = "popqa-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        items.append({
            "item_id": item_id,
            "relation": rel,
            "subject": it["subject"],
            "gold": it["gold"],
            "aliases": it["aliases"],
            "distractor": dist,
            "question": it["question"],
            "correct_statement": f"The {rel} of {it['subject']} is {it['gold']}.",
            "wrong_statement": f"The {rel} of {it['subject']} is {dist}.",
        })
    return items
