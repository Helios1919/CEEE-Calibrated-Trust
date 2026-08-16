"""CounterFact data -> ITEM list (dedicated to knowledge conflict: true fact vs counterfact).

CounterFact (NeelNanda/counterfact-tracing, 21,919 items, 34 relations) gives, for
each (subject, relation), a target_true (real answer) and a target_false (counterfactual
answer). Here:
  - correct context = the real fact (contains target_true)
  - wrong context   = the counterfactual fact (contains target_false)

Complementary to PopQA's corpus substitution: CounterFact's counterfactual distractors
are human-authored, semantically same-type, and highly natural — an ideal source for
testing whether the model gets misled by counterfactual context (resistance / correction).
"""

import random


def make_items(n=-1, seed=0):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "CounterFact needs the `datasets` package: pip install datasets. Or use --data facts for a smoke test.") from e

    ds = load_dataset("NeelNanda/counterfact-tracing", split="train",
                      trust_remote_code=True)

    random.seed(seed)
    rows = list(ds)
    random.shuffle(rows)
    if n > 0:
        rows = rows[:n]

    items = []
    for r in rows:
        subject = (r["subject"] or "").strip()
        gold = (r["target_true"] or "").strip()
        dist = (r["target_false"] or "").strip()
        prompt = (r["prompt"] or "").strip()
        rel = (r.get("relation_id") or r.get("relation") or "fact").strip()
        if not (subject and gold and dist and prompt):
            continue
        items.append({
            "relation": rel,
            "subject": subject,
            "gold": gold,
            "distractor": dist,
            "question": prompt,
            "correct_statement": f"{prompt} {gold}",
            "wrong_statement": f"{prompt} {dist}",
        })
    return items
