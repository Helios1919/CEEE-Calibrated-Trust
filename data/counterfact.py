"""CounterFact 数据 → ITEM 列表（知识冲突专用：真实事实 vs 反事实）。

CounterFact（NeelNanda/counterfact-tracing，21,919 条、34 类关系）为每个
(subject, relation) 给出 target_true（真实答案）与 target_false（反事实答案）。
这里：
  - 正确上下文 = 真实事实（含 target_true）
  - 错误上下文 = 反事实事实（含 target_false）

与 PopQA 的"语料替换"互为补充：CounterFact 的反事实干扰项由人工构造、语义同类型且
高度自然，是检验"模型是否被反事实上下文带偏"（抵抗/纠正）的理想来源。
"""

import random


def make_items(n=-1, seed=0):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "CounterFact 需要 `datasets` 包：pip install datasets。或改用 --data facts 冒烟。") from e

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
