"""PopQA 真实数据 → ITEM 列表（正式对比用）。

PopQA（https://huggingface.co/datasets/akariasai/PopQA）是 14k 条、16 类关系的
真实开放域事实问答。这里只用它的 (subj, prop, obj, question)，用"语料替换"构造
正确/错误上下文——即 NQ-Swap 的配方，因此干扰项与正确答案同关系、同类型、同样自然。

注意：这是"真实问题分布 + 合成注入上下文"。若要用"真实检索段落"，需再接一个
retriever（README 已标注为下一步），接口只依赖 ITEM schema，替换即可。
"""

import random


def _col(row, *names):
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return None


def make_items(split="test", n=-1, max_per_prop=150, seed=0):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "PopQA 需要 `datasets` 包：pip install datasets。或改用 --data facts 冒烟。") from e

    ds = load_dataset("akariasai/PopQA", split=split)

    # 归一化：不同版本列名略有差异，这里防御式取列。
    by_rel = {}
    for row in ds:
        # 用可读形式（_ln）作答案/主体，避免 Wikipedia 标题的下划线（如 Michael_Jackson）
        subj = _col(row, "subj_ln", "subj", "subject")
        gold = _col(row, "obj_ln", "obj", "answer")
        prop = _col(row, "prop", "prop_ln", "relation")   # prop 为裸关系名（如 capital）
        question = _col(row, "question")
        if not (subj and gold and prop):
            continue
        prop = str(prop)
        if question is None:
            question = f"What is the {prop} of {subj}?"
        by_rel.setdefault(prop, []).append((str(subj), str(gold), str(question)))

    # 抽样：每关系最多 max_per_prop 条，打散
    random.seed(seed)
    pool = []
    for rel, rows in by_rel.items():
        rows = rows[:max_per_prop]
        for subj, gold, q in rows:
            pool.append({"relation": rel, "subject": subj, "gold": gold, "question": q})
    random.shuffle(pool)
    if n > 0:
        pool = pool[:n]

    # 干扰项：同关系随机抽
    golds_by_rel = {}
    for it in pool:
        golds_by_rel.setdefault(it["relation"], []).append(it["gold"])
    items = []
    for it in pool:
        rel = it["relation"]
        cands = [g for g in golds_by_rel[rel] if g != it["gold"]]
        dist = random.choice(cands) if cands else it["gold"] + " (wrong)"
        items.append({
            "relation": rel,
            "subject": it["subject"],
            "gold": it["gold"],
            "distractor": dist,
            "question": it["question"],
            "correct_statement": f"The {rel} of {it['subject']} is {it['gold']}.",
            "wrong_statement": f"The {rel} of {it['subject']} is {dist}.",
        })
    return items
