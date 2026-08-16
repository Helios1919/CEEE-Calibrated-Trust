"""内置事实池（离线冒烟 / sanity）。产出 ITEM 列表交给 data/build.py 建四态。

ITEM 统一 schema（任何数据集只需产出它即可接入）：
    {
      "relation": str,             # 分组键：干扰项从同 relation 的其它 gold 抽
      "subject":  str,
      "gold":     str,             # 正确答案
      "distractor": str,           # 同类型错误答案（已抽好）
      "question": str,
      "correct_statement": str,    # 含 gold 的正确上下文陈述
      "wrong_statement":   str,    # 含 distractor 的错误上下文陈述
    }
"""

import random

TEMPLATES = {
    "capital": {
        "question": "What is the capital of {s}?",
        "statement": "The capital of {s} is {a}.",
    },
    "language": {
        "question": "What is the official language of {s}?",
        "statement": "The official language of {s} is {a}.",
    },
    "currency": {
        "question": "What is the currency of {s}?",
        "statement": "The currency of {s} is the {a}.",
    },
    "author": {
        "question": "Who is the author of {s}?",
        "statement": "The author of {s} is {a}.",
    },
}

# 每类混入冷门事实，保证 7B 会答错 → correction / double_wrong 非空。
FACTS = {
    "capital": [
        ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"),
        ("Italy", "Rome"), ("Spain", "Madrid"), ("United Kingdom", "London"),
        ("Canada", "Ottawa"), ("Australia", "Canberra"), ("China", "Beijing"),
        ("Russia", "Moscow"), ("Brazil", "Brasília"), ("India", "New Delhi"),
        ("Egypt", "Cairo"), ("Greece", "Athens"), ("Portugal", "Lisbon"),
        ("Austria", "Vienna"), ("Bhutan", "Thimphu"), ("Burkina Faso", "Ouagadougou"),
        ("Kyrgyzstan", "Bishkek"), ("Mongolia", "Ulaanbaatar"),
    ],
    "language": [
        ("Brazil", "Portuguese"), ("Japan", "Japanese"), ("Russia", "Russian"),
        ("France", "French"), ("Germany", "German"), ("Spain", "Spanish"),
        ("Portugal", "Portuguese"), ("China", "Mandarin"), ("Netherlands", "Dutch"),
        ("Poland", "Polish"), ("Turkey", "Turkish"), ("South Korea", "Korean"),
        ("Italy", "Italian"), ("Denmark", "Danish"), ("Norway", "Norwegian"),
        ("Sweden", "Swedish"), ("Finland", "Finnish"), ("Hungary", "Hungarian"),
        ("Vietnam", "Vietnamese"), ("Thailand", "Thai"),
    ],
    "currency": [
        ("Japan", "Japanese yen"), ("United Kingdom", "pound sterling"),
        ("United States", "US dollar"), ("Russia", "ruble"), ("China", "yuan"),
        ("India", "rupee"), ("South Korea", "won"), ("Brazil", "real"),
        ("Mexico", "peso"), ("Switzerland", "Swiss franc"), ("Poland", "złoty"),
        ("Turkey", "lira"), ("Canada", "Canadian dollar"),
        ("Australia", "Australian dollar"), ("Denmark", "krone"),
        ("Bhutan", "ngultrum"), ("Mongolia", "tögrög"), ("Nigeria", "naira"),
        ("Kenya", "shilling"), ("Vietnam", "dong"),
    ],
    "author": [
        ("1984", "George Orwell"), ("Pride and Prejudice", "Jane Austen"),
        ("The Great Gatsby", "F. Scott Fitzgerald"), ("Moby-Dick", "Herman Melville"),
        ("War and Peace", "Leo Tolstoy"), ("Crime and Punishment", "Fyodor Dostoevsky"),
        ("One Hundred Years of Solitude", "Gabriel García Márquez"),
        ("The Odyssey", "Homer"), ("Don Quixote", "Miguel de Cervantes"),
        ("The Catcher in the Rye", "J. D. Salinger"),
        ("To Kill a Mockingbird", "Harper Lee"),
        ("The Lord of the Rings", "J. R. R. Tolkien"),
        ("Hamlet", "William Shakespeare"), ("Madame Bovary", "Gustave Flaubert"),
        ("The Divine Comedy", "Dante Alighieri"), ("The Stranger", "Albert Camus"),
        ("Brave New World", "Aldous Huxley"),
        ("The Grapes of Wrath", "John Steinbeck"), ("Lolita", "Vladimir Nabokov"),
        ("The Sound and the Fury", "William Faulkner"),
    ],
}


def make_items(seed=0):
    """把 FACTS 展开成 ITEM 列表，干扰项同 relation 随机抽（语料替换）。"""
    random.seed(seed)
    items = []
    for rel, triples in FACTS.items():
        answers = [a for _, a in triples]
        t = TEMPLATES[rel]
        for s, a in triples:
            dist = a
            while dist == a:
                dist = random.choice(answers)
            items.append({
                "relation": rel,
                "subject": s,
                "gold": a,
                "distractor": dist,
                "question": t["question"].format(s=s),
                "correct_statement": t["statement"].format(s=s, a=a),
                "wrong_statement": t["statement"].format(s=s, a=dist),
            })
    return items
