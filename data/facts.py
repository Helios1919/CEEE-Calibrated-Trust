"""Built-in fact pool (offline smoke / sanity). Produces the ITEM list handed to
data/build.py for four-state construction.

Unified ITEM schema (any dataset only needs to produce this to plug in):
    {
      "relation": str,             # grouping key: distractors are drawn from other golds of the same relation
      "subject":  str,
      "gold":     str,             # the correct answer
      "distractor": str,           # a same-type wrong answer (pre-drawn)
      "question": str,
      "correct_statement": str,    # correct context statement containing gold
      "wrong_statement":   str,    # wrong context statement containing distractor
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

# Mix in rarer facts per category so the 7B model errs -> correction / double_wrong non-empty.
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
    """Expand FACTS into an ITEM list; distractors are randomly drawn from the same relation (corpus substitution)."""
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
