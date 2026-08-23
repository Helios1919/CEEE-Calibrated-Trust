"""Complete-answer evaluation for atomic fact QA.

PopQA answers express one entity-valued claim. Registered object aliases provide
an auditable equivalence set; token overlap is reported separately and is never
used as a correctness label.
"""

import re
from collections import Counter


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize_answer(text):
    """Normalize surface form without deleting entity-bearing words.

    Unlike SQuAD-style scoring, articles are retained: they can be part of a
    proper name (for example, "The Who" or "The Hague").
    """
    text = (text or "").lower().replace("_", " ")
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def answer_forms(gold, aliases=None):
    forms = {normalize_answer(gold)}
    forms.update(normalize_answer(alias) for alias in (aliases or []))
    return {form for form in forms if form}


def entity_exact_match(prediction, gold, aliases=None):
    """Exact normalized match to the gold entity or an official alias."""
    return int(normalize_answer(prediction) in answer_forms(gold, aliases))


def answer_initial_match(prediction, gold, aliases=None):
    """Production source-availability label used by the controlled experiment."""
    prediction = normalize_answer(prediction)
    return int(bool(prediction) and any(
        prediction == form or prediction.startswith(form + " ")
        for form in answer_forms(gold, aliases)
    ))


def token_f1(prediction, gold, aliases=None):
    """Maximum bag-of-token F1 over the registered answer forms."""
    pred = normalize_answer(prediction).split()
    if not pred:
        return 0.0
    best = 0.0
    for form in answer_forms(gold, aliases):
        target = form.split()
        common = sum((Counter(pred) & Counter(target)).values())
        if not common:
            continue
        precision = common / len(pred)
        recall = common / len(target)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def atomic_answer_outcome(prediction, sample):
    """Evaluate the one entity-valued claim represented by a PopQA answer.

    `supported_by_context` means that the answer equals the entity asserted by
    the injected context. It measures faithfulness to this controlled context,
    not entailment in arbitrary retrieved prose.
    """
    aliases = sample.get("aliases", [])
    correct = entity_exact_match(prediction, sample["gold"], aliases)
    asserted = sample["gold"] if sample["c_star"] else sample["distractor"]
    supported = entity_exact_match(
        prediction, asserted, aliases if sample["c_star"] else None)
    return {
        "normalized_answer": normalize_answer(prediction),
        "entity_exact_match": correct,
        "answer_initial_match": answer_initial_match(
            prediction, sample["gold"], aliases),
        "token_f1": token_f1(prediction, sample["gold"], aliases),
        "supported_by_context": supported,
        "copies_wrong_context": int(not sample["c_star"] and supported),
        "correct_but_unsupported": int(correct and not supported),
    }
