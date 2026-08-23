# Complete-answer outcome audit

## Purpose

This audit replaces the legacy first-token proxy with complete greedy generations on the persisted PopQA test split. It keeps three objects separate: source availability, the complete answer's world correctness, and whether that answer reproduces the injected context. First-token top-k behavior remains an observation-bank diagnostic; it is no longer treated as answer correctness.

PopQA contributes one atomic entity-valued claim per question. Consequently, correctness is evaluated by normalized equality to the gold object or an official `o_aliases` form. Token F1 is reported descriptively but is not promoted to a correctness label. A general semantic or claim-entailment judge is intentionally not used here: for this atomic schema it would add evaluator variance without adding information. Such a judge becomes necessary in the real-retrieval phase, where answers and passages contain multiple claims.

## Protocol

The audit regenerated both closed-book and context-conditioned answers for all 5,688 persisted test samples with Qwen2.5-7B, greedy decoding, and 12 new tokens. Generation is cached independently from the 17-dimensional feature artifact, so surface-form policies can be rescored without another model run.

The context-support metric has a deliberately narrow controlled meaning: the generated entity equals the entity asserted by the synthetic context. It is not presented as natural-language entailment for arbitrary passages.

## Results

Complete closed-book entity exact match is 0.1224 and token F1 is 0.1778. Context-conditioned entity exact match is 0.4930 and token F1 is 0.5180. The large gap between exact match and first-token-style diagnostics confirms that the old cache cannot serve as an end-to-end answer metric.

Context-conditioned answers reproduce the entity asserted by their context in 0.8643 of samples. Under wrong contexts, the model copies the injected distractor in 0.8038 of samples. This is the sharpest result of the audit: the controlled model is highly context-faithful but often faithful to false evidence. Faithfulness and correctness therefore move in opposite directions on wrong-context variants, precisely the distinction required by the research contract.

The state-conditioned complete-answer results are:

| Source state | Samples | Context answer EM | Context support | Closed-book EM |
|---|---:|---:|---:|---:|
| neither available | 2,444 | 0.0049 | 0.8662 | 0.0008 |
| memory only | 400 | 0.4050 | 0.4225 | 0.8650 |
| context only | 2,444 | 0.9145 | 0.9145 | 0.0008 |
| both available | 400 | 0.9875 | 0.9875 | 0.8650 |

The memory-only row shows that closed-book capability does not imply resistance to misleading evidence: context-conditioned accuracy falls from closed-book EM 0.8650 to 0.4050. Conversely, when both sources are available, context-conditioned EM reaches 0.9875.

## Label reproducibility

The regenerated closed-book outputs expose a protocol defect in the current cached labels. Among 5,688 test samples, 82 cached positives become negative under the regenerated answer-initial policy, while four cached negatives become positive. Because each item has two context variants, these correspond to 41 positive-to-negative and two negative-to-positive fact items. The apparent repeat run used the same nominal greedy protocol, so this movement is not ordinary sampling variance; likely contributors include the earlier label policy, alias reconstruction, model/runtime differences, or nondeterministic numerical execution.

This means the lexical Gate-B pass remains valid as a conditional sensitivity result, but protocol-level Gate B is not passed. `M_pi` should not yet be described as stable parametric knowledge. The next protocol audit must persist model revision and runtime metadata, regenerate repeated closed-book outputs under controlled prompts, and report item-level agreement and transition matrices.

## Decision consequence

The complete-answer layer validates the need to keep state, correctness, and support distinct, but it does not by itself validate the four-state estimator as the best decision representation. The next decision experiment must use realized outcome costs—wrong answers, unsupported claims, abstention, and retrieval—rather than source-state proxy matrices alone. A direct answer-risk predictor must be included as a baseline.

Machine-readable results are in `artifacts/popqa/complete_answer_evaluation.json`; sample-level generations and outcomes are in `artifacts/popqa/complete_answers_te.jsonl`.
