# Gate-B label audit

## Purpose

Gate B asks whether the controlled Stage-1 conclusion depends on a brittle definition of closed-book answerability. The audited variable is `M_pi(q)`: whether the model answers a PopQA question correctly under the frozen closed-book prompt and greedy decoding protocol.

This audit does not claim that one lexical policy is a complete semantic evaluator. Instead, it retrains the estimator under four deliberately different policies and asks whether the decision-theoretic conclusion survives the resulting label shifts.

## Policies

The **strict** policy accepts only a normalized gold answer at the start of the short-answer generation. The **official-alias** policy also accepts an official PopQA object alias at the start. The **answer-anywhere** policy accepts a gold answer or official object alias anywhere in the generation; this tests sensitivity to verbose answer formatting but can accept an answer mention inside a bad continuation. The **legacy-permissive** policy restores bidirectional substring matching without aliases and is retained only as an optimistic lexical envelope.

The official aliases come from PopQA's `o_aliases` field. The dataset's `possible_answers` field is intentionally not used: inspection showed that it can contain aliases pooled across several objects associated with an ambiguous subject, which can turn an unrelated answer into a false positive.

## Label movement

Across 14,218 fact items, the strict policy labels 1,792 items answerable. Official aliases at the answer start increase this to 2,260. Allowing an official answer anywhere increases it to 2,607. The legacy bidirectional substring policy labels 2,521 items answerable. In total, 937 items change label under at least one policy.

The spread is scientifically meaningful. It shows that one greedy generation is not a stable measurement of parametric knowledge, and that answer format and alias policy materially affect the estimated state distribution. Gate B therefore cannot be summarized as “the labels are clean.” They are not.

## Sensitivity result

For every policy, the joint MLP was retrained on the same persisted item-level split. It was compared with a joint multinomial logistic model and the product of separately trained binary context and memory models. Item-group bootstrap used 2,000 draws.

The joint MLP retains lower action loss under every combination of the four label policies, three registered diagnostic cost matrices, and both comparison models. All 24 lower bounds of the 95% bootstrap intervals remain above zero. The narrowest interval occurs under the answer-anywhere policy in the error-averse regime against joint linear: joint advantage 0.002740 with 95% CI [0.000738, 0.004783].

Discrimination and probability metrics also retain the same ordering. Joint-MLP macro-F1 ranges from 0.5657 to 0.6158 across policies, while the independent product ranges from 0.4937 to 0.5144. The size of the result changes, but its direction does not.

## Decision

**Gate B passes the lexical-policy sensitivity test.** The provisional Gate-A result is not an artifact of the original exact-gold policy or the legacy prefix behavior.

This is a qualified pass, not proof that `M_pi` measures stable knowledge. The remaining stability question is protocol-level rather than lexical: repeated inference, prompt paraphrase, decoding variation, and longer generation may move labels in ways that cannot be reconstructed from the current cache. That uncertainty must either be measured in an additional protocol audit or represented explicitly as a probabilistic/hierarchical memory state.

Machine-readable results are written to `artifacts/popqa/label_sensitivity.json`; all policy disagreements are in `artifacts/popqa/label_disagreements.jsonl`.
