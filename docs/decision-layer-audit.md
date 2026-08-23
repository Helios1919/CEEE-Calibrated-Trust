# Outcome-grounded decision-layer audit

## Purpose

This audit evaluates actions using realized complete-answer outcomes rather than treating source-state labels as answer correctness. The available actions are `use_context`, `use_memory`, `fuse`, `abstain`, and `retrieve_again`. The first three use cached greedy generations; `retrieve_again` remains a simulated perfect second retrieval and is therefore a value-of-information diagnostic, not an implemented retriever.

For an answer action, the realized cost is:

$$
L = \lambda_e \mathbf{1}[\text{entity incorrect}] + \lambda_u \mathbf{1}[\text{not supported by context}].
$$

Memory receives only the answer-error term. Abstention and retrieval receive their fixed registered costs. Fusion is scored from its own generated answer and is never assigned a source-state proxy outcome.

## Protocol

The all-split artifact contains 28,436 context variants from 14,218 PopQA facts. For every variant, the evaluator stores closed-book, context-conditioned, and fused complete generations. Entity exact match uses the gold object and official object aliases. Context support means equality with the entity asserted by the synthetic context; it is not general passage entailment.

The policy baselines are a calibrated joint-state MLP, joint multinomial logistic regression, an independent binary-product posterior, a direct answer-risk predictor, and fixed-action policies. Uncertainty uses 2,000 item-group bootstrap draws on the persisted test split.

## Complete-answer outcomes

Across all variants, closed-book entity EM is 0.1193, context-conditioned EM is 0.4917, and fused EM is 0.4709. Context-conditioned support is 0.8613; fused support is 0.8953. Under wrong contexts, the context-conditioned answer copies the distractor at rate 0.7991. Fusion is consequently more context-faithful but not more world-correct in this prompt, and it should not be described as a generally safer answer generator.

The state pattern is informative. In the memory-only state, context-conditioned EM is 0.4018, closed-book EM is 0.8457, and fused EM is 0.2280. In the context-only state, context-conditioned EM is 0.9141 and fused EM is 0.8986. In the both-available state, context-conditioned EM is 0.9815 and fused EM is 0.9755. These results show that the fusion prompt tends to surrender useful memory when the context is misleading, while offering no correctness gain when both sources are already available.

## Decision results

The joint-state MLP has lower realized mean cost than joint linear, the independent product, and the direct answer-risk baseline in all three registered regimes. The bootstrap advantage intervals are:

| Regime | vs joint linear | vs independent product | vs direct answer risk |
|---|---:|---:|---:|
| safety first | 0.00243 [0.00127, 0.00367] | 0.00445 [0.00304, 0.00570] | 0.00176 [0.00004, 0.00349] |
| balanced | 0.00626 [0.00373, 0.00884] | 0.00628 [0.00364, 0.00907] | 0.00382 [0.00111, 0.00666] |
| coverage first | 0.00373 [-0.00230, 0.00964] | 0.00633 [0.00016, 0.01256] | 0.00163 [-0.00454, 0.00805] |

The joint MLP therefore retains a conservative and balanced decision advantage, while superiority over linear and direct outcome-risk modeling is unresolved under coverage-first costs. This is evidence for conditional decision value, not universal dominance.

The learned policies select fusion rarely: zero test examples in safety-first and balanced regimes and 8.3% for the direct-risk baseline in coverage-first. This is a useful negative result. A fused answer is not automatically preferable merely because it consumes both sources. The policy must account for conflict asymmetry and the distinct failure modes of context-conditioned generation.

## Decision consequence

Task 9 is complete as a controlled outcome-grounded action layer, with fusion included and evaluated. Gate A remains a qualified pass: the joint posterior beats the independent product and direct answer-risk baseline for conservative regimes, but the result does not establish universal superiority. The simulated perfect retrieval action continues to dominate many decisions when its fixed cost is low. Gate C must replace that assumption with an empirical retrieval pilot before any end-to-end RAG claim is made.
