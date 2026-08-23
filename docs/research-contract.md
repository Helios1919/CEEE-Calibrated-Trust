# Credence Research Contract

**Status:** Stage 1 — thesis discovery and falsification

This document is the scientific contract for the project. It is deliberately stricter than the implementation: when an experiment contradicts a claim here, the claim must be revised.

## 1. Central question

Credence studies whether a RAG system should estimate a **joint epistemic state** of external evidence and closed-book answerability before selecting an action.

The central question is:

> When retrieved evidence and the model's closed-book capability can both be unreliable, does a calibrated joint state posterior support safer and more cost-efficient actions than a single confidence or conflict score?

The project is **not** primarily a paper about adding more internal signals to an MLP. Internal signals are observations. The scientific object is the state-to-action decision problem.

## 2. Variables and evaluation layers

For a question `q`, retrieved evidence `E`, model `M`, and a fixed inference protocol `pi`:

- `C(q, E)`: context usability. The evidence is relevant and sufficiently correct to support a correct answer under the task's evidence policy.
- `M_pi(q)`: closed-book answerability. The model produces a correct answer under the specified closed-book prompt and decoding protocol.
- `Z = (C, M_pi)`: source-state label.
- `Y`: complete generated-answer outcome, including world correctness and evidence support.
- `U`: source utilization, i.e. which evidence or internal knowledge the model actually used.
- `A`: action selected by the system.

These are distinct:

```text
source state != source utilization != answer correctness != context faithfulness
```

A source can be correct while ignored. A source can be wrong while faithfully reproduced. A correct answer can be unsupported by the retrieved evidence. A single first-token match is not a complete-answer outcome.

## 3. Minimal source-state representation

The current controlled setting uses four states:

| ID | Legacy name | Contract name | C | M_pi | Operational meaning |
|---:|---|---|---:|---:|---|
| 0 | `double_wrong` | `neither_available` | 0 | 0 | Neither source is sufficient for a correct answer |
| 1 | `resistance` | `memory_only` | 0 | 1 | Closed-book capability is available; context is not |
| 2 | `correction` | `context_only` | 1 | 0 | Evidence is available; closed-book capability is not |
| 3 | `agreement` | `both_available` | 1 | 1 | Both sources are available |

The legacy IDs remain stable for artifact compatibility during Stage 1. New public documentation should use the contract names and show the legacy aliases when needed.

The four-state representation is a **hypothesis**, not an axiom. It must beat at least the following alternatives on downstream decision risk:

1. one scalar conflict or confidence score;
2. two independently estimated marginal probabilities, combined as `P(C)P(M)`;
3. a direct answer-correctness or abstention score;
4. a retrieval-quality evaluator.

If it does not, the project will pivot to a richer or simpler representation rather than preserving four states for historical reasons.

## 4. Label semantics and current limitation

In the current synthetic PopQA pipeline, `C` is assigned by construction because correct and distractor contexts are deliberately generated. `M_pi` is assigned by one greedy closed-book generation. Therefore current labels mean:

- controlled context correctness under the dataset's fact policy; and
- correctness of one closed-book inference event.

They do **not** yet establish latent real-world context reliability or stable parametric memory.

The real-retrieval phase must replace this with evidence-level or claim-level annotation, and the closed-book phase must measure protocol stability using paraphrases or repeated semantic generations.

## 5. Action space

The intended actions are:

- `use_context`: answer with the retrieved evidence as the primary source;
- `use_memory`: answer closed-book or suppress misleading evidence;
- `fuse`: combine sources under an explicit evidence policy;
- `abstain`: decline to provide a substantive answer;
- `retrieve_again`: obtain new or broader evidence;
- `escalate`: defer to a stronger model or human process.

The policy should minimize expected loss:

$$
a^*(x) = argmin_a sum_z P_theta(z | x) L(a, z)$$

where `x` is the observation vector and `L` includes answer error, unsupported claims, abstention, retrieval, latency, and token cost. Classification accuracy is a diagnostic, not the final objective.

## 6. What can be claimed now

The current artifact can support only these narrow claims:

- The current 17-dimensional observation vector contains signal about the controlled four-state labels.
- A calibrated MLP can produce a probability distribution over those labels on the current leakage-controlled PopQA split.
- The current estimator is substantially better at the closed-book label than the context label.
- Selective prediction can trade coverage for accuracy under the current greedy-context answer proxy.

It cannot yet support claims that Credence:

- detects full RAG hallucinations;
- estimates stable parametric knowledge truth;
- solves real retrieval failure;
- improves end-to-end answer quality;
- provides a universal upper bound for weighted decoding;
- generalizes across models or domains;
- solves multi-document conflict.

## 7. Falsification gates

Stage 1 passes only if the following tests are specified and later run:

- Joint posterior versus independent marginal product;
- Joint posterior versus scalar confidence/conflict scores;
- Complete-answer and claim-level outcomes;
- Real retrieval with incomplete, noisy, and conflicting evidence;
- Cross-model and cross-domain evaluation;
- Cost-sensitive action evaluation;
- Calibration under distribution shift.

A strong result requires a stable downstream advantage of the joint posterior. If simple baselines explain the advantage, the four-state thesis is rejected or narrowed.

## 8. Upper-bound scope

For fixed candidate answers `r_ctx` and `r_mem`, a hard selector can be bounded by:

$$
Acc(selector) <= P(C=1 or M_pi=1) = 1 - P(C=0, M_pi=0).
$$

This statement requires that the selector can only choose between the two fixed candidates and cannot call a new retriever or synthesize a third answer.

It does **not** automatically apply to token-level distribution mixing, unconstrained generation, prompt steering, or any method that can compose a new answer. Those interfaces require separate empirical or formal analysis.

## 9. Target contribution

The target paper should not claim novelty from a feature bank alone. The intended contribution is a validated decision framework showing when joint source-state information is necessary, how it should be calibrated, and how it changes answer / abstain / retrieve-again decisions under explicit costs.
