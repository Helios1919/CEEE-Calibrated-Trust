# Credence

> **Current decision: Stage 1 = PIVOT.** The active research program is Stage 2: separating retrieval availability, retained evidence sufficiency, model-conditioned utilization, protocol-distributed closed-book answerability, and downstream action risk.

## Research status

Stage 1 is complete with a **PIVOT** decision. The active research plan is documented in `docs/stage-2-plan.md`, and its executable experiment contracts are in `docs/experiment-registry.md`. The next registered experiment is **S2-E1**, a CPU-compatible observability audit over the existing SciFact artifacts. It must finish before schema v2 is frozen or any Stage-2 model comparison begins.

## Stage-1 question and result

Credence began as a Stage-1 research project about **joint source-state estimation for risk-aware retrieval-augmented generation**. Its central question was deliberately narrow:

> When retrieved evidence and a model's closed-book capability can both fail, does estimating their joint state enable safer decisions than scalar confidence or independently estimated marginals?

The repository does not currently claim to solve general RAG hallucination, real retrieval failure, multi-document conflict, or end-to-end answer correctness. The present PopQA experiment is a controlled falsification baseline with synthetic injected context.

## Scientific contract

For question `q`, evidence `E`, model `M`, and closed-book protocol `pi`, the controlled experiment estimates:

| State | `C(q,E)` | `M_pi(q)` | Meaning |
|---:|---:|---:|---|
| 0 | 0 | 0 | neither source is available |
| 1 | 0 | 1 | memory only |
| 2 | 1 | 0 | context only |
| 3 | 1 | 1 | both sources are available |

Source state, source utilization, answer correctness, and context faithfulness are separate variables. The four-state representation is a hypothesis, not a protected design choice. See `docs/research-contract.md` for the claim boundary and `docs/stage-1-plan.md` for the gates.

## Current controlled baseline

The corrected PopQA artifact contains 14,218 fact items and 28,436 context variants. Every fact has a stable `item_id`; its correct and wrong context variants remain in the same persisted split. Distinct facts sharing a relation and subject are no longer collapsed or grouped together. Samples retain the closed-book generation and decoding protocol for label audit.

On the 5,688-sample test split, the joint MLP obtains macro-F1 0.594, balanced accuracy 0.573, NLL 0.827, and multiclass Brier 0.512. An independent binary-product baseline obtains macro-F1 0.497, balanced accuracy 0.481, NLL 0.897, and Brier 0.547. Subject-group bootstrap favors the joint MLP for all four registered diagnostics, including `neither_available` AUROC.

The stronger result is decision-theoretic but still controlled: under three fixed diagnostic cost matrices, the joint MLP has lower bootstrapped action loss than the independent product, the best scalar four-state baseline, and joint multinomial logistic regression. This is a **provisional Gate-A pass**, not an end-to-end RAG claim.

Gate B's lexical-policy audit changes the number of memory-answerable items from 1,792 under answer-initial exact-gold matching to 2,607 when official PopQA object aliases may occur anywhere. After retraining under four policies, the joint MLP retains lower action loss than the independent product and joint linear model for all three cost matrices; all 24 bootstrap intervals exclude zero. This passes the registered lexical label-sensitivity test, while prompt and decoding stability remain open. See `docs/gate-b-label-audit.md`.

The complete-answer audit replaces first-token correctness proxies with paired complete generations. Context-conditioned entity exact match is about 0.49, while roughly 80% of wrong-context answers copy the injected distractor. A nominally repeated greedy closed-book run also changes item labels, so protocol-level Gate B remains unresolved rather than silently promoted to a pass. See `docs/complete-answer-audit.md`.

The outcome-grounded decision layer now compares context, memory, generated fusion, abstention, and simulated retrieve-again actions. The joint-state MLP retains significant realized-cost advantages over the independent product and direct answer-risk baseline under safety-first and balanced costs; under coverage-first costs, intervals against joint linear and direct risk cross zero. The tested fusion prompt improves context support but not world correctness and is almost never selected by the joint-state policy. See `docs/decision-layer-audit.md`.

Gate C now has a complete fixed-corpus SciFact pilot. The deterministic TF-IDF retriever reaches recall@5 0.8225 and recall@10 0.8831, but extending retrieval from five to ten abstracts reduces context-conditioned verdict accuracy from 0.8023 to 0.7937 and fused accuracy from 0.8009 to 0.7792. On the fixed 139-claim test split, the joint MLP has the lowest realized safety-first cost (0.1709), but its advantage over the strongest direct baseline is not statistically resolved (versus direct answer risk: 0.0173, 95% bootstrap CI [-0.0112, 0.0471]). Joint linear is slightly better in the balanced and coverage-first regimes. **Gate C therefore does not pass the registered criterion.** The result points to a pivot: document retrieval availability must be separated from model-conditioned evidence usability. See `docs/gate-c-real-retrieval.md`.

## Reproduce

Install dependencies and run:

```bash
python -m pip install -r requirements.txt
python scripts/run_experiment.py --data popqa --force-build --force-extract
python scripts/analyze.py --data popqa
python scripts/stage1_falsification.py --data popqa --bootstrap 2000
```

The run writes under `artifacts/popqa/`:

```text
data.jsonl                  labeled samples and closed-book generations
features.npz                17-dimensional observation vectors and labels
topk_logits.pkl             legacy first-token diagnostic cache
split.npz                   persisted train/validation/test indices and item groups
estimator.pt                trained estimator, normalization, and temperature
analysis.txt                descriptive controlled analysis
stage1_falsification.json   machine-readable model and action-risk comparisons
label_sensitivity.json      Gate-B results across four answer-label policies
label_disagreements.jsonl   item-level label-policy disagreements
complete_answers_all.jsonl  complete memory, context, and fusion generations and outcomes
complete_answer_evaluation.json
                            complete-answer correctness and context-support audit
decision_evaluation.json    outcome-grounded five-action policy evaluation
```

The real-retrieval pilot additionally writes under `artifacts/scifact/`:

```text
raw/data/                         fixed SciFact corpus, claims, labels, rationales
retrieval_pilot.jsonl             top-five ranked abstracts with provenance
retrieval_pilot_top10.jsonl       empirical deeper-retrieval counterpart
complete_verdicts.jsonl           top-five closed-book, context, and fusion outcomes
complete_verdicts_top10.jsonl     top-ten verdict outcomes for retrieve-again
features.npz                      693 x 17 real-retrieval observation bank
split.npz                         fixed claim-level train/validation/test split
estimator.pt                      calibrated SciFact joint-state estimator
gate_c_decision_evaluation.json   realized five-action costs and bootstrap intervals
```

`run_experiment.py` requires a CUDA-capable environment for the current 7B-model configuration. The small built-in `facts` dataset is available only as a pipeline smoke test.

## Repository structure

```text
src/data/                   controlled ITEM loaders and auditable label construction
src/features/extract.py     current 17-dimensional observation bank
src/estimator.py            four-way MLP and temperature scaling
scripts/run_experiment.py   build, extract, persist split, and train
scripts/analyze.py          descriptive controlled diagnostics
scripts/stage1_falsification.py
                            joint, independent, scalar, and action-risk comparisons
scripts/label_sensitivity.py
                            Gate-B retraining across answer-label policies
scripts/evaluate_complete_answers.py
                            memory, context, and fusion answer outcomes
scripts/evaluate_decisions.py
                            outcome-grounded five-action policy evaluation
scripts/build_scifact_pilot.py
                            fixed-corpus real retrieval with evidence provenance
scripts/evaluate_scifact_answers.py
                            persisted complete SciFact verdict outcomes
scripts/build_scifact_observations.py
                            claim-level splits and observation extraction
scripts/evaluate_scifact_decisions.py
                            empirical retrieve-again and Gate-C decision comparison
docs/research-contract.md   source of truth for variables and claim scope
docs/stage-1-plan.md        completed falsification gates and PIVOT decision
docs/stage-2-plan.md        active hierarchy, hypotheses, work packages, and stopping rules
docs/experiment-registry.md executable Stage-2 experiment contracts and dependencies
docs/gate-b-label-audit.md  label policies, sensitivity result, and qualification
docs/complete-answer-audit.md
                            full-answer correctness and context-support result
docs/decision-layer-audit.md
                            realized-cost action policy result and qualification
docs/gate-c-real-retrieval.md
                            fixed-corpus real-retrieval protocol and current boundary
docs/signal-bank.md         legacy observation-bank implementation reference
```

## Current limitations

SciFact currently defines `C=1` as retrieval of a matching annotated evidence document. The state-2 subset exposes why this is not evidence usability: when memory is wrong but a gold document is retrieved, top-five context accuracy is only 0.50. Closed-book answerability is still one protocol-specific inference event, and protocol stability remains unresolved. The test split has only 139 claims and eight state-0 examples, so decision intervals are wide. Action costs remain preregistered diagnostics rather than deployment estimates.

The current Stage-1 verdict is **PIVOT**. Gate A passes in the controlled construction and lexical Gate B passes, but protocol-level Gate B remains unresolved and Gate C fails its registered realized-cost criterion. The next thesis should model a hierarchy—retrieval availability, evidence sufficiency, model-conditioned utilization, and answer risk—rather than defend document retrieval availability as a binary context-usability state.