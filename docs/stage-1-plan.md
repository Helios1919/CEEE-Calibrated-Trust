# Stage 1: Thesis Discovery and Falsification

## Goal

Stage 1 determines whether Credence has a defensible central thesis before the project invests in a benchmark, policy system, or multi-document pipeline.

The stage is complete when the repository has one scientific contract, one reproducible legacy baseline, explicit falsification experiments, and a go / pivot / stop decision protocol.

## Workstream A — Scientific reset

1. Adopt `docs/research-contract.md` as the source of truth.
2. Separate source state, source utilization, complete-answer correctness, and context faithfulness.
3. Use neutral state names in public materials while preserving legacy integer IDs for artifact compatibility.
4. Restrict the routing upper bound to fixed-candidate hard selection.
5. Treat the current four-state representation as a hypothesis to be challenged.

## Workstream B — Reproducibility baseline

The current baseline is a legacy controlled experiment. It must remain runnable while new work is built beside it.

Required outputs:

- `artifacts/<run>/manifest.json` with commit, model, dataset, split protocol, package versions, configuration, and file hashes;
- machine-readable analysis output;
- one canonical command for analysis;
- a visible distinction between current reproducible metrics and stale historical results.

The existing PopQA artifact is retained because it is useful for falsification experiments. Old downstream-decoder results are not accepted as current evidence.

## Workstream C — First falsification experiment

Using the frozen PopQA artifact and exactly the same grouped split, compare:

1. joint four-way MLP;
2. joint multinomial logistic regression;
3. independent binary logistic models for `C` and `M_pi`;
4. the product distribution from independent marginals;
5. direct binary predictors of context-answer correctness and `neither_available`;
6. scalar confidence and divergence baselines.

Required metrics:

- four-state macro-F1 and balanced accuracy;
- statewise and marginal AUROC/AUPRC;
- NLL and multiclass Brier score;
- adaptive and classwise calibration diagnostics;
- risk–coverage curves for a fixed answer source;
- expected action loss under preregistered cost matrices;
- bootstrap confidence intervals over subject groups.

### Gate A

Proceed with the four-state thesis only if the joint posterior demonstrates a stable benefit that is not explained by one marginal label, independent products, or one scalar score. Classification gains alone are insufficient; at least one downstream action-risk advantage must survive bootstrap uncertainty.

## Workstream D — Label audit

The current closed-book label uses one greedy generation and a bidirectional substring match. Audit:

- empty-string behavior;
- partial-answer false positives;
- aliases and normalization;
- prompt sensitivity;
- truncation sensitivity;
- disagreement between first-token and complete-answer correctness.

### Gate B

If label noise changes conclusions materially, rebuild labels before further method work.

## Workstream E — Real-retrieval pilot design

Specify, but do not yet scale, a small pilot that includes:

- relevant and sufficient evidence;
- relevant but insufficient evidence;
- irrelevant evidence;
- locally false or stale evidence;
- context–memory conflict;
- evidence missing from both sources.

Each example must preserve retrieved documents, ranks, scores, provenance, atomic support labels, complete generations, and chosen actions.

### Gate C

The four-state abstraction must remain useful when context usability is not assigned by construction. Otherwise pivot to a hierarchical evidence-state or claim-graph representation.

## Decision protocol

- **GO:** joint-state information yields robust decision value, label audit is stable, and the real-retrieval pilot preserves the effect.
- **PIVOT:** the problem is important but four states or the current observation bank are inadequate.
- **STOP THIS THESIS:** simple confidence or independent marginal baselines explain the result.

No paper claim advances beyond its gate. Negative findings are retained because they determine the next design rather than being hidden.

## Final result — 2026-08-23

**Decision: PIVOT.** Gate A conditionally passed in controlled PopQA and the lexical part of Gate B passed, but protocol-level closed-book stability remains unresolved. Gate C failed its registered criterion: on fixed-corpus SciFact, the joint MLP's safety-first advantage over direct answer risk had a 95% claim-bootstrap interval crossing zero, while joint linear modeling was slightly better in two other cost regimes.

The decisive conceptual failure is that the old `C` label measured retrieval of a matching annotated document, not evidence usability. Increasing retrieval depth from top-5 to top-10 improved evidence-document recall but reduced context and fusion verdict accuracy. The four-state result therefore cannot support a GO claim.

The successor program is defined in `docs/stage-2-plan.md`, with executable dependencies and pass/fail rules in `docs/experiment-registry.md`. It separates `R` (retrieval availability), `S` (retained evidence sufficiency proxy), `U` (model-conditioned utilization), `M_Pi` (protocol-distributed answerability), and realized action outcome. The mandatory next experiment is S2-E1; no new structured model should be trained before that audit freezes schema v2.
