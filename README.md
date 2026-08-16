# Credence — A Learned, Calibrated Trustworthiness Estimator for RAG

**Credence** replaces hand-crafted trust heuristics with a small, *learned*, and
*calibrated* estimator that scores whether the retrieved context ($c^*$) and the
model's parametric memory ($m^*$) are correct. Given a question–context pair, it
outputs two probabilities:

- $p_c = P(\text{context is correct})$
- $p_m = P(\text{memory is correct})$

which are then used to route RAG decoding instead of manually designed gates.

## Motivation

Knowledge conflicts between a model's parametric memory and retrieved context are
the core failure mode of retrieval-augmented generation. Existing routing methods
(CAD, ARR, AdaCAD, CoRect, …) share three weaknesses:

1. **Uncalibrated** — their scores are not usable probabilities.
2. **Single signal** — each method relies on one hand-picked heuristic.
3. **Ignore the *double-wrong* case** — context *and* memory are both wrong.

Credence addresses all three by framing trust estimation as a *supervised learning*
problem over a *signal bank* of $14$ internal signals (expanded into a $17$-dim
feature vector) drawn from the logit, hidden, and attention layers, fused by a
compact MLP into four states.

## Four-state formulation

Each sample is labeled into one of four states from the ground-truth context
correctness $c^*$ and the model's own closed-book memory correctness $m^*$:

| state | $c^*$ | $m^*$ | name |
|------:|:-----:|:-----:|------|
| 0 | 0 | 0 | double-wrong |
| 1 | 0 | 1 | resistance |
| 2 | 1 | 0 | correction |
| 3 | 1 | 1 | agreement |

```text
state = c* * 2 + m*
p_c   = P(correction) + P(agreement)   # context is correct
p_m   = P(resistance)  + P(agreement)  # memory is correct
```

## What this repository demonstrates

- **Learned > hand-crafted (discrimination).** The MLP's AUROC for $c^*$/$m^*$
  beats every single hand-crafted signal *and* a linear logistic over all features.
- **Calibration.** Temperature scaling keeps the ECE low, so $p_c$/$p_m$ are
  usable as routing thresholds.
- **Learned gating > hand-crafted gating (decoding).** `CRED-hard`/`CRED-mix`
  outperform hand-gated baselines on single-token EM, especially on the hard
  *resistance* state.
- **Generalization.** The estimator transfers to held-out relation types.

### Scope and limitations

- Contexts are synthesized by corpus substitution (NQ-Swap recipe) over real PopQA
  and CounterFact questions. Wiring a real retriever is a documented next step.
- Decoding is single-token (first-token) EM, matching the ARR/CAD/AdaCAD/CoRect
  convention. Full multi-token generation is left as future work.

## Repository layout

```text
src/                    core library (importable as top-level modules)
  config.py             global config (CRED_MODEL / CRED_DATA / CRED_N / CRED_SEED ...)
  estimator.py          4-way MLP + temperature scaling + AUROC/ECE metrics
  baselines.py          single-signal discriminators + downstream decoders
                        (greedy / CAD / ARR / AdaCAD / CoRect / CRED-hard / CRED-mix)
  data/                 ITEM schema loaders -> four-state labels
    build.py            builds four-state labels (closed-book m* + context c*)
    facts.py            built-in 80-item offline smoke set
    popqa.py            real PopQA loader (corpus-substituted contexts)
    counterfact.py      CounterFact loader (corpus-substituted contexts)
  features/             signal extraction
    extract.py          S1–S14 signals (two forward passes + hooks + LogitLens)
scripts/                entry points
  run_experiment.py     end-to-end pipeline (build → extract → split → train → ...)
  results.py            readable summary for every artifacts/*/results.json
  run.sh / run_all.sh   one-shot / full re-run
  slurm.sh              cluster (SLURM) template
docs/                   design docs (signal-bank.md) and drafts (plan.md, paper.tex)
artifacts/              per-dataset run outputs (git-ignored)
requirements.txt        Python dependencies
```

## Quick start

```bash
pip install -r requirements.txt

# 1) Offline smoke test (80 built-in facts, no network, a few minutes)
bash scripts/run.sh --data facts

# 2) Full comparison (real PopQA, all items by default)
bash scripts/run.sh --data popqa

# 3) Second full dataset (CounterFact, all items)
bash scripts/run.sh --data counterfact
```

### Command-line flags

| flag | effect |
|------|--------|
| `--data {facts,popqa,counterfact}` | data source (default from `CRED_DATA`) |
| `--model NAME` | HF model id (default `Qwen/Qwen2.5-7B`, or `CRED_MODEL`) |
| `--force-build` | rebuild labels even if cached |
| `--force-extract` | re-extract features even if cached |
| `--skip-decoder` / `--skip-ablation` / `--skip-generalization` | skip stages for a quick run |

The pipeline is resumable: each stage is skipped when its artifact already exists
(`data.jsonl`, `features.npz`, `estimator.pt`, `topk_logits.pkl`).

## Running on a remote GPU (e.g. an A100)

```bash
# package and upload from your workstation
tar czf credence.tar.gz credence/
scp credence.tar.gz user@a100-host:~:

# on the remote: unpack, set up the env, run
tar xzf credence.tar.gz && cd credence
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
# (optional, A100 speed-up) pip install flash-attn --no-build-isolation

# point to a pre-downloaded model cache to avoid re-downloading
export HF_HOME=/data/hf_cache
huggingface-cli download Qwen/Qwen2.5-7B --local-dir /data/hf_cache/Qwen/Qwen2.5-7B

nohup bash scripts/run.sh --data popqa > run.out 2>&1 &
tail -f logs/run_*.log          # progress
python scripts/results.py       # summary table
```

On a cluster, use `sbatch scripts/slurm.sh`.

## Representative results

Leakage-free split: train/val/test are partitioned by `(relation, subject)`, so
a subject's two context variants never cross the split boundary (the old
sample-level split inflated AUROC by ~0.03). 3000 items × 2 contexts per
dataset, `Qwen/Qwen2.5-7B` (base), main seed (the estimator is trained over
seeds 0/1/2 for stability).

| metric | PopQA | CounterFact |
|--------|------:|------------:|
| AUROC($c^*$) / AUROC($m^*$) | 0.707 / 0.870 | 0.760 / 0.931 |
| best single signal ($c^*$ / $m^*$) | 0.583 / 0.841 | 0.643 / 0.807 |
| logistic (linear, all features) | 0.671 / 0.857 | 0.748 / 0.924 |
| 4-way accuracy / macro-F1 | 53.2% / 0.540 | 58.4% / 0.624 |
| ECE | 0.021 | 0.026 |
| `CRED-mix` overall EM | 51.7% | 54.4% |
| `CRED-mix` EM on *resistance* | 55.7% | 71.9% |

## FAQ

- **`correction`/`double_wrong` counts are zero.** The model is too strong for that
  fact batch. Use the base model `Qwen/Qwen2.5-7B` (closed-book is weaker and
  errs more) or `--data popqa` (natural long-tail).
- **PopQA download slow / fails.** Run `--data facts` for a smoke test first, or
  `export HF_ENDPOINT=https://hf-mirror.com` (mirror for Mainland China).
- **Out of GPU memory.** Use a smaller model via `CRED_MODEL=...` or `--model`.
  7B in bf16 needs ~15 GB.
- **Changing model / data.** Use `--model` / `--data`; a new dataset only needs to
  produce the ITEM schema (see `src/data/facts.py`).

## Roadmap

1. **Real retrieved passages** — swap the synthetic contexts in `data/popqa.py`
   for an actual retriever (e.g. Contriever) or NQ-Swap original data.
2. **More signals** — S15 probe and S17–S19 external signals
   (retrieval confidence / staleness / source authority); hooks already reserved.
3. **Multi-token generation-level EM** — upgrade decoding from first-token.
4. **More baselines & models** — COIECD / ProbeRAG / PH3; LLaMA-3-8B cross-model.
