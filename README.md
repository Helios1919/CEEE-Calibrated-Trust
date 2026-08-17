# Credence — A Learned, Calibrated Trustworthiness Estimator for RAG

**Credence** replaces hand-crafted trust heuristics with a small, *learned*, and
*calibrated* estimator that scores whether the retrieved context ($c^*$) and the
model's parametric memory ($m^*$) are correct. Given a question–context pair, it
outputs two probabilities:

- $p_c = P(\text{context is correct})$
- $p_m = P(\text{memory is correct})$

which are then used to judge trustworthiness — flagging likely-wrong answers via
selective prediction — instead of relying on manually designed gates.

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

- **Learned > hand-crafted (classification).** The MLP's 4-way macro-F1 and
  per-state AUROC beat every single hand-crafted signal (which can only be
  thresholded into a $c^*$ binary classifier) and a linear logistic.
- **Calibration.** Temperature scaling keeps the ECE low, so $p_c$/$p_m$ are
  usable probabilities — something a scalar heuristic cannot provide.
- **Selective prediction.** Rejecting the likely-double-wrong samples (scored by
  $P(\text{double-wrong})$) lifts answer EM sharply at low coverage, beating the
  best hand-crafted confidence signal.

### Scope and limitations

- Contexts are synthesized by corpus substitution (NQ-Swap recipe) over real PopQA
  and CounterFact questions. Wiring a real retriever is a documented next step.
- The $c^*$ signal (context correctness) is still weak for synthetic one-line
  contexts (AUROC ≈ 0.65), which caps downstream utility; richer contexts are
  the main lever for improvement.

## Repository layout

```text
src/                    core library (importable as top-level modules)
  config.py             global config (CRED_MODEL / CRED_DATA / CRED_N / CRED_SEED ...)
  estimator.py          4-way MLP + temperature scaling + AUROC/ECE metrics
  data/                 ITEM schema loaders -> four-state labels
    build.py            builds four-state labels (closed-book m* + context c*)
    facts.py            built-in 80-item offline smoke set
    popqa.py            real PopQA loader (corpus-substituted contexts)
    counterfact.py      CounterFact loader (corpus-substituted contexts)
  features/             signal extraction
    extract.py          S1–S14 signals (two forward passes + hooks + LogitLens)
scripts/                entry points
  run_experiment.py     builds artifacts (build → extract → train → save)
  analyze.py            computes all metrics from the saved artifacts
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

The pipeline is resumable: each stage is skipped when its artifact already exists
(`data.jsonl`, `features.npz`, `topk_logits.pkl`, `estimator.pt`). Metrics are
computed afterward with `python scripts/analyze.py --data popqa`.

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
python scripts/analyze.py --data popqa   # metrics
```

On a cluster, use `sbatch scripts/slurm.sh`.

## Representative results

PopQA, 4000 samples (2000 items × 2 contexts), `Qwen/Qwen2.5-7B` (base), test
n=808. Leakage-free split: train/val/test are partitioned by `(relation, subject)`
so a subject's two context variants never cross a split boundary.

| metric | MLP (ours) | logistic (linear) | best single signal |
|--------|-----------:|------------------:|-------------------:|
| AUROC($c^*$) / AUROC($m^*$) | 0.648 / 0.910 | 0.655 / 0.900 | 0.566 / 0.895 |
| 4-way macro-F1 / balanced acc | 0.543 / 0.521 | 0.528 / 0.503 | — (binary only) |
| ECE / Brier | 0.011 / 0.541 | 0.026 / 0.546 | — (no probability) |
| AURC (selective prediction, lower better) | 0.415 | — | 0.443 |

At coverage 16% (reject when $P(\text{double-wrong}) \ge 0.30$), answered EM
rises from 48.8% to 70.0%. The oracle routing ceiling given the four-state
distribution is 53.6% — so routing EM is structurally capped, while selective
prediction and calibration are where the learned estimator wins.

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
