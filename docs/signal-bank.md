# Signal Bank Reference (Credence · step 4.1)

> For each signal: **which layer** it is read from, its **formula**, and **pseudocode**.
> Pseudocode is illustrative, based on HuggingFace `output_hidden_states` / `output_attentions` or TransformerLens hooks.

## 0. Prerequisite: two forward passes

Convention: model M with L layers, vocabulary V, unembedding weight `W_U` (= `lm_head.weight`, which may or may not be tied with the embedding).

```python
import torch, torch.nn.functional as F
import numpy as np

def forward(model, ids, attention_mask=None):
    out = model(ids, attention_mask=attention_mask,
                output_hidden_states=True, output_attentions=True, use_cache=False)
    hs    = out.hidden_states      # list[L+1]: hs[0]=embedding, hs[-1]=final output
    attns = out.attentions         # list[L] × [B, n_head, T, T]
    logits = out.logits            # final-layer logits via lm_head
    return hs, attns, logits

hs_pri, attn_pri, z_pri = forward(model, ids_pri)   # no context: instruction + question
hs_ctx, attn_ctx, z_ctx = forward(model, ids_ctx)   # with context: instruction + passage + question

p_pri = F.softmax(z_pri[0, -1], dim=-1)   # last generation position
p_ctx = F.softmax(z_ctx[0, -1], dim=-1)
```

## 1. Logit-layer signals (cheapest, do first)

| # | Name | Formula / meaning | Source |
|---|---|---|---|
| S1 | Max-prob gap | `g = max_y p_ctx(y) − max_y p_pri(y)` | ARR gate |
| S2 | Entropy diff | `H(p_pri) − H(p_ctx)` (or store both) | information theory |
| S3 | JSD | `JSD(p_ctx ∥ p_pri) / log2` | ARR strength |
| S4 | argmax agreement | `1[argmax p_pri == argmax p_ctx]` | — |
| S5 | top-k overlap | `|topK(p_pri) ∩ topK(p_ctx)| / K` | — |
| S6 | confidence | `max p_pri`, `max p_ctx` (two values) | — |
| S7 | margin | `z[top1] − z[top2]` (one for prior, one for context) | — |

```python
def entropy(p): return -(p * p.log()).sum(-1)
def jsd(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(m.log(), p, reduction='sum') +
                  F.kl_div(m.log(), q, reduction='sum'))

def logit_features(p_pri, p_ctx, z_pri, z_ctx, K=10):
    g      = (p_ctx.max() - p_pri.max()).item()
    h_diff = (entropy(p_pri) - entropy(p_ctx)).item()
    j      = (jsd(p_pri, p_ctx) / np.log(2)).item()
    same   = float(p_pri.argmax() == p_ctx.argmax())
    top_pri = set(p_pri.topk(K).indices.tolist())
    top_ctx = set(p_ctx.topk(K).indices.tolist())
    overlap = len(top_pri & top_ctx) / K
    conf    = (p_pri.max().item(), p_ctx.max().item())
    m_pri = z_pri[0,-1].topk(2).values[0] - z_pri[0,-1].topk(2).values[1]
    m_ctx = z_ctx[0,-1].topk(2).values[0] - z_ctx[0,-1].topk(2).values[1]
    return [g, h_diff, j, same, overlap, *conf, m_pri.item(), m_ctx.item()]
```

## 2. Hidden / residual-stream signals ("where knowledge gets suppressed")

Use LogitLens: project each layer's hidden state back onto the vocabulary to get "early-decoded" logits.

> ⚠️ The target token `t` must be the model's *own* prediction (`argmax p_pri`), **never the gold answer**:
> the ground truth is unavailable at feature-extraction time, and using gold introduces label leakage that inflates discrimination metrics.

```python
def logit_lens(hs, W_U, ln):          # ln = final LayerNorm
    return [ln(h_l) @ W_U.T for h_l in hs]
```

- **S8 parameter suppression (CoRect)**: how much the rank of the target token `t` (the model's own prediction `argmax p_pri`; **gold is forbidden** — label leakage otherwise) changes across layers.

  ```python
  lens = logit_lens(hs_ctx, W_U, ln)
  ranks = [(l[-1] > l[-1][t]).sum().item() for l in lens]  # rank of t at each layer
  suppress = ranks[-1] - min(ranks)   # >0 means it ranked high early, then got suppressed
  ```

- **S9 FFN direct contribution (CoRect)**: take each layer's FFN output `u_l` (via hook) and project it onto the target-token direction.

  ```python
  w_t = W_U[t]
  proj = [(u_l[-1] @ w_t).item() for u_l in ffn_outputs]   # u_l^T w_t per layer
  neg_count = sum(1 for x in proj if x < 0)                 # count of suppressing layers
  neg_sum   = sum(x for x in proj if x < 0)                 # total suppression strength
  ```

- **S10 layer-contrast divergence (DoLa)**: mature = final layer, premature = early layer with maximal JSD.

  ```python
  p_final = F.softmax(lens[-1][-1], dim=-1)
  early_jsd = [jsd(p_final, F.softmax(lens[i][-1], dim=-1)).item()
               for i in early_layer_ids]      # early = first 1/3 of layers
  dola = max(early_jsd)
  ```

## 3. Attention signals ("does the model read and copy the context?")

- **S11 context attention quality (AdaRes α)**: mean attention from query tokens to context tokens (final layer).

  ```python
  A = attn_ctx[-1][0]                                  # [n_head, T, T]
  alpha = A[:, q_positions][:, :, c_positions].mean().item()
  ```

- **S12 copying-head ECS (ReDeEP)**: the last token attends to the top-k% context tokens; take their final hidden states' mean `e` and compute cosine similarity.

  ```python
  att = A.mean(0)[last_pos, c_positions]               # last token → context tokens
  topk = att.topk(k=max(1, int(0.1 * len(c_positions)))).indices
  e = hs_ctx[-1][0, topk].mean(0)
  ecs = F.cosine_similarity(e, hs_ctx[-1][0, last_pos], dim=0).item()
  ```

- **S13 knowledge-FFN PKS (ReDeEP)**: JSD between pre- and post-FFN LogitLens distributions, averaged over deep layers.

  ```python
  pks = [jsd(F.softmax(ln(pre) @ W_U.T, -1), F.softmax(ln(post) @ W_U.T, -1)).item()
         for pre, post in zip(pre_ffn_states, post_ffn_states)[late_layers]]  # late = last 1/3
  pks = float(np.mean(pks))
  ```

- **S14 knowledge contest (Utilization)**: trend of key→answer attention across the last quarter of layers.

  ```python
  if_ka = [A[:, a_positions][:, :, k_positions].sum().item() for A in attn_ctx[3*L//4:]]
  trend = if_ka[-1] - if_ka[0]     # a drop (negative) = "contest"
  ```

## 4. Probing signals (ProbeRAG-style)

- **S15 latent-conflict probe**: train a 3-layer MLP on the frozen model; input = mean-pooled final hidden states of context tokens; output = align/conflict probability.

  ```python
  z = probe(hs_ctx[-1][0, c_positions].mean(0))   # frozen features → probe
  p_conflict = z.sigmoid().item()
  ```

  Training data: aligned/conflict samples from MQuAKE or a self-built four-state set.

## 5. External signals (optional fallback, model-internal free)

| # | Signal | How to obtain |
|---|---|---|
| S17 | retrieval confidence | retriever's score for the passage (BM25 / DPR score) |
| S18 | timestamp | staleness = document date vs model knowledge cutoff |
| S19 | source authority | domain/source type (wiki/paper/forum), one-hot or scored |

## 6. Assembly + normalization

```python
def extract(model, ids_ctx, ids_pri, probe=None, external=None):
    # compute S1..S16 ...
    feats = [*logit_feats, suppress, neg_count, neg_sum, dola,
             alpha, ecs, pks, trend, p_conflict, *(external or [])]
    feats = np.array(feats, dtype=np.float32)
    feats = (feats - mean) / std       # standardize with training-set statistics
    return feats                        # shape [D], D≈20~30
```

The estimator (small MLP, 4-way softmax, temperature scaling for calibration):

```python
mlp = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 4))
logp = F.log_softmax(mlp(feats), dim=-1)      # training: NLL
p    = F.softmax(mlp(feats) / T, dim=-1)      # inference: temperature T for calibration
# marginalize into two trust scores
p_c = p[agreement] + p[correction]   # P(context correct)
p_m = p[agreement] + p[resistance]   # P(memory correct)
```

## 7. Engineering notes

1. `output_attentions=True` is memory-heavy — use hooks to grab only the needed layers (final + deep + early candidate layers) instead of storing everything.
2. FFN outputs `u_l` and pre/post-FFN residual streams require a registered `forward_hook` to capture.
3. Batch extraction + cache the feature matrix (compute a few thousand samples once, store as `.npy`), then train directly from it.
4. If `W_U` is tied with the embedding, read `lm_head.weight` under tie-weighting.
5. When double-wrong / resistance samples are scarce, apply class weighting or oversampling to the minority classes.
