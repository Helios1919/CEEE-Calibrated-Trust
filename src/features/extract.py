"""Signal extractor: turns the S1-S14 signals from signal-bank.md into runnable code.

Signal groups (matching the signal-bank document):
  S1-S7   logit layer  (ARR): prob gap, entropy diff, JSD, argmax agreement, top-k
          overlap, confidence, margin
  S8-S10  hidden layer (CoRect / DoLa): parametric-suppression rank, FFN direct logit
          contribution, layer-contrast JSD
  S11-S14 attention layer (AdaRes / ReDeEP / Utilization): context attention quality,
          copy head ECS, knowledge FFN PKS, last-quarter key->answer attention trend

S15 (ProbeRAG probe) and S17-S19 (external retrieval confidence / staleness /
authority) are optional extensions, off by default.

Convention: each sample provides context / question / gold, with two forward passes —
  pri = question only (no context), ctx = context + question.
"""

import math
import gc

import torch
import torch.nn.functional as F

# Feature order and category split (used for training and ablation)
FEATURE_NAMES = [
    # logit (S1-S7)
    "max_prob_gap", "entropy_diff", "jsd", "argmax_same", "topk_overlap",
    "conf_pri", "conf_ctx", "margin_pri", "margin_ctx",
    # hidden (S8-S10)
    "param_suppress", "ffn_neg_count", "ffn_neg_sum", "dola_jsd",
    # attention (S11-S14)
    "ctx_attn_alpha", "copy_ecs", "knowledge_pks", "contest_trend",
]
CATEGORIES = {
    "logit":  list(range(0, 9)),
    "hidden": list(range(9, 13)),
    "attn":   list(range(13, 17)),
}


class Forwarder:
    """Register hooks to capture per-layer attention and FFN output deltas, and run full forward passes."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.L = model.config.num_hidden_layers
        # Unembedding matrix (for LogitLens and the FFN direct contribution u_l^T w_t)
        self.W_U = model.lm_head.weight.detach()
        self.final_norm = model.model.norm
        self._attn = {}
        self._mlp = {}
        self._hooks = []
        self._register()

    def _register(self):
        for l, layer in enumerate(self.model.model.layers):
            h1 = layer.self_attn.register_forward_hook(
                lambda m, i, o, l=l: self._attn.__setitem__(l, o[0].detach()))
            h2 = layer.mlp.register_forward_hook(
                lambda m, i, o, l=l: self._mlp.__setitem__(
                    l, (o[0] if isinstance(o, tuple) else o).detach()))
            self._hooks += [h1, h2]

    def clear(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._attn = {}
        self._mlp = {}

    def run(self, text):
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True,
                             output_attentions=True, use_cache=False)
        return {
            "input_ids": enc.input_ids[0],
            "hs": [h[0] for h in out.hidden_states],      # list[T,d] x (L+1)
            "attns": [a[0] for a in out.attentions],      # list[H,T,T] x L
            "logits": out.logits[0],                       # [T,V]
            "attn_delta": {l: self._attn[l][0] for l in self._attn},   # [T,d]
            "mlp_delta": {l: self._mlp[l][0] for l in self._mlp},       # [T,d]
        }

    def q_vec(self, state):
        """LogitLens: project a residual-stream vector state[d] through the final norm
        and lm_head into a vocab distribution [V]."""
        x = state.unsqueeze(0).unsqueeze(0)                 # [1,1,d]
        return self.model.lm_head(self.final_norm(x))[0, 0]  # [V]


def _dist(q):
    """softmax into a probability distribution (float for numerical stability). q: [V]"""
    return F.softmax(q.float(), dim=-1)


def _entropy(p):
    return -(p * torch.clamp(p.log(), min=-1e9)).sum().item()


def _jsd(p, q):
    m = 0.5 * (p + q)
    return _entropy(m) - 0.5 * (_entropy(p) + _entropy(q))  # JSD = H(m) - 1/2 H(p) - 1/2 H(q) (non-negative)


# --------------------------------------------------------------------------- #
# S1-S7 logit features (read at the last position / the "Answer:" position)
# --------------------------------------------------------------------------- #
def logit_features(pri_logits_last, ctx_logits_last):
    p_pri = _dist(pri_logits_last)
    p_ctx = _dist(ctx_logits_last)
    g = p_ctx.max().item() - p_pri.max().item()                     # S1 prob gap
    h_diff = _entropy(p_ctx) - _entropy(p_pri)                      # S2 entropy diff
    j = _jsd(p_ctx, p_pri) / math.log(2)                            # S3 JSD (normalized to [0,1], matching signal-bank's JSD/log2)
    same = int(p_ctx.argmax() == p_pri.argmax())                    # S4 argmax agreement
    topk = 5
    overlap = len(set(p_ctx.topk(topk).indices.tolist()) &
                  set(p_pri.topk(topk).indices.tolist())) / topk    # S5 top-k overlap
    return [g, h_diff, j, same, overlap,
            p_pri.max().item(), p_ctx.max().item(),                 # S6 confidence
            (p_pri.topk(2).values[0] - p_pri.topk(2).values[1]).item(),  # S7 margin pri
            (p_ctx.topk(2).values[0] - p_ctx.topk(2).values[1]).item()]  # S7 margin ctx


# --------------------------------------------------------------------------- #
# S8-S10 hidden features
# --------------------------------------------------------------------------- #
def hidden_features(fwd, hs_ctx, mlp_delta, target_tok):
    # S8 parametric suppression: rank of the target token under each layer's LogitLens,
    #    final-layer rank minus the minimum rank. The target token uses the model's own
    #    prediction (prior argmax), never gold — otherwise label leakage (see extract()).
    ranks = []
    for l in range(fwd.L):
        q = fwd.q_vec(hs_ctx[l + 1][-1])            # layer l output (last position) -> [V]
        r = int((q > q[target_tok]).sum().item())
        ranks.append(r)
    suppress = ranks[-1] - min(ranks)

    # S9 FFN direct logit contribution u_l^T w_t: projection onto the target token,
    #    counting negatives and their sum
    w_t = fwd.W_U[target_tok]
    neg_count = 0
    neg_sum = 0.0
    for l in range(fwd.L):
        u = mlp_delta[l][-1]                        # FFN output delta (last position)
        proj = float((u @ w_t).item())
        if proj < 0:
            neg_count += 1
            neg_sum += proj
    neg_count /= fwd.L

    # S10 DoLa: JSD between the final-layer and early-layer distributions
    #     (knowledge conflict is amplified in mature layers)
    p_final = _dist(fwd.q_vec(hs_ctx[-1][-1]))
    p_early = _dist(fwd.q_vec(hs_ctx[fwd.L // 2][-1]))
    dola = _jsd(p_final, p_early)

    return [float(suppress), float(neg_count), float(neg_sum), float(dola)]


# --------------------------------------------------------------------------- #
# S11-S14 attention features
# --------------------------------------------------------------------------- #
def attn_features(fwd, hs_ctx, attns_ctx, attn_delta, ctx_len):
    last = hs_ctx[-1].shape[0] - 1                   # the "Answer:" last position
    A = attns_ctx[-1]                                # last layer [H,T,T]

    # S11 context attention quality: mean attention from last position to the context region
    alpha = A[:, last, :ctx_len].mean().item()

    # S12 copy head ECS: cosine between the hidden state of the top-10% attended context
    #     tokens and the last-position hidden state
    att = A.mean(0)[last, :ctx_len]                  # [C]
    k = max(1, int(0.1 * ctx_len))
    topk_idx = att.topk(k).indices
    e = hs_ctx[-1][topk_idx].mean(0).float()
    ecs = F.cosine_similarity(e, hs_ctx[-1][last].float(), dim=0).item()

    # S13 knowledge FFN PKS: JSD of pre-FFN vs post-FFN distributions over the last
    #     quarter of layers (knowledge-write strength)
    pks_vals = []
    for l in range(fwd.L - fwd.L // 4, fwd.L):
        pre = hs_ctx[l][-1] + attn_delta[l][-1]      # post-attention residual (pre-FFN)
        post = hs_ctx[l + 1][-1]                      # post-FFN residual
        q_pre = _dist(fwd.q_vec(pre))
        q_post = _dist(fwd.q_vec(post))
        pks_vals.append(_jsd(q_pre, q_post))
    pks = sum(pks_vals) / len(pks_vals)

    # S14 contest trend: first-vs-last difference of the last-position -> context
    #     attention sum over the last quarter of layers
    qs = fwd.L - fwd.L // 4
    vals = [attns_ctx[l][:, last, :ctx_len].sum(dtype=torch.float32).item()
            for l in range(qs, fwd.L)]
    trend = vals[-1] - vals[0]

    return [float(alpha), float(ecs), float(pks), float(trend)]


# --------------------------------------------------------------------------- #
# assembly: one sample -> 17-dim feature vector
# --------------------------------------------------------------------------- #
def extract(fwd, sample):
    """sample: {context, question, gold, pri_prompt, ctx_prompt}
    returns (features[17], meta, z_pri[V], z_ctx[V])"""
    pri = fwd.run(sample["pri_prompt"])
    pri_logits = pri["logits"][-1]
    # Release the prior pass activations before running the longer evidence pass.
    # Keeping both attention tensors live makes SciFact's multi-abstract prompts
    # unnecessarily close to the GPU memory ceiling.
    del pri
    ctx = fwd.run(sample["ctx_prompt"])

    # Answers are generated after "Answer:"; the first token usually carries a leading
    # space (e.g. " Paris" not "Paris"). It must align with the model's actual predicted
    # first token, or downstream decoding EM would be systematically off.
    # gold_tok is used only for downstream EM comparison and must never enter features
    # (otherwise label leakage).
    gold_ids = fwd.tokenizer(" " + sample["gold"], add_special_tokens=False).input_ids
    gold_tok = gold_ids[0]                            # first token of a multi-token answer

    # S8/S9's target token uses the model's own prediction (prior argmax), not gold:
    # at deployment there is no gold, and pri_argmax detects whether the context pulls
    # the model away from its own parametric answer — no label leakage.
    pri_argmax = int(pri_logits.argmax().item())

    ctx_len = len(fwd.tokenizer(sample["context"], add_special_tokens=False).input_ids)

    feat = []
    feat += logit_features(pri_logits, ctx["logits"][-1])
    feat += hidden_features(fwd, ctx["hs"], ctx["mlp_delta"], pri_argmax)
    feat += attn_features(fwd, ctx["hs"], ctx["attns"], ctx["attn_delta"], ctx_len)

    meta = {
        "ctx_len": ctx_len,
        "gold_tok": gold_tok,
        "pri_argmax": pri_argmax,
        "ctx_argmax": int(ctx["logits"][-1].argmax().item()),
    }
    return feat, meta, pri_logits, ctx["logits"][-1]


def _topk_cache(z_pri, z_ctx, gold_tok, k):
    """Compress [V] final-layer logits to the top-K union tokens' logits (lossless for
    downstream decoding). Returns {"tokens": int[U], "z_pri": f32[U], "z_ctx": f32[U], "gold_tok": int}."""
    idx = torch.cat([z_pri.topk(k).indices, z_ctx.topk(k).indices,
                     torch.tensor([gold_tok], device=z_pri.device)])
    idx = torch.unique(idx)
    return {
        "tokens": idx.cpu().numpy(),
        "z_pri": z_pri[idx].float().cpu().numpy(),
        "z_ctx": z_ctx[idx].float().cpu().numpy(),
        "gold_tok": gold_tok,
    }


def extract_all(fwd, samples, top_k=100, progress=True):
    """Batch extraction, returns (X[N,17], metas, topk_caches[list])."""
    import numpy as np
    X = []
    metas = []
    topks = []
    n = len(samples)
    for i, s in enumerate(samples):
        if progress and (i + 1) % 25 == 0:
            print(f"  extract {i + 1}/{n}")
        feat, meta, z_pri, z_ctx = extract(fwd, s)
        X.append(feat)
        metas.append(meta)
        topks.append(_topk_cache(z_pri, z_ctx, meta["gold_tok"], top_k))
        del z_pri, z_ctx
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.asarray(X, dtype=np.float32), metas, topks
