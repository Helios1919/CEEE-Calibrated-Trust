"""信号提取器：把信号库 signal-bank.md 里的 S1–S14 落成可运行代码。

信号分组（对应 signal-bank 文档）：
  S1–S7   logit 层  (ARR)：概率 gap、熵差、JSD、argmax 一致、top-k 重叠、置信度、margin
  S8–S10  hidden 层 (CoRect / DoLa)：参数抑制 rank、FFN 直接 logit 贡献、层对比 JSD
  S11–S14 attention 层 (AdaRes / ReDeEP / Utilization)：上下文注意力质量、复制头 ECS、
           知识 FFN PKS、末四分之一层 key→answer 注意力趋势

S15（ProbeRAG 探针）与 S17–S19（外部检索置信度/时效/权威）为可选扩展，默认关闭。

约定：每个样本给出 context / question / gold，两次前向——
  pri = 仅问题（无上下文），ctx = 上下文 + 问题。
"""

import torch
import torch.nn.functional as F

# 特征顺序与类别划分（供训练与消融用）
FEATURE_NAMES = [
    # logit (S1–S7)
    "max_prob_gap", "entropy_diff", "jsd", "argmax_same", "topk_overlap",
    "conf_pri", "conf_ctx", "margin_pri", "margin_ctx",
    # hidden (S8–S10)
    "param_suppress", "ffn_neg_count", "ffn_neg_sum", "dola_jsd",
    # attention (S11–S14)
    "ctx_attn_alpha", "copy_ecs", "knowledge_pks", "contest_trend",
]
CATEGORIES = {
    "logit":  list(range(0, 9)),
    "hidden": list(range(9, 13)),
    "attn":   list(range(13, 17)),
}


class Forwarder:
    """注册 hook 抓取每层 attention 输出增量与 FFN 输出增量，并跑完整前向。"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.L = model.config.num_hidden_layers
        # 未嵌入矩阵（用于 LogitLens 与 FFN 直接贡献 u_l^T w_t）
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
            "hs": [h[0] for h in out.hidden_states],      # list[T,d] × (L+1)
            "attns": [a[0] for a in out.attentions],      # list[H,T,T] × L
            "logits": out.logits[0],                       # [T,V]
            "attn_delta": {l: self._attn[l][0] for l in self._attn},   # [T,d]
            "mlp_delta": {l: self._mlp[l][0] for l in self._mlp},       # [T,d]
        }

    def q_vec(self, state):
        """LogitLens：把某个残差流向量 state[d] 经末层 norm + lm_head 投影成词表分布[V]。"""
        x = state.unsqueeze(0).unsqueeze(0)                 # [1,1,d]
        return self.model.lm_head(self.final_norm(x))[0, 0]  # [V]


def _dist(q):
    """softmax 成概率分布（转 float 保证数值稳定）。q: [V]"""
    return F.softmax(q.float(), dim=-1)


def _entropy(p):
    return -(p * torch.clamp(p.log(), min=-1e9)).sum().item()


def _jsd(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (_entropy(p) + _entropy(q)) - _entropy(m)


# --------------------------------------------------------------------------- #
# S1–S7 logit 特征（在最后位置 / "Answer:" 处读取）
# --------------------------------------------------------------------------- #
def logit_features(pri_logits_last, ctx_logits_last):
    p_pri = _dist(pri_logits_last)
    p_ctx = _dist(ctx_logits_last)
    g = p_ctx.max().item() - p_pri.max().item()                     # S1 概率 gap
    h_diff = _entropy(p_ctx) - _entropy(p_pri)                      # S2 熵差
    j = _jsd(p_ctx, p_pri)                                          # S3 JSD
    same = int(p_ctx.argmax() == p_pri.argmax())                    # S4 argmax 一致
    topk = 5
    overlap = len(set(p_ctx.topk(topk).indices.tolist()) &
                  set(p_pri.topk(topk).indices.tolist())) / topk    # S5 top-k 重叠
    return [g, h_diff, j, same, overlap,
            p_pri.max().item(), p_ctx.max().item(),                 # S6 置信度
            (p_pri.topk(2).values[0] - p_pri.topk(2).values[1]).item(),  # S7 margin pri
            (p_ctx.topk(2).values[0] - p_ctx.topk(2).values[1]).item()]  # S7 margin ctx


# --------------------------------------------------------------------------- #
# S8–S10 hidden 特征
# --------------------------------------------------------------------------- #
def hidden_features(fwd, hs_ctx, mlp_delta, target_tok):
    # S8 参数抑制：目标 token 在各层 LogitLens 下的 rank，末层 rank − 最小 rank。
    #    目标 token 用模型自身预测（先验 argmax），绝不用 gold——否则标签泄漏（见 extract()）。
    ranks = []
    for l in range(fwd.L):
        q = fwd.q_vec(hs_ctx[l + 1][-1])            # 层 l 输出（末位置）→ [V]
        r = int((q > q[target_tok]).sum().item())
        ranks.append(r)
    suppress = ranks[-1] - min(ranks)

    # S9 FFN 直接 logit 贡献 u_l^T w_t：对目标 token 的投影，统计负值与负和
    w_t = fwd.W_U[target_tok]
    neg_count = 0
    neg_sum = 0.0
    for l in range(fwd.L):
        u = mlp_delta[l][-1]                        # FFN 输出增量（末位置）
        proj = float((u @ w_t).item())
        if proj < 0:
            neg_count += 1
            neg_sum += proj
    neg_count /= fwd.L

    # S10 DoLa：末层分布 与 早期层分布 的 JSD（知识冲突在成熟层被放大）
    p_final = _dist(fwd.q_vec(hs_ctx[-1][-1]))
    p_early = _dist(fwd.q_vec(hs_ctx[fwd.L // 2][-1]))
    dola = _jsd(p_final, p_early)

    return [float(suppress), float(neg_count), float(neg_sum), float(dola)]


# --------------------------------------------------------------------------- #
# S11–S14 attention 特征
# --------------------------------------------------------------------------- #
def attn_features(fwd, hs_ctx, attns_ctx, attn_delta, ctx_len):
    last = hs_ctx[-1].shape[0] - 1                   # "Answer:" 末位置
    A = attns_ctx[-1].float()                        # 末层 [H,T,T]

    # S11 上下文注意力质量：末位置 → 上下文区域 的注意力均值
    alpha = A[:, last, :ctx_len].mean().item()

    # S12 复制头 ECS：注意力最高的前 10% 上下文 token 的隐状态，与末位置隐状态的余弦
    att = A.mean(0)[last, :ctx_len]                  # [C]
    k = max(1, int(0.1 * ctx_len))
    topk_idx = att.topk(k).indices
    e = hs_ctx[-1][topk_idx].mean(0).float()
    ecs = F.cosine_similarity(e, hs_ctx[-1][last].float(), dim=0).item()

    # S13 知识 FFN PKS：末四分之一层内，pre-FFN 与 post-FFN 分布的 JSD（知识写入强度）
    pks_vals = []
    for l in range(fwd.L - fwd.L // 4, fwd.L):
        pre = hs_ctx[l][-1] + attn_delta[l][-1]      # post-attention 残差（pre-FFN）
        post = hs_ctx[l + 1][-1]                      # post-FFN 残差
        q_pre = _dist(fwd.q_vec(pre))
        q_post = _dist(fwd.q_vec(post))
        pks_vals.append(_jsd(q_pre, q_post))
    pks = sum(pks_vals) / len(pks_vals)

    # S14 对抗趋势：末四分之一层内，末位置 → 上下文 注意力总和的"首尾差"
    qs = fwd.L - fwd.L // 4
    vals = [attns_ctx[l].float()[:, last, :ctx_len].sum().item()
            for l in range(qs, fwd.L)]
    trend = vals[-1] - vals[0]

    return [float(alpha), float(ecs), float(pks), float(trend)]


# --------------------------------------------------------------------------- #
# 总装：一个样本 → 17 维特征向量
# --------------------------------------------------------------------------- #
def extract(fwd, sample):
    """sample: {context, question, gold, pri_prompt, ctx_prompt}
    返回 (features[17], meta, z_pri[V], z_ctx[V])"""
    pri = fwd.run(sample["pri_prompt"])
    ctx = fwd.run(sample["ctx_prompt"])

    # 答案在 "Answer:" 之后生成，首 token 通常带前导空格（如 " Paris" 而非 "Paris"），
    # 必须与模型实际预测的首 token 对齐，否则下游解码 EM 会整体错位。
    # gold_tok 仅用于下游 EM 评测比对，绝不可进入特征（否则标签泄漏）。
    gold_ids = fwd.tokenizer(" " + sample["gold"], add_special_tokens=False).input_ids
    gold_tok = gold_ids[0]                            # 多 token 答案取首 token

    # S8/S9 的目标 token 用模型自身预测（先验 argmax），而非 gold：
    # 部署时没有 gold，用 pri_argmax 检测"模型是否被上下文拉离自身参数化答案"，无标签泄漏。
    pri_argmax = int(pri["logits"][-1].argmax().item())

    ctx_len = len(fwd.tokenizer(sample["context"], add_special_tokens=False).input_ids)

    feat = []
    feat += logit_features(pri["logits"][-1], ctx["logits"][-1])
    feat += hidden_features(fwd, ctx["hs"], ctx["mlp_delta"], pri_argmax)
    feat += attn_features(fwd, ctx["hs"], ctx["attns"], ctx["attn_delta"], ctx_len)

    meta = {
        "ctx_len": ctx_len,
        "gold_tok": gold_tok,
        "pri_argmax": pri_argmax,
        "ctx_argmax": int(ctx["logits"][-1].argmax().item()),
    }
    return feat, meta, pri["logits"][-1], ctx["logits"][-1]


def _topk_cache(z_pri, z_ctx, gold_tok, k):
    """把 [V] 末层 logits 压缩成 top-K 联合 token 的 logits（下游解码近似不失真）。
    返回 {"tokens": int[U], "z_pri": f32[U], "z_ctx": f32[U], "gold_tok": int}"""
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
    """批量提取，返回 (X[N,17], metas, topk_caches[list])"""
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
    return np.asarray(X, dtype=np.float32), metas, topks
