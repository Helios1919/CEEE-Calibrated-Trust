"""通用四态构建：ITEM 列表 → 带 model-aware 标签的样本（TriState-Bench 思路）。

对每个 ITEM：
  1. 封闭式前向（无上下文）贪心生成 → m*（模型记忆是否正确）。
  2. 生成正确上下文（含 gold）与错误上下文（含 distractor）。
  3. 每个 ITEM 产出 2 条样本：
       c*=1（正确上下文）+ m* → 一致(3) 或 纠正(2)
       c*=0（错误上下文）+ m* → 抵抗(1) 或 双错(0)

保存 JSONL（每条含 pri_prompt / ctx_prompt，供特征提取直接用），并打印四态分布。
"""

import json
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from config import STATE_NAMES


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def gen_closedbook(model, tok, prompt):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)


def answer_correct(gen, gold):
    """双向子串匹配：容忍多 token 答案的截断（如 gold="New York City"、gen="New York"）。"""
    a, b = norm(gold), norm(gen)
    return (a in b) or (b in a)


def build_samples(model, tok, items, out_path, seed):
    random.seed(seed)
    samples = []
    counts = [0, 0, 0, 0]

    for it in items:
        question = it["question"]
        pri_prompt = f"{question}\nAnswer:"
        gen = gen_closedbook(model, tok, pri_prompt)
        m_star = int(answer_correct(gen, it["gold"]))

        correct_ctx = f"According to Wikipedia, {it['correct_statement']}"
        wrong_ctx = f"According to Wikipedia, {it['wrong_statement']}"

        for c_star, ctx in [(1, correct_ctx), (0, wrong_ctx)]:
            state = c_star * 2 + m_star
            counts[state] += 1
            samples.append({
                "id": f"{it['relation']}-{it['subject']}-{'c' if c_star else 'w'}",
                "relation": it["relation"],
                "subject": it["subject"],
                "gold": it["gold"],
                "distractor": it["distractor"],
                "c_star": c_star,
                "m_star": m_star,
                "state": state,
                "state_name": STATE_NAMES[state],
                "question": question,
                "context": ctx,
                "pri_prompt": pri_prompt,
                "ctx_prompt": f"{ctx}\n\n{question}\nAnswer:",
            })

    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[build] 共 {len(samples)} 条样本 -> {out_path}")
    for st in range(4):
        print(f"  {STATE_NAMES[st]:<12} (state={st}): {counts[st]}")
    if counts[2] + counts[0] == 0:
        print("  ⚠ correction/double_wrong 为 0：该模型对这批事实太强，"
              "加冷门事实或换 base 模型（Qwen/Qwen2.5-7B）。")
    return samples


def load_model(model_name=None):
    model_name = model_name or config.MODEL_NAME
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=getattr(torch, config.TORCH_DTYPE),
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model, tok
