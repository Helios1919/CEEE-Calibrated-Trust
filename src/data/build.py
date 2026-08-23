"""Common four-state builder: ITEM list -> model-aware labeled samples (TriState-Bench style).

For each ITEM:
  1. Closed-book forward (no context) greedy generation -> m* (is memory correct).
  2. Build the correct context (with gold) and the wrong context (with distractor).
  3. Each ITEM yields 2 samples:
       c*=1 (correct context) + m* -> agreement(3) or correction(2)
       c*=0 (wrong context)   + m* -> resistance(1) or double_wrong(0)

Saves JSONL (each line has pri_prompt / ctx_prompt for feature extraction) and prints
the four-state distribution.
"""

import hashlib
import json
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from config import STATE_NAMES
from evaluation import answer_initial_match


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


def gen_closedbook_batch(model, tok, prompts, batch_size=64):
    """Batch closed-book greedy generation (identical greedy results to gen_closedbook,
    but ~5-10x faster for large item sets). Left-pads so newly generated tokens are
    appended at a fixed offset (in_len) for every sample in the batch."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    old_side = tok.padding_side
    tok.padding_side = "left"
    outs = []
    try:
        model.eval()
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False,
                    pad_token_id=pad_id, eos_token_id=tok.eos_token_id,
                )
            in_len = enc.input_ids.shape[1]
            for j in range(len(batch)):
                outs.append(tok.decode(out[j][in_len:], skip_special_tokens=True))
    finally:
        tok.padding_side = old_side
    return outs


def answer_correct(gen, gold, aliases=None):
    """Match a complete answer at the start of a short-answer generation."""
    return bool(answer_initial_match(gen, gold, aliases))


def build_samples(model, tok, items, out_path, seed, batch_size=64):
    random.seed(seed)
    samples = []
    counts = [0, 0, 0, 0]

    # Batch closed-book generation (greedy results identical to per-item, much faster)
    pri_prompts = [f"{it['question']}\nAnswer:" for it in items]
    gens = gen_closedbook_batch(model, tok, pri_prompts, batch_size)

    for it, gen in zip(items, gens):
        question = it["question"]
        pri_prompt = f"{question}\nAnswer:"
        m_star = int(answer_correct(gen, it["gold"], it.get("aliases")))

        correct_ctx = f"According to Wikipedia, {it['correct_statement']}"
        wrong_ctx = f"According to Wikipedia, {it['wrong_statement']}"

        for c_star, ctx in [(1, correct_ctx), (0, wrong_ctx)]:
            state = c_star * 2 + m_star
            counts[state] += 1
            variant = "correct_context" if c_star else "wrong_context"
            item_id = it.get("item_id")
            if not item_id:
                identity = "\x1f".join([it["relation"], it["subject"], question, it["gold"]])
                item_id = "item-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            samples.append({
                "id": f"{item_id}:{variant}",
                "item_id": item_id,
                "context_variant": variant,
                "relation": it["relation"],
                "subject": it["subject"],
                "gold": it["gold"],
                "aliases": it.get("aliases", []),
                "distractor": it["distractor"],
                "closedbook_answer": gen,
                "closedbook_protocol": {
                    "prompt": pri_prompt,
                    "decoding": "greedy",
                    "max_new_tokens": config.MAX_NEW_TOKENS,
                },
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

    print(f"[build] {len(samples)} samples -> {out_path}")
    for st in range(4):
        print(f"  {STATE_NAMES[st]:<12} (state={st}): {counts[st]}")
    if counts[2] + counts[0] == 0:
        print("  WARNING correction/double_wrong is 0: the model is too strong for this fact "
              "batch; add rarer facts or use the base model (Qwen/Qwen2.5-7B).")
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
