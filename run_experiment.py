"""主实验：端到端验证 Credence（学习到的校准可信度估计器 > 手工信号）。

流程（断点续跑：某阶段产物已存在即跳过，--force-* 重做）：
  build → extract → split → train(多种子) → 判别基线 → 解码对比 → 消融 → 泛化 → 跨模型 → 汇总

用法：
  python run_experiment.py --data facts                 # 离线冒烟（80 条内置事实）
  python run_experiment.py --data popqa                 # 正式对比（真实 PopQA）
  python run_experiment.py --skip-ablation --skip-generalization   # 快速版
"""

import argparse
import json
import logging
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import config
from config import STATE_NAMES
from data import facts, popqa, counterfact
from data.build import build_samples, load_model
from features.extract import CATEGORIES, FEATURE_NAMES, Forwarder, extract_all
from estimator import (evaluate_probs, predict_cm, predict_probs, train_estimator)
from baselines import (best_single_signal, logistic_auc, run_decoders,
                       single_signal_aurocs)


def setup_logging():
    config.LOG_DIR.mkdir(exist_ok=True)
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logf = config.LOG_DIR / f"run_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(logf, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])
    return logf


def get_items(source):
    if source == "facts":
        return facts.make_items(seed=config.SEED)
    if source == "popqa":
        return popqa.make_items(split=config.POPQA_SPLIT, n=config.POPQA_N,
                                max_per_prop=config.POPQA_MAX_PER_PROP,
                                seed=config.SEED)
    if source == "counterfact":
        return counterfact.make_items(n=config.COUNTERFACT_N, seed=config.SEED)
    raise ValueError(source)


def load_samples(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _split(a, y_a, test_size, seed):
    try:
        return train_test_split(a, test_size=test_size, random_state=seed,
                                stratify=y_a)
    except ValueError:
        # 某类样本过少导致 stratified 失败时，退化为随机切分
        return train_test_split(a, test_size=test_size, random_state=seed)


def group_ids(samples):
    """为每个样本赋 (relation, subject) 组 id，供无泄漏分组切分。"""
    keys = {}
    ids = []
    for s in samples:
        k = (s["relation"], s["subject"])
        if k not in keys:
            keys[k] = len(keys)
        ids.append(keys[k])
    return np.array(ids)


def make_group_split(groups, group_label, seed=0):
    """按组切分 train/val/test：同一 subject 的两条变体（正确/错误上下文）永不跨 split。

    数据构建阶段每个 subject 产出 c*=1 与 c*=0 两条样本，且共享 question/上下文文本
    与相同的 m_star。若按样本随机切分，模型可在 train 上"背下"某 subject 的 m*，
    再凭相同文本在 test 上认出同一 subject 直接猜中 m*，导致标签泄漏、指标虚高。

    groups: [N] 组 id（group_ids() 产出）；group_label: [N] 组级标签（用 m_star 分层，
    同一组内 m_star 恒等，故组级分层等价于样本级分层且无泄漏）。
    返回 {"tr","va","te"} 的样本索引（0..N-1）。
    """
    uniq_g, first_idx = np.unique(groups, return_index=True)
    g_lab = np.asarray(group_label)[first_idx]          # 每组取一条标签（组内恒等）
    g_arr = np.arange(len(uniq_g))

    g_tr, g_rest = _split(g_arr, g_lab, 1 - config.TRAIN_FRAC, seed)
    g_va, g_te = _split(g_rest, g_lab[g_rest],
                        config.TEST_FRAC / (config.VAL_FRAC + config.TEST_FRAC),
                        seed)

    def _expand(g_sel):
        return np.where(np.isin(groups, uniq_g[g_sel]))[0]

    return {"tr": _expand(g_tr), "va": _expand(g_va), "te": _expand(g_te)}


def sanitize(o):
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [sanitize(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def mean_std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["facts", "popqa", "counterfact"],
                    default=config.DATA_SOURCE)
    ap.add_argument("--model", default=config.MODEL_NAME)
    ap.add_argument("--force-build", action="store_true")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-generalization", action="store_true")
    ap.add_argument("--skip-decoder", action="store_true")
    args = ap.parse_args()
    config.MODEL_NAME = args.model

    # 按数据源隔离产物路径：facts/popqa 各自独立，避免 --data 切换时复用/覆盖彼此的数据
    tag = args.data
    config.DATA_PATH = config.ROOT / f"data_{tag}.jsonl"
    config.FEATURE_PATH = config.ROOT / f"features_{tag}.npz"
    config.TOPK_PATH = config.ROOT / f"topk_logits_{tag}.pkl"
    config.ESTIMATOR_PATH = config.ROOT / f"estimator_{tag}.pt"
    # 结果按数据源隔离：results_{tag}.json，避免多数据源互相覆盖
    config.RESULT_PATH = config.ROOT / f"results_{tag}.json"

    logf = setup_logging()
    logging.info(f"日志: {logf} | 模型: {args.model} | 数据源: {args.data}")

    # ------------------------------------------------ 1 build
    if args.force_build or not config.DATA_PATH.exists():
        logging.info("==> [1/8] 构建数据集")
        items = get_items(args.data)
        logging.info(f"  ITEM 数: {len(items)}")
        model, tok = load_model(args.model)
        build_samples(model, tok, items, config.DATA_PATH, config.SEED)
        del model, tok
        torch.cuda.empty_cache()

    samples = load_samples(config.DATA_PATH)
    y = np.array([s["state"] for s in samples])
    c_star = np.array([s["c_star"] for s in samples])
    m_star = np.array([s["m_star"] for s in samples])
    dist = {STATE_NAMES[i]: int((y == i).sum()) for i in range(4)}
    logging.info(f"  样本 {len(samples)} 条，分布 {dist}")

    # ------------------------------------------------ 2 extract
    if args.force_extract or not (config.FEATURE_PATH.exists() and config.TOPK_PATH.exists()):
        logging.info("==> [2/8] 提取特征（两次前向）")
        model, tok = load_model(args.model)
        fwd = Forwarder(model, tok)
        X, metas, topks = extract_all(fwd, samples, top_k=config.TOP_K)
        fwd.clear()
        np.savez(config.FEATURE_PATH, X=X, y=y, c_star=c_star, m_star=m_star)
        with open(config.TOPK_PATH, "wb") as f:
            pickle.dump(topks, f)
        del model, tok
        torch.cuda.empty_cache()
    else:
        logging.info("==> [2/8] 载入缓存特征")
        X = np.load(config.FEATURE_PATH)["X"]
        with open(config.TOPK_PATH, "rb") as f:
            topks = pickle.load(f)
    logging.info(f"  特征 X: {X.shape}（{len(FEATURE_NAMES)} 维）")

    # ------------------------------------------------ 3 split
    # 无泄漏分组切分：同一 subject 的正确/错误上下文两条变体绑定在同一集合。
    gid = group_ids(samples)
    splits = make_group_split(gid, m_star, seed=config.SEED)
    te = splits["te"]

    # ------------------------------------------------ 4 train
    logging.info("==> [3/8] 训练估计器（多种子）")
    model_est, T, norm, rep, per_seed = train_estimator(
        X, y, c_star, m_star, splits, epochs=config.EPOCHS, lr=config.LR,
        seeds=config.SEEDS, device="cuda")
    torch.save({"state_dict": model_est.state_dict(), "T": T, "norm": norm,
                "feat_dim": X.shape[1], "feature_names": FEATURE_NAMES},
               config.ESTIMATOR_PATH)
    logging.info(f"  test acc4={rep['acc4']:.3f} f1={rep['f1_macro']:.3f} "
                 f"ece={rep['ece']:.3f} auroc_c={rep['auroc_c']:.3f} auroc_m={rep['auroc_m']:.3f}")
    seed_stats = {k: {"mean": mean_std([r[k] for r in per_seed])[0],
                      "std": mean_std([r[k] for r in per_seed])[1]}
                  for k in ["acc4", "f1_macro", "ece", "auroc_c", "auroc_m"]}

    p_c_te, p_m_te = predict_cm(model_est, X[te], T, norm, "cuda")

    # ------------------------------------------------ 5 判别基线
    logging.info("==> [4/8] 单信号 vs 学习判别（AUROC，阈值无关）")
    aurocs = single_signal_aurocs(
        X[splits["tr"]], X[te], c_star[splits["tr"]], c_star[te],
        m_star[splits["tr"]], m_star[te])
    bcn, bca, bmn, bma = best_single_signal(aurocs)
    log_c = logistic_auc(X[splits["tr"]], X[te], c_star[splits["tr"]], c_star[te])
    log_m = logistic_auc(X[splits["tr"]], X[te], m_star[splits["tr"]], m_star[te])
    logging.info(f"  最佳单信号  c*: {bcn}={bca:.3f}   m*: {bmn}={bma:.3f}")
    logging.info(f"  logistic(全特征线性)  c*={log_c:.3f}  m*={log_m:.3f}")
    logging.info(f"  MLP(全特征非线性)     c*={rep['auroc_c']:.3f}  m*={rep['auroc_m']:.3f}")

    # ------------------------------------------------ 6 解码
    decoders = None
    if not args.skip_decoder:
        logging.info("==> [5/8] 下游解码对比（单 token EM）")
        # CoRect 门控阈值在训练集上估计（避免测试集泄漏）
        corect_thresh = float(np.median(X[splits["tr"]][:, 9]))
        decoders = run_decoders([topks[i] for i in te], X[te], c_star[te], m_star[te],
                                p_c_te, p_m_te, corect_thresh=corect_thresh)
        for name, d in decoders.items():
            logging.info(f"  {name:<12} EM={d['em']:.3f} 纠正={d['em_correction']:.3f} "
                         f"抵抗={d['em_resistance']:.3f}")

    # ------------------------------------------------ 7 消融
    ablation = None
    if not args.skip_ablation:
        logging.info("==> [6/8] 特征分组消融")
        ablation = {}
        for cat, cols in CATEGORIES.items():
            mask = np.ones(X.shape[1], dtype=bool)
            mask[cols] = False
            if mask.sum() == 0:
                continue
            _, _, _, repm, _ = train_estimator(
                X[:, mask], y, c_star, m_star, splits, epochs=config.EPOCHS,
                lr=config.LR, seeds=config.SEEDS, device="cuda")
            ablation[cat] = {"acc4": repm["acc4"], "auroc_c": repm["auroc_c"],
                             "ece": repm["ece"]}
            logging.info(f"  去掉 {cat:<6}: acc4={repm['acc4']:.3f} auroc_c={repm['auroc_c']:.3f}")

    # ------------------------------------------------ 8 泛化
    gen = None
    if not args.skip_generalization:
        logging.info("==> [7/8] 泛化（held-out relation）")
        rels = np.array([s["relation"] for s in samples])
        if args.data == "facts":
            held = set(config.HELD_OUT_RELATIONS)
        else:
            held = set(sorted(set(rels))[::5])          # 每 5 个 prop 留 1 个
        te_idx = np.array([i for i, r in enumerate(rels) if r in held])
        rest_idx = np.array([i for i, r in enumerate(rels) if r not in held])
        if len(te_idx) and len(rest_idx):
            sub = make_group_split(gid[rest_idx], m_star[rest_idx], seed=config.SEED)
            gsplits = {"tr": rest_idx[sub["tr"]], "va": rest_idx[sub["va"]], "te": te_idx}
            _, _, _, grep, _ = train_estimator(
                X, y, c_star, m_star, gsplits, epochs=config.EPOCHS, lr=config.LR,
                seeds=config.SEEDS, device="cuda")
            gen = {"held_out": sorted(held), "acc4": grep["acc4"],
                   "auroc_c": grep["auroc_c"], "n_test": grep["n_test"]}
            logging.info(f"  held-out {sorted(held)}: acc4={grep['acc4']:.3f} "
                         f"auroc_c={grep['auroc_c']:.3f} (n={grep['n_test']})")

    # ------------------------------------------------ 9 跨模型
    cross = None
    if config.CROSS_MODEL:
        logging.info(f"==> [8/8] 跨模型泛化 -> {config.CROSS_MODEL}")
        model2, tok2 = load_model(config.CROSS_MODEL)
        fwd2 = Forwarder(model2, tok2)
        X2, _, _ = extract_all(fwd2, samples, top_k=config.TOP_K)
        fwd2.clear()
        prob2 = predict_probs(model_est, X2, T, norm, "cuda")
        rep2 = evaluate_probs(prob2, y, c_star, m_star)
        cross = {"model": config.CROSS_MODEL, "acc4": rep2["acc4"],
                 "auroc_c": rep2["auroc_c"], "ece": rep2["ece"]}
        logging.info(f"  {config.CROSS_MODEL}: acc4={rep2['acc4']:.3f} "
                     f"auroc_c={rep2['auroc_c']:.3f}")
        del model2, tok2
        torch.cuda.empty_cache()

    # ------------------------------------------------ 汇总保存
    results = {
        "model": args.model,
        "data_source": args.data,
        "n_samples": int(len(y)),
        "class_distribution": dist,
        "estimator_test": rep,
        "estimator_seed_stats": seed_stats,
        "single_signal_aurocs": {k: {"auroc_c": v["auroc_c"], "auroc_m": v["auroc_m"]}
                                 for k, v in aurocs.items()},
        "best_single_signal": {"c": [bcn, bca], "m": [bmn, bma]},
        "logistic_linear": {"auroc_c": log_c, "auroc_m": log_m},
        "decoders": decoders,
        "ablation": ablation,
        "generalization": gen,
        "cross_model": cross,
    }
    with open(config.RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(sanitize(results), f, ensure_ascii=False, indent=2)
    logging.info(f"\n完成 -> {config.RESULT_PATH}  |  权重 -> {config.ESTIMATOR_PATH}")
    logging.info("结论判据：auroc_c(MLP) 应 > 最佳单信号 & logistic；ECE 应 < 0.1；")
    logging.info("         下游 EM：CRED-* 应 ≥ ARR/AdaCAD/CoRect 手工门控。")


if __name__ == "__main__":
    main()
