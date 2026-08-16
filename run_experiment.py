"""Main experiment: end-to-end validation of Credence (a learned, calibrated
credibility estimator beats hand-crafted signals).

Pipeline (resumable: a stage is skipped when its artifact exists, --force-* redoes it):
  build -> extract -> split -> train(multi-seed) -> discrimination baselines ->
  decode comparison -> ablation -> generalization -> cross-model -> summarize

Usage:
  python run_experiment.py --data facts                 # offline smoke (80 built-in facts)
  python run_experiment.py --data popqa                 # main comparison (real PopQA)
  python run_experiment.py --skip-ablation --skip-generalization   # quick run
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
from estimator import (evaluate_probs, p_cm_from_probs, predict_probs, train_estimator)
from baselines import (best_single_signal, evaluate_abstention, logistic_auc,
                       run_decoders, single_signal_aurocs, tune_baseline_params)


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
        # fall back to random split when a class is too rare for stratification
        return train_test_split(a, test_size=test_size, random_state=seed)


def group_ids(samples):
    """Assign each sample a (relation, subject) group id for leakage-free grouping."""
    keys = {}
    ids = []
    for s in samples:
        k = (s["relation"], s["subject"])
        if k not in keys:
            keys[k] = len(keys)
        ids.append(keys[k])
    return np.array(ids)


def make_group_split(groups, group_label, seed=0):
    """Split train/val/test by group: a subject's two variants (correct/wrong context)
    never cross a split boundary.

    The build stage produces two samples per subject (c*=1 and c*=0) that share the
    question/context text and the same m*. A sample-level random split would let the
    model "memorize" a subject's m* on train and then, given the same text on test,
    recognize the subject and guess m* directly — label leakage that inflates metrics.

    groups: [N] group ids (from group_ids()); group_label: [N] group-level labels
    (stratified by m*; m* is constant within a group, so group-level stratification
    equals sample-level stratification with no leakage).
    Returns sample indices (0..N-1) for {"tr","va","te"}.
    """
    uniq_g, first_idx = np.unique(groups, return_index=True)
    g_lab = np.asarray(group_label)[first_idx]          # one label per group (constant within)
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

    # Isolate artifact paths by data source: facts/popqa are independent, avoiding
    # reuse/overwrite of each other's data when --data switches.
    tag = args.data
    config.DATA_PATH = config.ROOT / f"data_{tag}.jsonl"
    config.FEATURE_PATH = config.ROOT / f"features_{tag}.npz"
    config.TOPK_PATH = config.ROOT / f"topk_logits_{tag}.pkl"
    config.ESTIMATOR_PATH = config.ROOT / f"estimator_{tag}.pt"
    # Results are isolated per data source too: results_{tag}.json, avoiding overwrite.
    config.RESULT_PATH = config.ROOT / f"results_{tag}.json"

    logf = setup_logging()
    logging.info(f"log: {logf} | model: {args.model} | data source: {args.data}")

    # ------------------------------------------------ 1 build
    if args.force_build or not config.DATA_PATH.exists():
        logging.info("==> [1/8] build dataset")
        items = get_items(args.data)
        logging.info(f"  ITEM count: {len(items)}")
        model, tok = load_model(args.model)
        build_samples(model, tok, items, config.DATA_PATH, config.SEED)
        del model, tok
        torch.cuda.empty_cache()

    samples = load_samples(config.DATA_PATH)
    y = np.array([s["state"] for s in samples])
    c_star = np.array([s["c_star"] for s in samples])
    m_star = np.array([s["m_star"] for s in samples])
    dist = {STATE_NAMES[i]: int((y == i).sum()) for i in range(4)}
    logging.info(f"  {len(samples)} samples, distribution {dist}")

    # ------------------------------------------------ 2 extract
    if args.force_extract or not (config.FEATURE_PATH.exists() and config.TOPK_PATH.exists()):
        logging.info("==> [2/8] extract features (two forward passes)")
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
        logging.info("==> [2/8] load cached features")
        X = np.load(config.FEATURE_PATH)["X"]
        with open(config.TOPK_PATH, "rb") as f:
            topks = pickle.load(f)
    logging.info(f"  features X: {X.shape} ({len(FEATURE_NAMES)} dims)")

    # ------------------------------------------------ 3 split
    # Leakage-free grouped split: a subject's correct/wrong context variants stay together.
    gid = group_ids(samples)
    splits = make_group_split(gid, m_star, seed=config.SEED)
    te = splits["te"]

    # ------------------------------------------------ 4 train
    logging.info("==> [3/8] train estimator (multi-seed)")
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

    probs_te = predict_probs(model_est, X[te], T, norm, "cuda")
    p_c_te, p_m_te = p_cm_from_probs(probs_te)

    # ------------------------------------------------ 5 discrimination baselines
    logging.info("==> [4/8] single signal vs learned discrimination (AUROC, threshold-free)")
    aurocs = single_signal_aurocs(
        X[splits["tr"]], X[te], c_star[splits["tr"]], c_star[te],
        m_star[splits["tr"]], m_star[te])
    bcn, bca, bmn, bma = best_single_signal(aurocs)
    log_c = logistic_auc(X[splits["tr"]], X[te], c_star[splits["tr"]], c_star[te])
    log_m = logistic_auc(X[splits["tr"]], X[te], m_star[splits["tr"]], m_star[te])
    logging.info(f"  best single signal  c*: {bcn}={bca:.3f}   m*: {bmn}={bma:.3f}")
    logging.info(f"  logistic(all features, linear)  c*={log_c:.3f}  m*={log_m:.3f}")
    logging.info(f"  MLP(all features, nonlinear)    c*={rep['auroc_c']:.3f}  m*={rep['auroc_m']:.3f}")

    # ------------------------------------------------ 6 decoding
    decoders = None
    base_params = None
    abstain = None
    if not args.skip_decoder:
        logging.info("==> [5/8] downstream decoding comparison (single-token EM)")
        # Baseline hyperparams are searched on validation (matching the learned side's
        # tuning budget, so the comparison is not unfair)
        base_params = tune_baseline_params(
            [topks[i] for i in splits["va"]], X[splits["va"]])
        logging.info(f"  baseline tuning (val): CAD_alpha={base_params['cad_alpha']:.2f} "
                     f"AdaCAD_theta={base_params['adacad_theta']:.2f} "
                     f"AdaCAD_gamma={base_params['adacad_gamma']:.2f} "
                     f"CoRect_thresh={base_params['corect_thresh']:.3f}")
        decoders = run_decoders([topks[i] for i in te], X[te], c_star[te], m_star[te],
                                p_c_te, p_m_te,
                                cad_alpha=base_params["cad_alpha"],
                                adacad_theta=base_params["adacad_theta"],
                                adacad_gamma=base_params["adacad_gamma"],
                                corect_thresh=base_params["corect_thresh"])
        for name, d in decoders.items():
            logging.info(f"  {name:<22} EM={d['em']:.3f} corr={d['em_correction']:.3f} "
                         f"resist={d['em_resistance']:.3f}")

        abstain = evaluate_abstention([topks[i] for i in te], probs_te)
        logging.info(f"  abstention (selective prediction) AURC={abstain['aurc']:.3f} (lower is better)")
        for th, cov, em in zip(abstain["thresholds"], abstain["coverage"], abstain["em"]):
            logging.info(f"    threshold P(dw)>={th:.2f} coverage={cov:.3f} answeredEM={em:.3f}")

    # ------------------------------------------------ 7 ablation
    ablation = None
    if not args.skip_ablation:
        logging.info("==> [6/8] feature-group ablation")
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
            logging.info(f"  drop {cat:<6}: acc4={repm['acc4']:.3f} auroc_c={repm['auroc_c']:.3f}")

    # ------------------------------------------------ 8 generalization
    gen = None
    if not args.skip_generalization:
        logging.info("==> [7/8] generalization (held-out relation)")
        rels = np.array([s["relation"] for s in samples])
        if args.data == "facts":
            held = set(config.HELD_OUT_RELATIONS)
        else:
            held = set(sorted(set(rels))[::5])          # hold out every 5th prop
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

    # ------------------------------------------------ 9 cross-model
    cross = None
    if config.CROSS_MODEL:
        logging.info(f"==> [8/8] cross-model generalization -> {config.CROSS_MODEL}")
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

    # ------------------------------------------------ summarize & save
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
        "baseline_params": base_params,
        "abstention": abstain,
        "ablation": ablation,
        "generalization": gen,
        "cross_model": cross,
    }
    with open(config.RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(sanitize(results), f, ensure_ascii=False, indent=2)
    logging.info(f"\ndone -> {config.RESULT_PATH}  |  weights -> {config.ESTIMATOR_PATH}")
    logging.info("verdict: AUROC(c*)(MLP) should beat best single signal & logistic; ECE < 0.1;")
    logging.info("         downstream EM: CRED-* should be >= ARR/AdaCAD/CoRect hand gates.")


if __name__ == "__main__":
    main()
