"""Build the artifacts for the Credence experiment (a learned, calibrated 4-state
credibility estimator vs hand-crafted signals).

Pipeline (resumable: a stage is skipped when its artifact exists, --force-* redoes it):
  build -> extract -> split -> train(multi-seed) -> save artifacts

All metrics (discrimination / calibration / selective prediction / oracle ceiling)
are computed separately by scripts/analyze.py from the saved artifacts.

Usage:
  python scripts/run_experiment.py --data facts                 # offline smoke (80 built-in facts)
  python scripts/run_experiment.py --data popqa                 # main comparison (real PopQA)
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

# Make src/ importable regardless of the working directory (entry point lives under scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import config
from config import STATE_NAMES
from data import facts, popqa, counterfact
from data.build import build_samples, load_model
from features.extract import FEATURE_NAMES, Forwarder, extract_all
from estimator import train_estimator


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["facts", "popqa", "counterfact"],
                    default=config.DATA_SOURCE)
    ap.add_argument("--model", default=config.MODEL_NAME)
    ap.add_argument("--force-build", action="store_true")
    ap.add_argument("--force-extract", action="store_true")
    args = ap.parse_args()
    config.MODEL_NAME = args.model

    # Per-dataset artifacts live under artifacts/<tag>/ (isolated per data source).
    tag = args.data
    adir = config.ARTIFACT_DIR / tag
    adir.mkdir(parents=True, exist_ok=True)
    config.DATA_PATH = adir / "data.jsonl"
    config.FEATURE_PATH = adir / "features.npz"
    config.TOPK_PATH = adir / "topk_logits.pkl"
    config.ESTIMATOR_PATH = adir / "estimator.pt"

    logf = setup_logging()
    logging.info(f"log: {logf} | model: {args.model} | data source: {args.data}")

    # ------------------------------------------------ 1 build
    if args.force_build or not config.DATA_PATH.exists():
        logging.info("==> [1/3] build dataset")
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
        logging.info("==> [2/3] extract features (two forward passes)")
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
        logging.info("==> [2/3] load cached features")
        X = np.load(config.FEATURE_PATH)["X"]
        with open(config.TOPK_PATH, "rb") as f:
            topks = pickle.load(f)
    logging.info(f"  features X: {X.shape} ({len(FEATURE_NAMES)} dims)")

    # 3 split: leakage-free grouped split (a subject's correct/wrong variants stay together)
    gid = group_ids(samples)
    splits = make_group_split(gid, m_star, seed=config.SEED)

    # 4 train: multi-seed 4-way estimator
    logging.info("==> [3/3] train estimator (multi-seed)")
    model_est, T, norm, rep, per_seed = train_estimator(
        X, y, c_star, m_star, splits, epochs=config.EPOCHS, lr=config.LR,
        seeds=config.SEEDS, device="cuda")
    torch.save({"state_dict": model_est.state_dict(), "T": T, "norm": norm,
                "feat_dim": X.shape[1], "feature_names": FEATURE_NAMES},
               config.ESTIMATOR_PATH)
    logging.info(f"  test acc4={rep['acc4']:.3f} f1={rep['f1_macro']:.3f} "
                 f"ece={rep['ece']:.3f} auroc_c={rep['auroc_c']:.3f} auroc_m={rep['auroc_m']:.3f}")

    logging.info(f"\nartifacts ready -> {adir}")
    logging.info(f"  next: python scripts/analyze.py --data {args.data}")


if __name__ == "__main__":
    main()
