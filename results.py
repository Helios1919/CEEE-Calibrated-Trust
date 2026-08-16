"""Read results.json and print a readable summary table (after the main experiment).

Usage: python results.py [results.json]
"""

import json
import sys


def _pct(x):
    return f"{100 * x:.1f}%" if isinstance(x, (int, float)) else "  -  "


def print_one(path):
    r = json.load(open(path, encoding="utf-8"))

    print(f"\n{'=' * 72}")
    print(f"# Credence results  |  model={r['model']}  data={r['data_source']}  "
          f"N={r['n_samples']}")
    print(f"state distribution: {r['class_distribution']}\n")

    print("## 1. learned estimator (test set)")
    e = r["estimator_test"]
    print(f"  4-way acc={_pct(e['acc4'])}   F1(macro)={e['f1_macro']:.3f}   "
          f"ECE={e['ece']:.3f}   AUROC(c*)={e['auroc_c']:.3f}   AUROC(m*)={e['auroc_m']:.3f}")

    print("\n## 2. discrimination: learned vs hand-crafted (AUROC, higher is better)")
    bs = r["best_single_signal"]
    lg = r["logistic_linear"]
    print(f"  best single signal c*: {bs['c'][0]} = {bs['c'][1]:.3f}")
    print(f"  best single signal m*: {bs['m'][0]} = {bs['m'][1]:.3f}")
    print(f"  logistic(linear, all features): c*={lg['auroc_c']:.3f}  m*={lg['auroc_m']:.3f}")
    print(f"  MLP(nonlinear, all features): c*={e['auroc_c']:.3f}  m*={e['auroc_m']:.3f}")
    s = r["single_signal_aurocs"]
    top_c = sorted(s.items(), key=lambda kv: (kv[1]["auroc_c"] or -1), reverse=True)[:5]
    print("  top-5 single signals for c* discrimination:")
    for k, v in top_c:
        print(f"    {k:<22} {v['auroc_c']:.3f}")

    if r.get("decoders"):
        print("\n## 3. downstream decoding (single-token EM, higher is better)")
        print(f"  {'method':<20}{'overall':>8}{'corr':>8}{'resist':>8}{'agree':>8}{'dw':>8}")
        for name, d in r["decoders"].items():
            print(f"  {name:<20}{_pct(d['em']):>8}{_pct(d['em_correction']):>8}"
                  f"{_pct(d['em_resistance']):>8}{_pct(d['em_agreement']):>8}"
                  f"{_pct(d['em_double_wrong']):>8}")
        if r.get("baseline_params"):
            bp = r["baseline_params"]
            print(f"  baseline hyperparams (val-tuned): CAD_alpha={bp.get('cad_alpha', '-')} "
                  f"AdaCAD(theta={bp.get('adacad_theta', '-')},gamma={bp.get('adacad_gamma', '-')}) "
                  f"CoRect_thresh={bp.get('corect_thresh', '-')}")
        if r.get("abstention"):
            a = r["abstention"]
            print("\n## 3.5 abstention (selective prediction: refuse when P(double-wrong) >= threshold)")
            print(f"  {'thresh':>8}{'coverage':>10}{'answerEM':>10}")
            for th, cov, em in zip(a["thresholds"], a["coverage"], a["em"]):
                print(f"  {th:>8.2f}{cov * 100:>9.1f}%{_pct(em):>10}")
            print(f"  AURC (risk-coverage area, lower is better) = {a['aurc']:.3f}")

    if r.get("ablation"):
        print("\n## 4. feature ablation (acc4 / AUROC(c*) after removing that group)")
        for cat, v in r["ablation"].items():
            print(f"  drop {cat:<6}: acc4={_pct(v['acc4'])}  auroc_c={v['auroc_c']:.3f}")

    if r.get("generalization"):
        g = r["generalization"]
        print(f"\n## 5. generalization (held-out {g['held_out']}, n={g['n_test']})")
        print(f"  acc4={_pct(g['acc4'])}  auroc_c={g['auroc_c']:.3f}")

    if r.get("cross_model"):
        c = r["cross_model"]
        print(f"\n## 6. cross-model ({c['model']})")
        print(f"  acc4={_pct(c['acc4'])}  auroc_c={c['auroc_c']:.3f}  ece={c['ece']:.3f}")


def main():
    import glob
    files = sys.argv[1:]
    if not files:
        files = sorted(glob.glob("results_*.json"))
    if not files:
        print("no results_*.json found -- run run_experiment.py first")
        return
    for f in files:
        print_one(f)


if __name__ == "__main__":
    main()
