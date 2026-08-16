"""读取 results.json，打印一份可读的结论表（跑完主实验后看结果用）。

用法：python results.py [results.json]
"""

import json
import sys


def _pct(x):
    return f"{100 * x:.1f}%" if isinstance(x, (int, float)) else "  -  "


def main(path=None):
    path = path or (sys.argv[1] if len(sys.argv) > 1 else "results.json")
    r = json.load(open(path, encoding="utf-8"))

    print(f"# Credence 实验结果  |  model={r['model']}  data={r['data_source']}  "
          f"N={r['n_samples']}")
    print(f"\n四态分布: {r['class_distribution']}\n")

    print("## 1. 学习估计器（test 集）")
    e = r["estimator_test"]
    print(f"  4路 acc={_pct(e['acc4'])}   F1(macro)={e['f1_macro']:.3f}   "
          f"ECE={e['ece']:.3f}   AUROC(c*)={e['auroc_c']:.3f}   AUROC(m*)={e['auroc_m']:.3f}")

    print("\n## 2. 判别层：学习 vs 手工（AUROC，越高越好）")
    bs = r["best_single_signal"]
    lg = r["logistic_linear"]
    print(f"  最佳单信号 c*: {bs['c'][0]} = {bs['c'][1]:.3f}")
    print(f"  最佳单信号 m*: {bs['m'][0]} = {bs['m'][1]:.3f}")
    print(f"  logistic(线性,全特征): c*={lg['auroc_c']:.3f}  m*={lg['auroc_m']:.3f}")
    print(f"  MLP(非线性,全特征):   c*={e['auroc_c']:.3f}  m*={e['auroc_m']:.3f}")
    s = r["single_signal_aurocs"]
    top_c = sorted(s.items(), key=lambda kv: (kv[1]["auroc_c"] or -1), reverse=True)[:5]
    print("  单信号 c* 判别力 top5:")
    for k, v in top_c:
        print(f"    {k:<22} {v['auroc_c']:.3f}")

    if r.get("decoders"):
        print("\n## 3. 下游解码（单 token EM，越高越好）")
        print(f"  {'方法':<12}{'总体':>8}{'纠正':>8}{'抵抗':>8}{'一致':>8}{'双错':>8}")
        for name, d in r["decoders"].items():
            print(f"  {name:<12}{_pct(d['em']):>8}{_pct(d['em_correction']):>8}"
                  f"{_pct(d['em_resistance']):>8}{_pct(d['em_agreement']):>8}"
                  f"{_pct(d['em_double_wrong']):>8}")

    if r.get("ablation"):
        print("\n## 4. 特征消融（去掉该类信号后的 acc4 / AUROC(c*)）")
        for cat, v in r["ablation"].items():
            print(f"  去掉 {cat:<6}: acc4={_pct(v['acc4'])}  auroc_c={v['auroc_c']:.3f}")

    if r.get("generalization"):
        g = r["generalization"]
        print(f"\n## 5. 泛化（held-out {g['held_out']}, n={g['n_test']}）")
        print(f"  acc4={_pct(g['acc4'])}  auroc_c={g['auroc_c']:.3f}")

    if r.get("cross_model"):
        c = r["cross_model"]
        print(f"\n## 6. 跨模型（{c['model']}）")
        print(f"  acc4={_pct(c['acc4'])}  auroc_c={c['auroc_c']:.3f}  ece={c['ece']:.3f}")


if __name__ == "__main__":
    main()
