"""Part 3 figure and table from results/parts/part3.csv.

At each size N, every method's operating point is chosen as in Part 2: the
setting with the highest tune QPS among those with tune recall >= 0.95; its
test-split numbers are reported. Exact search has no setting.

The growth rate is the slope b of log(ms/query) against log(N) (least
squares over the sizes), i.e. time ~ N^b; 10^b is the slowdown for 10x data.

Outputs:
  results/figures/part3_scaling.{png,pdf}
  results/tables/part3_scaling.{md,tex}
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ann import results
from experiments.part2_report import COLORS, EXACT_COLOR, GRID, INK, MUTED, write_table

TARGET = 0.95
METHODS = ["Exact(FlatL2)", "IVF-Flat", "HNSW"]
LABEL = {"Exact(FlatL2)": "Exact", "IVF-Flat": "IVF-Flat", "HNSW": "HNSW"}
COLOR = {"Exact(FlatL2)": EXACT_COLOR, **COLORS}


def operating_points(df):
    key = ["n_base", "index", "build_params", "search_params"]
    tune = df[df.query_split == "tune"].set_index(key)
    test = df[df.query_split == "test"].set_index(key)
    pp = tune[["recall10", "qps"]].add_prefix("tune_").join(test.add_prefix("test_"),
                                                          how="right").reset_index()
    out = []
    for (n, idx), g in pp.groupby(["n_base", "index"]):
        if idx.startswith("Exact"):
            r = g.iloc[0]
            setting = "-"
        else:
            ok = g[g.tune_recall10 >= TARGET]
            if not len(ok):
                raise ValueError(f"{idx} at n={n} never reached tune recall {TARGET}")
            r = ok.loc[ok.tune_qps.idxmax()]
            b = json.loads(r.build_params)
            b.pop("efConstruction", None)
            setting = ", ".join(f"{k}={v}" for k, v in {**b, **json.loads(r.search_params)}.items())
        out.append({"n": int(n), "method": idx, "setting": setting,
                    "recall": r.test_recall10, "qps": r.test_qps,
                    "ms": 1000 / r.test_qps, "build_s": r.test_build_s,
                    "bytes_per_vector": r.test_bytes_per_vector})
    return pd.DataFrame(out)


def slope(n, ms):
    b, _ = np.polyfit(np.log10(n), np.log10(ms), 1)
    return b


def plot(op, sfx):
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    ns = np.array(sorted(op.n.unique()), dtype=float)
    for m in METHODS:
        g = op[op.method == m].sort_values("n")
        b = slope(g.n, g.ms)
        ax.plot(g.n, g.ms, "-o", color=COLOR[m], lw=2, ms=6,
                label=f"{LABEL[m]}  (slope b = {b:.2f}, 10x data -> {10 ** b:.1f}x time)")
    # Reference growth rates, anchored at the smallest size of exact and IVF-Flat.
    for m, p, txt in (("Exact(FlatL2)", 1.0, "~ N"), ("IVF-Flat", 0.5, "~ sqrt(N)")):
        y0 = op[(op.method == m) & (op.n == ns[0])].ms.iloc[0]
        ax.plot(ns, y0 * (ns / ns[0]) ** p, ":", color=MUTED, lw=1, zorder=1)
        ax.annotate(txt, (ns[-1], y0 * (ns[-1] / ns[0]) ** p), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=MUTED, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{int(n / 1000)}K" if n < 1e6 else f"{n / 1e6:g}M" for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("number of vectors N (first N SIFT vectors, log scale)", color=INK)
    ax.set_ylabel("ms per query at recall@10 >= 0.95\n(1 thread, log scale)", color=INK)
    ax.set_title("SIFT scaling: cost of reaching recall 0.95", fontsize=10, color=INK)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=1)
    fig.tight_layout()
    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(results.FIG_DIR, f"part3_scaling{sfx}.{ext}")
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    sfx = "_quick" if args.quick else ""
    df = pd.read_csv(os.path.join(results.PARTS_DIR, f"part3{sfx}.csv"))
    op = operating_points(df)
    os.makedirs(results.FIG_DIR, exist_ok=True)
    print("wrote", *plot(op, sfx))

    n0 = op.n.min()
    rows = []
    for m in METHODS:
        g = op[op.method == m].sort_values("n")
        base_ms = g[g.n == n0].ms.iloc[0]
        for _, r in g.iterrows():
            rows.append({
                "method": LABEL[m], "N": f"{r.n:,}", "setting @0.95": r.setting,
                "recall (test)": f"{r.recall:.3f}", "QPS": f"{r.qps:,.0f}",
                "ms/query": f"{r.ms:.3f}", "slowdown vs smallest": f"{r.ms / base_ms:.2f}x",
                "build (s)": f"{r.build_s:.1f}", "bytes/vector": f"{r.bytes_per_vector:.0f}",
            })
    write_table(f"part3_scaling{sfx}", pd.DataFrame(rows))

    print("\nGrowth (time ~ N^b):")
    for m in METHODS:
        g = op[op.method == m]
        b = slope(g.n, g.ms)
        print(f"  {LABEL[m]:9s} b = {b:.3f}   -> {10 ** b:.2f}x slower for 10x data")


if __name__ == "__main__":
    main()
