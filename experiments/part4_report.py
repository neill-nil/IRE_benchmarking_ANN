"""Part 4 figure and tables from results/part4_mechanism.csv.

Main figure: recall@10 and QPS against the shift size alpha for queries from
the new (shifted) region, one colour per index; solid = stale (built on old
data, new data added), dashed = rebuilt from scratch.

Outputs:
  results/figures/part4_drift.{png,pdf}
  results/tables/part4_drift.{md,tex}        (new queries: recall, QPS, mechanism)
  results/tables/part4_drift_oldq.{md,tex}   (old queries: recall, QPS)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ann import results
from experiments.part2_report import COLORS, GRID, INK, MUTED, write_table

ORDER = ["IVF-Flat", "IVF-PQ", "HNSW"]
STYLE = {"stale": "-", "rebuilt": "--"}


def style_axes(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(which="both", colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)


def plot(m, sfx):
    new = m[m.queries == "new"]
    alphas = sorted(new.alpha.unique())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.6), dpi=150)
    for kind in ORDER:
        for version in ("stale", "rebuilt"):
            g = new[(new["index"] == kind) & (new.version == version)].sort_values("alpha")
            kw = dict(color=COLORS[kind], ls=STYLE[version], lw=2 if version == "stale" else 1.4,
                      marker="o", ms=4, label=f"{kind} ({version})")
            a1.plot(g.alpha, g.recall10, **kw)
            a2.plot(g.alpha, g.qps, **kw)
    for ax in (a1, a2):
        ax.set_xticks(alphas)
        ax.set_xticklabels(["0\n(control)" if a == 0 else f"{a:g}" for a in alphas])
        ax.set_xlabel("shift size alpha (x data spread)", color=INK, fontsize=9)
        style_axes(ax)
    a1.set_ylabel("recall@10 (new-region queries)", color=INK, fontsize=9)
    a2.set_ylabel("queries per second (1 thread, log)", color=INK, fontsize=9)
    a2.set_yscale("log")
    a1.set_title("Recall", fontsize=10, color=INK)
    a2.set_title("Speed", fontsize=10, color=INK)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Drift: index built on old SIFT data, then shifted data added",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(results.FIG_DIR, f"part4_drift{sfx}.{ext}")
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def fmt(v, f):
    return "-" if pd.isna(v) else format(v, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    sfx = "_quick" if args.quick else ""
    m = pd.read_csv(os.path.join(results.RESULTS_DIR, f"part4{sfx}_mechanism.csv"))
    os.makedirs(results.FIG_DIR, exist_ok=True)
    print("wrote", *plot(m, sfx))

    rows = []
    for _, r in m[m.queries == "new"].sort_values(["index", "version", "alpha"]).iterrows():
        rows.append({
            "index": r["index"], "version": r.version, "alpha": f"{r.alpha:g}",
            "recall@10": f"{r.recall10:.3f}", "QPS": f"{r.qps:,.0f}",
            "distances/query": f"{r.dist_per_query:,.0f}",
            "new vecs in top-1% lists": fmt(r.get("new_in_top1pct_lists"), ".1%"),
            "largest list": fmt(r.get("largest_list"), ",.0f"),
            "PQ rel. error (new)": fmt(r.get("pq_rel_error_new"), ".3f"),
            "new->old edges": fmt(r.get("new_edges_to_old"), ".1%"),
        })
    write_table(f"part4_drift{sfx}", pd.DataFrame(rows))

    old = m[m.queries == "old"].pivot_table(index=["index", "version"], columns="alpha",
                                             values=["recall10", "qps"])
    old.columns = [f"{'recall' if a == 'recall10' else 'QPS'} @alpha={b:g}" for a, b in old.columns]
    old = old.reset_index()
    for c in old.columns[2:]:
        old[c] = old[c].map(lambda v, c=c: f"{v:.3f}" if c.startswith("recall") else f"{v:,.0f}")
    write_table(f"part4_drift_oldq{sfx}", old)


if __name__ == "__main__":
    main()
