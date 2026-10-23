"""Part 2 figures and tables from results/parts/part2.csv (+ part1.csv).

Operating points are chosen on the TUNE split only: for each index family and
target recall, among all (build, search) settings whose tune recall reaches the
target, pick the one with the highest tune QPS. The reported recall and QPS are
that setting's TEST-split numbers.

Outputs:
  results/figures/part2_recall_qps_<dataset>.{png,pdf}
  results/tables/part2_qps_at_recall.{md,tex}
  results/tables/part2_cost_at_095.{md,tex}
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ann import results

FAMILIES = ["HNSW", "IVF-Flat", "IVF-PQ"]
COLORS = {"HNSW": "#2a78d6", "IVF-Flat": "#eb6834", "IVF-PQ": "#1baf7a"}
EXACT_COLOR = "#52514e"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TARGETS = [0.90, 0.95]
DATASET_TITLE = {"sift": "SIFT (128-d)", "glove": "GloVe-100 (normalised)"}


def load(quick):
    sfx = "_quick" if quick else ""
    p2 = pd.read_csv(os.path.join(results.PARTS_DIR, f"part2{sfx}.csv"))
    p1 = pd.read_csv(os.path.join(results.PARTS_DIR, f"part1{sfx}.csv"))
    return p2, p1


def paired(p2):
    """One row per (dataset, index, build, search) with tune_* and test_* columns."""
    key = ["dataset", "index", "build_params", "search_params"]
    tune = p2[p2.query_split == "tune"].set_index(key)
    test = p2[p2.query_split == "test"].set_index(key)
    keep = ["recall10", "qps", "build_s", "index_bytes", "bytes_per_vector", "n_base"]
    out = tune[["recall10", "qps"]].add_prefix("tune_").join(test[keep].add_prefix("test_"))
    return out.reset_index()


def frontier(df):
    """Pareto frontier (max QPS for a given recall) over test points."""
    df = df.sort_values(["test_recall10", "test_qps"], ascending=[False, False])
    best, keep = -1.0, []
    for _, r in df.iterrows():
        if r.test_qps > best:
            keep.append(r)
            best = r.test_qps
    return pd.DataFrame(keep).sort_values("test_recall10")


def choose(pp, dataset, family, target):
    """Best setting by tune QPS among those reaching `target` tune recall.
    Returns (row, reached)."""
    c = pp[(pp.dataset == dataset) & (pp["index"] == family)]
    ok = c[c.tune_recall10 >= target]
    if len(ok):
        return ok.loc[ok.tune_qps.idxmax()], True
    return c.loc[c.tune_recall10.idxmax()], False


def fmt_params(r):
    b = json.loads(r.build_params)
    b.pop("efConstruction", None)
    s = json.loads(r.search_params)
    return ", ".join(f"{k}={v}" for k, v in {**b, **s}.items())


def plot(pp, p1, dataset, sfx=""):
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    for fam in FAMILIES:
        c = pp[(pp.dataset == dataset) & (pp["index"] == fam)]
        ax.scatter(c.test_recall10, c.test_qps, s=10, color=COLORS[fam], alpha=0.25,
                   linewidths=0, zorder=2)
        f = frontier(c)
        ax.plot(f.test_recall10, f.test_qps, "-o", color=COLORS[fam], lw=2, ms=4,
                label=fam, zorder=3)
    ex = p1[(p1.dataset == dataset) & (p1["mode"] == "single") & (p1.threads == 1)].iloc[0]
    ax.axhline(ex.qps, color=EXACT_COLOR, lw=1.2, ls="--", zorder=1)
    ax.scatter([ex.recall10], [ex.qps], marker="*", s=120, color=EXACT_COLOR, zorder=4,
               label=f"Exact (FlatL2), {ex.qps:.0f} QPS")
    for t in TARGETS:
        ax.axvline(t, color=GRID, lw=1, zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(left=max(0.0, pp[pp.dataset == dataset].test_recall10.min() - 0.02), right=1.005)
    ax.set_xlabel("recall@10 (test queries)", color=INK)
    ax.set_ylabel("queries per second (1 thread, log scale)", color=INK)
    n = int(pp[pp.dataset == dataset].test_n_base.iloc[0])
    ax.set_title(f"{DATASET_TITLE[dataset]}, n={n:,}: best trade-off per index",
                 fontsize=10, color=INK)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=4)
    fig.tight_layout()
    out = []
    for ext in ("png", "pdf"):
        path = os.path.join(results.FIG_DIR, f"part2_recall_qps_{dataset}{sfx}.{ext}")
        fig.savefig(path)
        out.append(path)
    plt.close(fig)
    return out


def write_table(name, df):
    tdir = os.path.join(results.RESULTS_DIR, "tables")
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, f"{name}.md"), "w") as f:
        f.write(df.to_markdown(index=False) + "\n")
    with open(os.path.join(tdir, f"{name}.tex"), "w") as f:
        f.write(df.to_latex(index=False, escape=True))
    print(f"\n{name}\n" + df.to_markdown(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    p2, p1 = load(args.quick)
    pp = paired(p2)
    os.makedirs(results.FIG_DIR, exist_ok=True)
    datasets = [d for d in ("sift", "glove") if d in set(pp.dataset)]

    sfx = "_quick" if args.quick else ""
    for d in datasets:
        print("wrote", *plot(pp, p1, d, sfx))

    qps_rows, cost_rows = [], []
    for d in datasets:
        ex = p1[(p1.dataset == d) & (p1["mode"] == "single") & (p1.threads == 1)].iloc[0]
        for fam in FAMILIES:
            row = {"dataset": d, "index": fam}
            for t in TARGETS:
                r, ok = choose(pp, d, fam, t)
                if ok:
                    row[f"QPS@{t:.2f}"] = f"{r.test_qps:,.0f}"
                    row[f"recall (test)@{t:.2f}"] = f"{r.test_recall10:.3f}"
                    row[f"setting@{t:.2f}"] = fmt_params(r)
                else:
                    row[f"QPS@{t:.2f}"] = "not reached"
                    row[f"recall (test)@{t:.2f}"] = f"max {r.test_recall10:.3f}"
                    row[f"setting@{t:.2f}"] = fmt_params(r)
            qps_rows.append(row)

            r, ok = choose(pp, d, fam, 0.95)
            cost_rows.append({
                "dataset": d, "index": fam,
                "setting": fmt_params(r) + ("" if ok else " (max recall; 0.95 not reached)"),
                "build time (s)": f"{r.test_build_s:.1f}",
                "index size (MiB)": f"{r.test_index_bytes / 2**20:.1f}",
                "bytes/vector": f"{r.test_bytes_per_vector:.1f}",
            })
        cost_rows.append({
            "dataset": d, "index": "Exact (FlatL2)", "setting": "-",
            "build time (s)": f"{ex.build_s:.1f}",
            "index size (MiB)": f"{ex.index_bytes / 2**20:.1f}",
            "bytes/vector": f"{ex.bytes_per_vector:.1f}",
        })
        qps_rows.append({"dataset": d, "index": "Exact (FlatL2)",
                         "QPS@0.90": f"{ex.qps:,.0f}", "recall (test)@0.90": "1.000",
                         "setting@0.90": "-", "QPS@0.95": f"{ex.qps:,.0f}",
                         "recall (test)@0.95": "1.000", "setting@0.95": "-"})

    write_table(f"part2_qps_at_recall{sfx}", pd.DataFrame(qps_rows))
    write_table(f"part2_cost_at_095{sfx}", pd.DataFrame(cost_rows))


if __name__ == "__main__":
    main()
