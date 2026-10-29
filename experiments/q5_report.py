"""Report Q5 tables from results/parts/q5.csv (+ earlier parts).

Outputs:
  results/tables/q5_noise.{md,tex}   single-query QPS of the same settings:
                                     rounds within this session, and earlier
                                     sessions (Parts 1-3)
  results/tables/q5_batch.{md,tex}   single-query vs batch QPS and ranking
Also prints the within-run spread of the timing repeats over all Part 2 runs.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from ann import results
from experiments.part2_report import write_table


def part(name, quick):
    path = os.path.join(results.PARTS_DIR, f"{name}{'_quick' if quick else ''}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def same_setting(df, r):
    return df[(df.dataset == r.dataset) & (df.n_base == r.n_base) & (df["index"] == r["index"])
              & (df.build_params == r.build_params) & (df.search_params == r.search_params)
              & (df.query_split == "test") & (df["mode"] == "single") & (df.threads == 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    sfx = "_quick" if args.quick else ""
    q5 = part("q5", args.quick)

    # (a) noise
    single = q5[(q5["mode"] == "single") & (q5.dataset == "sift")]
    earlier = {"Part 1 run": part("part1", args.quick), "Part 2 run": part("part2", args.quick),
               "Part 3 run": part("part3", args.quick)}
    rows = []
    for kind, g in single.groupby("index", sort=False):
        r0 = g.iloc[0]
        cur = g.qps.to_numpy()
        row = {"index": kind, "setting": ", ".join(f"{k}={v}" for k, v in
                                                   {**json.loads(r0.build_params),
                                                    **json.loads(r0.search_params)}.items()
                                                   if k != "efConstruction") or "-"}
        all_q = list(cur)
        for label, df in earlier.items():
            m = same_setting(df, r0) if df is not None else pd.DataFrame()
            row[label] = f"{m.qps.iloc[0]:,.0f}" if len(m) else "-"
            all_q += list(m.qps)
        row[f"this session: {len(cur)} rounds, median (min-max)"] = (
            f"{np.median(cur):,.0f} ({cur.min():,.0f}-{cur.max():,.0f})")
        row["spread within session"] = f"{cur.max() / cur.min():.2f}x"
        row["spread across all runs"] = f"{max(all_q) / min(all_q):.2f}x"
        rows.append(row)
    write_table(f"q5_noise{sfx}", pd.DataFrame(rows))

    p2 = part("part2", args.quick)
    t = p2[(p2.query_split == "test")]
    spread = t.all_s.map(lambda s: (lambda a: (max(a) - min(a)) / np.median(a))(json.loads(s)))
    print(f"\nPart 2 within-run repeat spread ((max-min)/median over {len(t)} runs): "
          f"median {spread.median():.1%}, 90th percentile {spread.quantile(0.9):.1%}, "
          f"max {spread.max():.1%}")

    # (b) batch vs single
    rows = []
    for ds, g in q5.groupby("dataset", sort=False):
        single_med = g[g["mode"] == "single"].groupby("index", sort=False).qps.median()
        b1 = g[(g["mode"] == "batch") & (g.threads == 1)].set_index("index").qps
        bn = g[(g["mode"] == "batch") & (g.threads > 1)]
        nthr = int(bn.threads.iloc[0])
        bn = bn.set_index("index").qps
        rec = g[g["mode"] == "batch"].groupby("index").recall10.first()
        for kind in single_med.index:
            rows.append({
                "dataset": ds, "index": kind, "recall (test)": f"{rec[kind]:.3f}",
                "single, 1 thread": f"{single_med[kind]:,.0f}",
                "batch, 1 thread": f"{b1[kind]:,.0f}",
                f"batch, {nthr} threads": f"{bn[kind]:,.0f}",
                "batch speed-up (1 thr)": f"{b1[kind] / single_med[kind]:.1f}x",
                f"batch speed-up ({nthr} thr)": f"{bn[kind] / single_med[kind]:.1f}x",
            })
        for label, s in (("single", single_med), ("batch-1", b1), (f"batch-{nthr}", bn)):
            print(f"{ds} ranking ({label}): " + " > ".join(s.sort_values(ascending=False).index))
    write_table(f"q5_batch{sfx}", pd.DataFrame(rows))


if __name__ == "__main__":
    main()
