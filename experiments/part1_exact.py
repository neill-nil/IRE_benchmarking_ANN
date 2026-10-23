"""Part 1: exact baseline.

1. Check that our exact ground truth agrees with the neighbours stored in the
   ann-benchmarks files (raw top-10 overlap, and tie-aware recall@10).
2. Time exact search (IndexFlatL2) on the test queries:
   one query at a time on 1 thread, and all queries in one batch on 1 thread
   and on all threads.
"""
import argparse
import json
import os
import time

import faiss
import numpy as np

from ann import data, metrics, results, timing

K = 10


def agreement(ds, stored):
    """Compare our exact top-10 against the file's stored top-10."""
    ours = np.concatenate([ds.gt_tune, ds.gt_test])[:, :K]
    theirs = stored[:, :K]
    queries = np.concatenate([ds.q_tune, ds.q_test])
    overlap = np.array([len(set(a) & set(b)) / K for a, b in zip(ours, theirs)])
    tie_recall = metrics.recall_at_k_per_query(ds.base, queries, theirs, ours, K)
    return {
        "raw_top10_overlap": float(overlap.mean()),
        "queries_with_different_ids": int((overlap < 1).sum()),
        "tie_aware_recall_of_stored": float(tie_recall.mean()),
        "queries_with_real_mismatch": int((tie_recall < 1).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=["sift", "glove"])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    n_base = 100_000 if args.quick else None
    n_single = 200 if args.quick else 500
    all_threads = os.cpu_count()
    rows, summary = [], {}

    for name in args.datasets:
        ds = data.load(name, n_base)
        print(f"[{name}] n={ds.n} d={ds.d}")
        info = {"n_base": ds.n}

        if n_base is None:
            _, _, stored = data.load_raw(name)
            info["agreement"] = agreement(ds, stored)
            print("  agreement:", info["agreement"])
        else:
            info["agreement"] = "skipped: stored neighbours refer to the full dataset"

        t0 = time.perf_counter()
        index = faiss.IndexFlatL2(ds.d)
        index.add(ds.base)
        build_s = time.perf_counter() - t0
        nbytes = timing.index_bytes(index)

        q = ds.q_test
        configs = [
            ("single", 1, q[:n_single], timing.search_single),
            ("batch", 1, q, timing.search_batch),
            ("batch", all_threads, q, timing.search_batch),
        ]
        info["timing"] = []
        for mode, threads, qs, fn in configs:
            timing.set_threads(threads)
            med, times, ids = timing.timed(fn, index, qs, K, args.repeats)
            gt = ds.gt_test[:len(qs)]
            rec = metrics.recall_at_k(ds.base, qs, ids, gt, K)
            qps = len(qs) / med
            print(f"  {mode:6s} threads={threads:2d} nq={len(qs)}: "
                  f"{qps:10.1f} QPS ({1000 / qps:.3f} ms/query), recall={rec:.4f}")
            info["timing"].append({"mode": mode, "threads": threads, "n_queries": len(qs),
                                   "qps": qps, "ms_per_query": 1000 / qps, "recall10": rec})
            rows.append(results.row(
                part="1", dataset=name, n_base=ds.n, index="Exact(FlatL2)",
                build_params={}, search_params={}, query_split="test",
                n_queries=len(qs), mode=mode, threads=threads, repeats=args.repeats,
                recall10=rec, qps=qps, median_s=med, all_s=times,
                build_s=build_s, build_threads=all_threads, index_bytes=nbytes,
                bytes_per_vector=nbytes / ds.n,
            ))
        summary[name] = info

    tag = "part1_quick" if args.quick else "part1"
    print("wrote", results.save_part(tag, rows))
    path = os.path.join(results.RESULTS_DIR, f"{tag}_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
