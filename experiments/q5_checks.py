"""Report Q5: two ways our benchmark could mislead, tested.

(a) Timing noise. On SIFT, the recall-0.95 settings of HNSW and IVF-Flat
    (chosen on the tune split in Part 2) and exact search are timed one query
    at a time in several rounds, interleaved, to measure the spread within a
    session. These are compared with the same settings measured in earlier
    sessions (Parts 1-3) to see the spread between sessions.
(b) Single-query vs batch timing. On both datasets, the same settings are
    timed with all test queries in one batch, on 1 thread and on all threads,
    to see whether the ranking of the indexes changes.
"""
import argparse
import json
import os
import time

import faiss
import pandas as pd

from ann import data, indexes, metrics, results, timing
from experiments.part2_report import choose, load as load_part2, paired

K = 10
N_EXACT_SINGLE = 500


def settings_095(name, quick):
    """Recall-0.95 operating points from Part 2 (tune split)."""
    pp = paired(load_part2(quick)[0])
    out = {}
    for kind in ("HNSW", "IVF-Flat"):
        r, ok = choose(pp, name, kind, 0.95)
        assert ok, f"{kind} did not reach 0.95 on {name}"
        out[kind] = (json.loads(r.build_params), json.loads(r.search_params))
    return out


def build_all(ds, settings):
    timing.set_threads(os.cpu_count())
    built = {}
    t0 = time.perf_counter()
    flat = faiss.IndexFlatL2(ds.d)
    flat.add(ds.base)
    built["Exact(FlatL2)"] = (flat, {}, {}, time.perf_counter() - t0)
    quantizers = indexes.QuantizerCache(ds.base)
    for kind, (bp, sp) in settings.items():
        index, build_s = indexes.build(kind, bp, ds.base, quantizers)
        indexes.set_search_param(index, kind, list(sp.values())[0])
        built[kind] = (index, bp, sp, build_s)
    return built


def row(ds, kind, bp, sp, build_s, mode, threads, q, med, times, ids, gt, note, repeats):
    return results.row(
        part="Q5", dataset=ds.name, n_base=ds.n, index=kind, build_params=bp,
        search_params=sp, query_split="test", n_queries=len(q), mode=mode,
        threads=threads, repeats=repeats,
        recall10=metrics.recall_at_k(ds.base, q, ids, gt, K), qps=len(q) / med,
        median_s=med, all_s=times, build_s=build_s, build_threads=os.cpu_count(),
        note=note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_base = 100_000 if args.quick else None
    n_test = 1000 if args.quick else data.N_TUNE
    rounds = 2 if args.quick else 5
    repeats = 1 if args.quick else 3
    tag = "q5_quick" if args.quick else "q5"
    rows = []

    for name in ("sift", "glove"):
        ds = data.load(name, n_base)
        built = build_all(ds, settings_095(name, args.quick))
        q, gt = ds.q_test[:n_test], ds.gt_test[:n_test]
        print(f"[{name}] n={ds.n}")

        # (a) single-query timing in interleaved rounds (SIFT only)
        n_rounds = rounds if name == "sift" else 1
        for rnd in range(n_rounds):
            for kind, (index, bp, sp, build_s) in built.items():
                qs = q[:N_EXACT_SINGLE] if kind.startswith("Exact") else q
                timing.set_threads(1)
                med, times, ids = timing.timed(timing.search_single, index, qs, K, repeats)
                rows.append(row(ds, kind, bp, sp, build_s, "single", 1, qs, med, times, ids,
                                gt[:len(qs)], f"round={rnd}", repeats))
                print(f"  round {rnd} single {kind:13s} {len(qs) / med:9.1f} QPS")

        # (b) batch timing, 1 thread and all threads
        for threads in (1, os.cpu_count()):
            for kind, (index, bp, sp, build_s) in built.items():
                timing.set_threads(threads)
                med, times, ids = timing.timed(timing.search_batch, index, q, K, repeats)
                rows.append(row(ds, kind, bp, sp, build_s, "batch", threads, q, med, times,
                                ids, gt, "", repeats))
                print(f"  batch threads={threads:2d} {kind:13s} {len(q) / med:9.1f} QPS")
        results.save_part(tag, rows)
        del ds, built
    print("wrote", results.save_part(tag, rows))


if __name__ == "__main__":
    main()
