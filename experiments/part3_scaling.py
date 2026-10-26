"""Part 3: scaling on SIFT.

For N in {100K, 300K, 1M} (the first N SIFT vectors, so smaller sets are
nested in larger ones), measure exact search, HNSW and IVF-Flat with the same
protocol as Part 2. IVF-Flat's nlist grows with the data (~sqrt(N), ~4 sqrt(N),
rounded to a power of two). Sweeps stop once tune recall reaches 0.97, since
only the cost of reaching 0.95 is needed.

--quick uses N in {25K, 50K, 100K} and 1000 test queries, 1 repeat.
"""
import argparse
import gc
import os
import time

import faiss

from ann import data, indexes, metrics, results, timing
from experiments.part2_benchmark import K, run_config

SIZES = [100_000, 300_000, 1_000_000]
QUICK_SIZES = [25_000, 50_000, 100_000]
STOP_RECALL = 0.97
N_EXACT_SINGLE = 500


def nlists_for(n):
    """Powers of two nearest to sqrt(n) and 4*sqrt(n)."""
    r = n ** 0.5
    return sorted({min((1 << b for b in range(4, 16)), key=lambda v: abs(v - f * r))
                   for f in (1, 4)})


def grid(n):
    return ([("HNSW", {"M": M, "efConstruction": 200}) for M in (8, 16, 32)]
            + [("IVF-Flat", {"nlist": nl}) for nl in nlists_for(n)])


def exact_rows(ds, repeats, n_single):
    t0 = time.perf_counter()
    index = faiss.IndexFlatL2(ds.d)
    index.add(ds.base)
    build_s = time.perf_counter() - t0
    nbytes = timing.index_bytes(index)
    timing.set_threads(1)
    q = ds.q_test[:n_single]
    med, times, ids = timing.timed(timing.search_single, index, q, K, repeats)
    rec = metrics.recall_at_k(ds.base, q, ids, ds.gt_test[:n_single], K)
    return [results.row(
        part="3", dataset=ds.name, n_base=ds.n, index="Exact(FlatL2)",
        build_params={}, search_params={}, query_split="test", n_queries=len(q),
        mode="single", threads=1, repeats=repeats, recall10=rec, qps=len(q) / med,
        median_s=med, all_s=times, build_s=build_s, build_threads=os.cpu_count(),
        index_bytes=nbytes, bytes_per_vector=nbytes / ds.n,
    )]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    sizes = QUICK_SIZES if args.quick else SIZES
    n_test = 1000 if args.quick else data.N_TUNE
    repeats = 1 if args.quick else 3
    n_single = 200 if args.quick else N_EXACT_SINGLE
    tag = "part3_quick" if args.quick else "part3"
    logf = open(os.path.join(results.RESULTS_DIR, f"{tag}.log"), "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    rows = []
    for n in sizes:
        ds = data.load("sift", n)
        log(f"[sift n={ds.n}] nlists={nlists_for(n)} test queries={n_test} repeats={repeats}")
        rows += exact_rows(ds, repeats, n_single)
        log(f"  exact: {rows[-1]['qps']:.1f} QPS")
        quantizers = indexes.QuantizerCache(ds.base)
        for kind, params in grid(n):
            rows += run_config(ds, kind, params, quantizers, repeats, n_test, log,
                               part="3", stop_recall=STOP_RECALL)
            results.save_part(tag, rows)
        del ds, quantizers
        gc.collect()
    log(f"wrote {results.save_part(tag, rows)}")


if __name__ == "__main__":
    main()
