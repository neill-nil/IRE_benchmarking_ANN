"""Part 2: benchmark HNSW, IVF-Flat and IVF-PQ on SIFT and GloVe.

For every build setting, the search parameter is swept upwards. At each point:
  - tune split (queries 0-4999): recall over all tune queries (computed with
    one batched search -- results are identical to one-at-a-time search) and
    single-thread QPS over the first N_TUNE_TIMED tune queries. These are used
    later to choose operating points.
  - test split (queries 5000-9999): `repeats` single-thread passes, median
    time, gives the reported recall/QPS.
Builds use all threads; timed search always uses 1 thread, one query per call.
A sweep stops once tune recall >= STOP_RECALL, or when recall has stopped
improving (IVF-PQ hits a ceiling set by quantisation error).

Results are saved after every build setting; --resume skips finished ones.
"""
import argparse
import gc
import json
import os

import pandas as pd

from ann import data, indexes, metrics, results, timing

K = 10
STOP_RECALL = 0.99
PLATEAU = 0.005     # stop if recall gained less than this over the last 2 steps
N_TUNE_TIMED = 1000

EF = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
NPROBE = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]


def grid(name, quick):
    # nlist ~ sqrt(n) and 4*sqrt(n); scaled down for the 100K quick subset.
    nlists = (256, 1024) if quick else (1024, 4096)
    ms = (16, 32) if name == "sift" else (10, 25, 50)   # m must divide d
    return (
        [("HNSW", {"M": M, "efConstruction": 200}) for M in (8, 16, 32)]
        + [("IVF-Flat", {"nlist": n}) for n in nlists]
        + [("IVF-PQ", {"nlist": n, "m": m}) for n in nlists for m in ms]
    )


def sweep_values(kind, params):
    if kind == "HNSW":
        return EF
    return [p for p in NPROBE if p <= params["nlist"]]


def run_config(ds, kind, params, quantizers, repeats, n_test, log,
               part="2", stop_recall=STOP_RECALL):
    timing.set_threads(os.cpu_count())
    index, build_s = indexes.build(kind, params, ds.base, quantizers)
    nbytes = timing.index_bytes(index)
    log(f"  built {kind} {params} in {build_s:.1f}s, {nbytes / 2**20:.1f} MiB")

    pname = indexes.SEARCH_PARAM[kind]
    q_test, gt_test = ds.q_test[:n_test], ds.gt_test[:n_test]
    q_tuned = ds.q_tune[:N_TUNE_TIMED]
    rows, history = [], []
    for value in sweep_values(kind, params):
        indexes.set_search_param(index, kind, value)
        common = dict(
            part=part, dataset=ds.name, n_base=ds.n, index=kind, build_params=params,
            search_params={pname: value}, mode="single", threads=1,
            build_s=build_s, build_threads=os.cpu_count(), index_bytes=nbytes,
            bytes_per_vector=nbytes / ds.n,
        )
        timing.set_threads(os.cpu_count())
        _, ids = timing.search_batch(index, ds.q_tune, K)
        r_tune = metrics.recall_at_k(ds.base, ds.q_tune, ids, ds.gt_tune, K)
        timing.set_threads(1)
        t, _ = timing.search_single(index, q_tuned, K)
        rows.append(results.row(**common, query_split="tune", n_queries=len(q_tuned),
                                repeats=1, recall10=r_tune, qps=len(q_tuned) / t,
                                median_s=t, all_s=[t],
                                note=f"recall over all {len(ds.q_tune)} tune queries"))

        med, times, ids = timing.timed(timing.search_single, index, q_test, K, repeats)
        r_test = metrics.recall_at_k(ds.base, q_test, ids, gt_test, K)
        rows.append(results.row(**common, query_split="test", n_queries=len(q_test),
                                repeats=repeats, recall10=r_test, qps=len(q_test) / med,
                                median_s=med, all_s=times))
        log(f"    {pname}={value:5d}  tune R={r_tune:.4f}  test R={r_test:.4f}  "
            f"QPS={len(q_test) / med:9.1f}")

        history.append(r_tune)
        if r_tune >= stop_recall:
            break
        if len(history) >= 3 and history[-1] - history[-3] < PLATEAU:
            break
    del index
    gc.collect()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=["sift", "glove"])
    ap.add_argument("--repeats", type=int, default=None,
                    help="timing repeats (default 3, or 1 with --quick)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    n_base = 100_000 if args.quick else None
    n_test = 1000 if args.quick else data.N_TUNE
    repeats = args.repeats or (1 if args.quick else 3)
    tag = "part2_quick" if args.quick else "part2"
    path = os.path.join(results.PARTS_DIR, f"{tag}.csv")
    log_path = os.path.join(results.RESULTS_DIR, f"{tag}.log")

    rows, done = [], set()
    if args.resume and os.path.exists(path):
        old = pd.read_csv(path)
        rows = old.to_dict("records")
        done = set(zip(old.dataset, old["index"], old.build_params))

    logf = open(log_path, "a" if args.resume else "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    for name in args.datasets:
        ds = data.load(name, n_base)
        log(f"[{name}] n={ds.n} d={ds.d} test queries={n_test} repeats={repeats}")
        quantizers = indexes.QuantizerCache(ds.base)
        for kind, params in grid(name, args.quick):
            key = (name, kind, json.dumps(params, sort_keys=True))
            if key in done:
                log(f"  skip {kind} {params} (done)")
                continue
            rows += run_config(ds, kind, params, quantizers, repeats, n_test, log)
            results.save_part(tag, rows)
        del ds, quantizers
        gc.collect()
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
