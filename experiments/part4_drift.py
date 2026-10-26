"""Part 4: drifting data (see hypothesis.md and ann/drift.py).

For each shift alpha, two versions of each index are measured:
  stale:   trained/built on the old vectors, then the new (shifted) vectors are
           added without retraining;
  rebuilt: trained/built from scratch on all vectors, inserted in a seeded
           random order (so old and new are interleaved).
Search settings are tuned once on the control (alpha = 0, stale, tune queries)
and then kept fixed.

Measured on the test queries (5000-9999), single thread, one query at a time:
  - recall@10 and QPS for "new" queries (shifted, the main measurement);
  - recall@10 and QPS for "old" queries (unshifted, does old data suffer?).
Mechanism measurements (results/part4_mechanism.csv):
  - distance computations per query (how much each search had to scan);
  - IVF: share of new vectors that went into the 1% largest lists;
  - IVF-PQ: relative encoding error of new vectors;
  - HNSW: share of new nodes' level-0 edges that point to old nodes.
"""
import argparse
import gc
import json
import os
import time

import faiss
import numpy as np
import pandas as pd

from ann import data, drift, indexes, metrics, results, timing

K = 10
TARGET = 0.95
PQ_SLACK = 0.005
EF = [16, 24, 32, 48, 64, 96, 128, 192, 256]
NPROBE = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]


def configs(quick):
    nlist = 256 if quick else 1024
    return [("IVF-Flat", {"nlist": nlist}),
            ("IVF-PQ", {"nlist": nlist, "m": 32}),
            ("HNSW", {"M": 16, "efConstruction": 200})]


def build(kind, params, train_x, add_parts):
    """Train on train_x, add each array of add_parts in turn."""
    t0 = time.perf_counter()
    index, _ = indexes.build(kind, params, add_parts[0], indexes.QuantizerCache(train_x))
    for part in add_parts[1:]:
        index.add(part)
    return index, time.perf_counter() - t0


def tune(index, kind, dd, gt_tune):
    """Smallest search value reaching TARGET tune recall (IVF-PQ: within
    PQ_SLACK of its best recall) on the control."""
    timing.set_threads(os.cpu_count())
    values = EF if kind == "HNSW" else NPROBE
    recalls = []
    q = dd.q_new[:data.N_TUNE]
    for v in values:
        indexes.set_search_param(index, kind, v)
        _, ids = timing.search_batch(index, q, K)
        recalls.append(metrics.recall_at_k(dd.base, q, ids, gt_tune, K))
        if kind != "IVF-PQ" and recalls[-1] >= TARGET:
            return v, recalls
    if kind == "IVF-PQ":
        best = max(recalls)
        return next(v for v, r in zip(values, recalls) if r >= best - PQ_SLACK), recalls
    raise RuntimeError(f"{kind} never reached {TARGET} on the control: {recalls}")


def distance_counts(index, kind, q):
    timing.set_threads(os.cpu_count())
    stats = faiss.cvar.hnsw_stats if kind == "HNSW" else faiss.cvar.indexIVF_stats
    stats.reset()
    index.search(q, K)
    return stats.ndis / len(q)


def mechanism(index, kind, dd, perm):
    """Structure-level measurements; perm maps internal id -> original id."""
    n_old = len(dd.old)
    out = {}
    if kind in ("IVF-Flat", "IVF-PQ"):
        nlist = index.nlist
        sizes = np.array([index.invlists.list_size(i) for i in range(nlist)])
        _, assign = index.quantizer.search(dd.new, 1)
        top = np.argsort(sizes)[::-1][:max(1, nlist // 100)]
        out["new_in_top1pct_lists"] = float(np.isin(assign[:, 0], top).mean())
        out["largest_list"] = int(sizes.max())
    if kind == "IVF-PQ":
        sample = dd.new[:: max(1, len(dd.new) // 10000)]
        rec = index.sa_decode(index.sa_encode(sample))
        _, a = index.quantizer.search(sample, 1)
        cent = index.quantizer.reconstruct_n(0, index.nlist)[a[:, 0]]
        err = ((sample - rec) ** 2).sum(1)
        resid = ((sample - cent) ** 2).sum(1)
        out["pq_rel_error_new"] = float(err.mean() / resid.mean())
        out["pq_sq_error_new"] = float(err.mean())
    if kind == "HNSW":
        h = index.hnsw
        nbrs = faiss.vector_to_array(h.neighbors)
        offs = faiss.vector_to_array(h.offsets).astype(np.int64)
        width = h.nb_neighbors(0)
        orig_of = perm                             # internal -> original id
        internal_new = np.where(orig_of >= n_old)[0]
        e = nbrs[offs[internal_new][:, None] + np.arange(width)]
        valid = e >= 0
        out["new_edges_to_old"] = float((orig_of[e[valid]] < n_old).mean())
    return out


def measure(index, kind, value, dd, gt, perm, q_set, n_test, repeats):
    q = (dd.q_new if q_set == "new" else dd.q_old)[data.N_TUNE:data.N_TUNE + n_test]
    indexes.set_search_param(index, kind, value)
    timing.set_threads(1)
    med, times, ids = timing.timed(timing.search_single, index, q, K, repeats)
    ids = np.where(ids >= 0, perm[np.maximum(ids, 0)], -1)
    rec = metrics.recall_at_k(dd.base, q, ids, gt[q_set][:n_test], K)
    return rec, len(q) / med, med, times, distance_counts(index, kind, q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_old, n_new = (50_000, 25_000) if args.quick else (200_000, 100_000)
    n_test = 1000 if args.quick else data.N_TUNE
    repeats = 1 if args.quick else 5   # speed is a key outcome here, so 5 repeats
    tag = "part4_quick" if args.quick else "part4"
    logf = open(os.path.join(results.RESULTS_DIR, f"{tag}.log"), "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    rows, mech, fixed = [], [], {}
    for alpha in drift.ALPHAS:
        dd = drift.make(alpha, n_old, n_new)
        base = dd.base
        timing.set_threads(os.cpu_count())
        gt = {}
        for name, q in (("new", dd.q_new), ("old", dd.q_old)):
            _, ids = data.exact_knn(base, q, K)
            gt[name] = ids[data.N_TUNE:]
            if name == "new":
                gt_tune = ids[:data.N_TUNE]
        log(f"[alpha={alpha}] spread={dd.spread:.1f} n_old={n_old} n_new={n_new}")

        order = np.random.default_rng(drift.SEED).permutation(len(base))
        identity = np.arange(len(base))
        for kind, params in configs(args.quick):
            versions = {
                "stale": (lambda: build(kind, params, dd.old, [dd.old, dd.new]), identity),
                "rebuilt": (lambda: build(kind, params, base[order], [base[order]]), order),
            }
            for version, (make_index, perm) in versions.items():
                timing.set_threads(os.cpu_count())
                index, build_s = make_index()
                if kind not in fixed:   # first pass: alpha = 0, stale
                    fixed[kind], curve = tune(index, kind, dd, gt_tune)
                    log(f"  tuned {kind}: {indexes.SEARCH_PARAM[kind]}={fixed[kind]} "
                        f"(control tune recalls {np.round(curve, 3).tolist()})")
                value = fixed[kind]
                m = mechanism(index, kind, dd, perm)
                nbytes = timing.index_bytes(index)
                for q_set in ("new", "old"):
                    rec, qps, med, times, ndis = measure(index, kind, value, dd, gt, perm,
                                                         q_set, n_test, repeats)
                    bp = {**params, "alpha": alpha, "version": version}
                    rows.append(results.row(
                        part="4", dataset="sift-drift", n_base=len(base), index=kind,
                        build_params=bp, search_params={indexes.SEARCH_PARAM[kind]: value},
                        query_split=f"test-{q_set}", n_queries=n_test, mode="single",
                        threads=1, repeats=repeats, recall10=rec, qps=qps, median_s=med,
                        all_s=times, build_s=build_s, build_threads=os.cpu_count(),
                        index_bytes=nbytes, bytes_per_vector=nbytes / len(base),
                        note=f"n_old={n_old}, n_new={n_new}, seed={drift.SEED}"))
                    mech.append({"alpha": alpha, "index": kind, "version": version,
                                 "queries": q_set, "recall10": rec, "qps": qps,
                                 "dist_per_query": ndis, **m})
                    log(f"  {kind:8s} {version:7s} {q_set} queries: R={rec:.4f} "
                        f"QPS={qps:9.1f} dist/query={ndis:9.0f} {json.dumps(m)}")
                del index
                gc.collect()
        results.save_part(tag, rows)
        pd.DataFrame(mech).to_csv(os.path.join(results.RESULTS_DIR, f"{tag}_mechanism.csv"),
                                  index=False)
        del dd, base
        gc.collect()
    log("done")


if __name__ == "__main__":
    main()
