"""Search timing: one query at a time, or all queries in one batch.

Each measurement is repeated and the median wall-clock time is used.
"""
import time

import faiss
import numpy as np


def set_threads(n):
    faiss.omp_set_num_threads(n)


def search_single(index, queries, k):
    """Search queries one by one. Returns (seconds, ids)."""
    nq = queries.shape[0]
    ids = np.empty((nq, k), dtype=np.int64)
    t0 = time.perf_counter()
    for i in range(nq):
        _, ids[i:i + 1] = index.search(queries[i:i + 1], k)
    return time.perf_counter() - t0, ids


def search_batch(index, queries, k):
    """Search all queries in one call. Returns (seconds, ids)."""
    t0 = time.perf_counter()
    _, ids = index.search(queries, k)
    return time.perf_counter() - t0, ids


def timed(search_fn, index, queries, k, repeats=3):
    """Run search_fn `repeats` times. Returns (median seconds, all times, ids)."""
    times, ids = [], None
    for _ in range(repeats):
        t, ids = search_fn(index, queries, k)
        times.append(t)
    return float(np.median(times)), times, ids


def index_bytes(index):
    """Size of the serialised index in bytes."""
    return int(faiss.serialize_index(index).nbytes)
