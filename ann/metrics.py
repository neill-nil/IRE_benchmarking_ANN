"""Tie-aware recall@k.

A returned id counts as a hit if its true distance to the query is no larger
than the true distance of the k-th exact neighbour. This way, when several
vectors are tied at the k-th distance, returning any of them is not counted
as a mistake. Distances are always recomputed from the raw vectors (never
taken from the index), because e.g. IVF-PQ reports approximate distances.
"""
import numpy as np


def true_distances(base, queries, ids):
    """Exact squared L2 distances between each query and its returned ids.

    ids: (nq, k) int array; -1 marks a missing result and gets distance inf.
    """
    ids = np.asarray(ids)
    valid = ids >= 0
    safe = np.where(valid, ids, 0)
    diff = base[safe].astype(np.float64) - queries[:, None, :].astype(np.float64)
    d = np.einsum("qkd,qkd->qk", diff, diff)
    d[~valid] = np.inf
    return d


def recall_per_query(found_ids, found_dists, kth_dist, k=10, rtol=1e-6):
    """Per-query recall@k.

    found_ids:   (nq, >=k) ids returned by the index (-1 = no result)
    found_dists: (nq, >=k) true distances of those ids (see true_distances)
    kth_dist:    (nq,) true distance of the k-th exact neighbour
    """
    found_ids = np.asarray(found_ids)[:, :k]
    found_dists = np.asarray(found_dists, dtype=np.float64)[:, :k]
    kth_dist = np.asarray(kth_dist, dtype=np.float64)
    thresh = kth_dist * (1 + rtol) + 1e-9
    hit = (found_ids >= 0) & (found_dists <= thresh[:, None])
    # A repeated id may only count once.
    sorted_ids = np.sort(np.where(hit, found_ids, -1), axis=1)
    dup = np.zeros_like(hit)
    dup[:, 1:] = (sorted_ids[:, 1:] == sorted_ids[:, :-1]) & (sorted_ids[:, 1:] >= 0)
    n_hits = (sorted_ids >= 0).sum(axis=1) - dup.sum(axis=1)
    return np.minimum(n_hits, k) / k


def recall_at_k(base, queries, found_ids, gt_ids, k=10):
    """Mean tie-aware recall@k of found_ids against exact neighbours gt_ids."""
    kth = true_distances(base, queries, np.asarray(gt_ids)[:, k - 1:k])[:, 0]
    found_d = true_distances(base, queries, np.asarray(found_ids)[:, :k])
    return float(recall_per_query(found_ids, found_d, kth, k).mean())


def recall_at_k_per_query(base, queries, found_ids, gt_ids, k=10):
    kth = true_distances(base, queries, np.asarray(gt_ids)[:, k - 1:k])[:, 0]
    found_d = true_distances(base, queries, np.asarray(found_ids)[:, :k])
    return recall_per_query(found_ids, found_d, kth, k)
