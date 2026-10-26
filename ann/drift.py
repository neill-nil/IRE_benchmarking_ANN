"""Part 4 data generator: drifting data made by transforming SIFT.

Old data:  the first n_old SIFT vectors (the index is built on these).
New data:  the next n_new SIFT vectors, all moved by the same offset
           alpha * spread * u, where u is a fixed random unit direction and
           spread is the RMS distance of the old vectors from their mean.
Queries:   the 10,000 SIFT queries, moved by the same offset ("new" queries);
           the unshifted queries are kept as "old" queries.

alpha = 0 is the control: the new data comes from the same distribution.
Everything is deterministic given SEED.
"""
from dataclasses import dataclass

import numpy as np

from ann import data

SEED = 2026
ALPHAS = [0.0, 0.5, 1.0, 2.0]


@dataclass
class DriftData:
    alpha: float
    spread: float
    old: np.ndarray        # (n_old, d)
    new: np.ndarray        # (n_new, d), shifted
    q_new: np.ndarray      # (10000, d), shifted queries
    q_old: np.ndarray      # (10000, d), original queries
    base: np.ndarray       # all vectors in id order: old ids first, then new


def direction(d, seed=SEED):
    u = np.random.default_rng(seed).standard_normal(d)
    return (u / np.linalg.norm(u)).astype(np.float32)


def spread_of(x):
    """RMS distance of the vectors from their mean."""
    c = x.mean(axis=0, dtype=np.float64)
    return float(np.sqrt(((x - c) ** 2).sum(axis=1).mean()))


def make(alpha, n_old=200_000, n_new=100_000, seed=SEED):
    base, queries, _ = data.load_raw("sift", n_old + n_new)
    old, new = base[:n_old], base[n_old:]
    spread = spread_of(old)
    offset = (alpha * spread * direction(base.shape[1], seed)).astype(np.float32)
    new = new + offset
    return DriftData(alpha=alpha, spread=spread, old=old, new=new,
                     q_new=queries + offset, q_old=queries,
                     base=np.concatenate([old, new]))
