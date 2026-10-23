"""Dataset loading: SIFT-128 and GloVe-100 from ann-benchmarks HDF5 files.

All data is returned as float32 and searched with Euclidean (L2) distance.
GloVe vectors (base and queries) are L2-normalised, so L2 ranking equals the
angular ranking used by the original file.

Queries are split in half: the first 5,000 are for tuning, the last 5,000 for
reporting. Ground truth is recomputed with exact FAISS search (k=100) and
cached, so that it is in L2 space and valid for any base-subset size.
"""
import os
from dataclasses import dataclass

import faiss
import h5py
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("ANN_DATA_DIR", os.path.join(CODE_DIR, "..", "data"))
CACHE_DIR = os.path.join(DATA_DIR, "cache")

FILES = {
    "sift": "sift-128-euclidean.hdf5",
    "glove": "glove-100-angular.hdf5",
}
N_TUNE = 5000
GT_K = 100


@dataclass
class Dataset:
    name: str
    base: np.ndarray        # (n, d) vectors to index
    q_tune: np.ndarray      # (5000, d) queries for tuning
    q_test: np.ndarray      # (5000, d) queries for reporting
    gt_tune: np.ndarray     # (5000, GT_K) exact neighbour ids
    gt_test: np.ndarray
    gtd_tune: np.ndarray    # (5000, GT_K) exact squared L2 distances
    gtd_test: np.ndarray

    @property
    def n(self):
        return self.base.shape[0]

    @property
    def d(self):
        return self.base.shape[1]


def normalise(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


def load_raw(name, n_base=None):
    """Return (base, queries, stored_neighbors) exactly as in the file,
    except GloVe is normalised. n_base keeps only the first n_base vectors."""
    path = os.path.join(DATA_DIR, FILES[name])
    with h5py.File(path, "r") as f:
        base = f["train"][:n_base] if n_base else f["train"][:]
        queries = f["test"][:]
        stored = f["neighbors"][:]
    base = np.ascontiguousarray(base, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    if name == "glove":
        base, queries = normalise(base), normalise(queries)
    return base, queries, stored


def exact_knn(base, queries, k):
    """Exact k-NN by brute force (batched, all threads). Returns (dist, ids)."""
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(base)
    return index.search(queries, k)


def load(name, n_base=None):
    base, queries, _ = load_raw(name, n_base)
    tag = f"{name}_{n_base or 'full'}"
    os.makedirs(CACHE_DIR, exist_ok=True)
    ids_path = os.path.join(CACHE_DIR, f"gt_{tag}_ids.npy")
    dist_path = os.path.join(CACHE_DIR, f"gt_{tag}_dist.npy")
    if os.path.exists(ids_path) and os.path.exists(dist_path):
        gt, gtd = np.load(ids_path), np.load(dist_path)
    else:
        gtd, gt = exact_knn(base, queries, GT_K)
        np.save(ids_path, gt)
        np.save(dist_path, gtd)
    return Dataset(
        name=name, base=base,
        q_tune=queries[:N_TUNE], q_test=queries[N_TUNE:],
        gt_tune=gt[:N_TUNE], gt_test=gt[N_TUNE:],
        gtd_tune=gtd[:N_TUNE], gtd_test=gtd[N_TUNE:],
    )
