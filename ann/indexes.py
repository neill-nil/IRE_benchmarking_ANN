"""Building HNSW, IVF-Flat and IVF-PQ indexes with FAISS.

IVF-Flat and IVF-PQ with the same nlist share one trained coarse quantizer
(k-means centroids). Its training time is added to the build time of every
index that uses it, so build times are still per-index totals.
"""
import time

import faiss

SEED = 1234


class QuantizerCache:
    def __init__(self, base):
        self.base = base
        self._cache = {}

    def get(self, nlist):
        """Return (flat quantizer holding nlist centroids, training seconds)."""
        if nlist not in self._cache:
            d = self.base.shape[1]
            t0 = time.perf_counter()
            km = faiss.Kmeans(d, nlist, niter=20, seed=SEED)
            km.train(self.base)
            q = faiss.IndexFlatL2(d)
            q.add(km.centroids)
            self._cache[nlist] = (q, time.perf_counter() - t0)
        return self._cache[nlist]


def build(kind, params, base, quantizers):
    """Build one index. Returns (index, build seconds)."""
    d = base.shape[1]
    if kind == "HNSW":
        t0 = time.perf_counter()
        index = faiss.IndexHNSWFlat(d, params["M"])
        index.hnsw.efConstruction = params["efConstruction"]
        index.add(base)
        return index, time.perf_counter() - t0

    quantizer, q_train_s = quantizers.get(params["nlist"])
    t0 = time.perf_counter()
    if kind == "IVF-Flat":
        index = faiss.IndexIVFFlat(quantizer, d, params["nlist"])
    elif kind == "IVF-PQ":
        index = faiss.IndexIVFPQ(quantizer, d, params["nlist"], params["m"], 8)
        index.cp.seed = SEED
        index.pq.cp.seed = SEED
    else:
        raise ValueError(kind)
    index.train(base)   # quantizer already trained; only trains PQ for IVF-PQ
    index.add(base)   # the QuantizerCache keeps `quantizer` alive
    return index, q_train_s + time.perf_counter() - t0


def set_search_param(index, kind, value):
    if kind == "HNSW":
        index.hnsw.efSearch = value
    else:
        index.nprobe = value


SEARCH_PARAM = {"HNSW": "efSearch", "IVF-Flat": "nprobe", "IVF-PQ": "nprobe"}
