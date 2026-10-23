"""Results storage.

Each experiment writes its own CSV under results/parts/; merge() concatenates
them into results/results.csv, one row per benchmark run.
"""
import json
import os

import pandas as pd

from ann.data import CODE_DIR

RESULTS_DIR = os.environ.get("ANN_RESULTS_DIR", os.path.join(CODE_DIR, "..", "results"))
PARTS_DIR = os.path.join(RESULTS_DIR, "parts")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

COLUMNS = [
    "part", "dataset", "n_base", "index", "build_params", "search_params",
    "query_split", "n_queries", "mode", "threads", "repeats",
    "recall10", "qps", "median_s", "all_s",
    "build_s", "build_threads", "index_bytes", "bytes_per_vector", "note",
]


def row(**kw):
    r = {c: kw.get(c) for c in COLUMNS}
    for c in ("build_params", "search_params", "all_s"):
        if r[c] is not None and not isinstance(r[c], str):
            r[c] = json.dumps(r[c], sort_keys=True)
    extra = set(kw) - set(COLUMNS)
    if extra:
        raise KeyError(f"unknown result columns: {extra}")
    return r


def save_part(name, rows):
    os.makedirs(PARTS_DIR, exist_ok=True)
    path = os.path.join(PARTS_DIR, f"{name}.csv")
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


def merge(quick=False):
    """Full runs and --quick runs are kept apart: *_quick.csv parts only go
    into results_quick.csv."""
    files = sorted(f for f in os.listdir(PARTS_DIR)
                   if f.endswith(".csv") and f.endswith("_quick.csv") == quick)
    df = pd.concat([pd.read_csv(os.path.join(PARTS_DIR, f)) for f in files],
                   ignore_index=True)
    path = os.path.join(RESULTS_DIR, "results_quick.csv" if quick else "results.csv")
    df.to_csv(path, index=False)
    return path
