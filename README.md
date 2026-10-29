# Benchmarking Approximate Nearest Neighbour Search

CS4.406 Information Retrieval & Extraction, Assignment 3.

Benchmarks exact search, HNSW, IVF-Flat and IVF-PQ (FAISS) on SIFT-1M and
GloVe-100, plus a scaling experiment (Part 3), a drifting-data experiment on
generated data (Part 4) and two checks of how the benchmark could mislead (Q5).

## Layout

```
Benchmark_ANN/
  code/                 this git repository
    ann/                library code
      data.py             dataset loading, GloVe normalisation, query split, exact ground truth
      metrics.py          tie-aware recall@10
      timing.py           single-query / batch timing (median of repeats), index size
      indexes.py          building HNSW / IVF-Flat / IVF-PQ
      drift.py            Part 4 data generator (seeded)
      results.py          per-part CSVs -> results.csv
      sysinfo.py          machine specs and library versions
    experiments/        one script per part (+ *_report.py for figures/tables)
    tests/              unit tests (recall@10, drift generator)
    hypothesis.md       Part 3 and Part 4 hypotheses (committed before the runs)
    run_all.sh          one-command reproduction
  data/                 datasets + cached ground truth (not submitted)
  results/              results.csv, figures/, tables/, logs, sysinfo.json
```

## Setup

Python 3.12. With [uv](https://github.com/astral-sh/uv):

```bash
cd code
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

(or `python -m venv .venv && .venv/bin/pip install -r requirements.txt`).

Download the datasets into `../data/` (or point `ANN_DATA_DIR` elsewhere):

```bash
mkdir -p ../data
curl -L -o ../data/sift-128-euclidean.hdf5 http://ann-benchmarks.com/sift-128-euclidean.hdf5
curl -L -o ../data/glove-100-angular.hdf5  http://ann-benchmarks.com/glove-100-angular.hdf5
```

## Reproduction

```bash
cd code
bash run_all.sh           # full data: every number, figure and table in the report (~2 h)
bash run_all.sh --quick   # first 100K vectors, 1000 test queries, 1 timing repeat (< 15 min)
```

`run_all.sh` runs the unit tests, then Parts 1-4 and the Q5 checks, then
writes the figures/tables and merges all runs into `../results/results.csv`.
`--quick` writes the same outputs with a `_quick` suffix
(`results_quick.csv`, `*_quick.png`, ...), so it never overwrites the full results.
Set `PYTHON=...` to use an interpreter other than `.venv/bin/python`.

Outputs:

| Path | Content |
|---|---|
| `results/results.csv` | one row per benchmark run behind a number in the report |
| `results/figures/` | `part2_recall_qps_{sift,glove}`, `part3_scaling`, `part4_drift` (PNG + PDF) |
| `results/tables/` | the report tables as Markdown and LaTeX |
| `results/sysinfo.json` | CPU, RAM, OS, library versions |

Individual parts can be run on their own, e.g.
`.venv/bin/python -m experiments.part2_benchmark [--quick] [--resume]`.

## Method

- **Distance.** Everything uses L2. GloVe base vectors and queries are
  L2-normalised, so L2 ranking equals the original angular ranking.
- **Query split.** Queries 0-4999 are used for tuning (choosing settings),
  queries 5000-9999 for every reported number.
- **Ground truth.** Recomputed with exact FAISS search (`IndexFlatL2`, k=100)
  for each base set and cached in `../data/cache/`. It agrees with the
  neighbours stored in the files up to ties (Part 1).
- **recall@10** (`ann/metrics.py`) recomputes true distances from the raw
  vectors and counts a returned id as correct if its distance is no larger
  than the true 10th-neighbour distance, so ties at the 10th neighbour are not
  mistakes. Missing (-1) and repeated ids are handled.
- **Timing.** Search uses 1 thread and one query per call; each timing is
  repeated (3x; 5x in Part 4) and the median is used. Index builds use all
  threads. QPS = queries / median time.
- **Operating points.** For a target recall, among all settings whose tune
  recall reaches it, the one with the highest tune QPS is chosen; its test
  recall and QPS are reported.
- **Index size** is the size of the serialised FAISS index.

## Experiments

| Script | What it does |
|---|---|
| `part1_exact.py` | exact search; GT agreement check; single vs batch timing |
| `part2_benchmark.py` | HNSW (M 8/16/32), IVF-Flat (nlist 1024/4096), IVF-PQ (nlist 1024/4096 x m) sweeps on both datasets |
| `part3_scaling.py` | SIFT at 100K/300K/1M: exact, HNSW, IVF-Flat (nlist ~ sqrt(N), 4 sqrt(N)) |
| `part4_drift.py` | index built on 200K SIFT vectors, then 100K vectors shifted by alpha x spread added (alpha 0/0.5/1/2); stale vs rebuilt |
| `q5_checks.py` | timing noise across rounds/sessions; single-query vs batch timing |

The Part 4 generator (`ann/drift.py`) is seeded (`SEED = 2026`): the shift
direction is a fixed random unit vector, so every run produces the same data.
