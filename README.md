# Benchmarking Approximate Nearest Neighbour Search

CS4.406 Information Retrieval & Extraction, Assignment 3.

## Layout

```
Benchmark_ANN/
  code/        this git repository
  data/        HDF5 datasets + cached exact ground truth (not submitted)
  results/     results.csv, sysinfo.json, figures/
  ai_log/      AI prompts/responses and ai_statement.md
```

## Setup

```bash
cd code
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

Download the datasets into `../data/` (or set `ANN_DATA_DIR`):

```bash
mkdir -p ../data
curl -L -o ../data/sift-128-euclidean.hdf5 http://ann-benchmarks.com/sift-128-euclidean.hdf5
curl -L -o ../data/glove-100-angular.hdf5  http://ann-benchmarks.com/glove-100-angular.hdf5
```

## Data handling

- GloVe base vectors and queries are L2-normalised; all search uses L2 distance.
- Queries 0–4999 are used for tuning, 5000–9999 for reported numbers.
- Exact ground truth (k=100) is recomputed with FAISS `IndexFlatL2` and cached
  in `../data/cache/`.

## Reproduction

_TODO: `bash run_all.sh` and `bash run_all.sh --quick`._
