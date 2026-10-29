#!/usr/bin/env bash
# Regenerate every number, figure and table in the report.
#   bash run_all.sh           full data (about 2 hours on the reference laptop)
#   bash run_all.sh --quick   first 100K vectors, fewer queries (under 15 minutes)
# Expects the datasets in ../data (or $ANN_DATA_DIR); see README.md.
set -euo pipefail
cd "$(dirname "$0")"

QUICK=""
if [[ "${1:-}" == "--quick" ]]; then
  QUICK="--quick"
elif [[ -n "${1:-}" ]]; then
  echo "usage: bash run_all.sh [--quick]" >&2
  exit 1
fi

PY="${PYTHON:-.venv/bin/python}"
DATA_DIR="${ANN_DATA_DIR:-../data}"
for f in sift-128-euclidean.hdf5 glove-100-angular.hdf5; do
  if [[ ! -f "$DATA_DIR/$f" ]]; then
    echo "missing $DATA_DIR/$f -- see README.md for the download commands" >&2
    exit 1
  fi
done
mkdir -p ../results/figures ../results/tables ../results/parts

step() { echo; echo "=== $* ==="; }
start=$(date +%s)

step "Machine and library versions";  $PY -m ann.sysinfo ../results/sysinfo.json
step "Unit tests";                    $PY -m pytest -q tests
step "Part 1: exact baseline";        $PY -m experiments.part1_exact $QUICK
step "Part 2: index benchmark";       $PY -m experiments.part2_benchmark $QUICK
step "Part 2: figures and tables";    $PY -m experiments.part2_report $QUICK
step "Part 3: scaling";               $PY -m experiments.part3_scaling $QUICK
step "Part 3: figure and table";      $PY -m experiments.part3_report $QUICK
step "Part 4: drifting data";         $PY -m experiments.part4_drift $QUICK
step "Part 4: figure and tables";     $PY -m experiments.part4_report $QUICK
step "Q5: timing noise, batch vs single"; $PY -m experiments.q5_checks $QUICK
step "Q5: tables";                    $PY -m experiments.q5_report $QUICK
MERGE_QUICK=False; [[ -n "$QUICK" ]] && MERGE_QUICK=True
step "Merge results";                 $PY -c "from ann import results; print(results.merge(quick=$MERGE_QUICK))"

echo; echo "done in $(( ($(date +%s) - start) / 60 )) min"
