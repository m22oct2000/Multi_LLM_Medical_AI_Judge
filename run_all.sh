#!/usr/bin/env bash
# Full pipeline for Multi-SLM Medical AI Judge.
#
# Steps:
#   0  tests (no LLM needed)
#   1  build benchmark CSV
#   2  build adrd_questions.json (needed by exp3)
#   3  exp1 - dataset analysis (no LLM)
#   4  exp2 - agreement analysis (needs vLLM on 8001-8004)
#   5  exp3 - rubric sensitivity (needs vLLM)
#   6  exp4 - box plot figures (reads exp2 output, no LLM)
#
# Usage:
#   bash run_all.sh                        # full run
#   bash run_all.sh --no-llm               # steps 0-3 + 6 only
#   bash run_all.sh --tests-only
#   bash run_all.sh --use-source-datasets  # pull from MedQuAD/MedDialog/medical_meadow
#                                          # CSVs must be in benchmark_dataset/source_datasets/

set -euo pipefail

PYTHON=${PYTHON:-python3}
NO_LLM=false
TESTS_ONLY=false
USE_SRC=false

for arg in "$@"; do
  case $arg in
    --no-llm)              NO_LLM=true ;;
    --tests-only)          TESTS_ONLY=true ;;
    --use-source-datasets) USE_SRC=true ;;
  esac
done

if $USE_SRC; then
  export USE_SOURCE_DATASETS=1
  echo "[info] source-dataset mode: will ingest MedQuAD / MedDialog / medical_meadow"
  echo "[info] CSVs must be in benchmark_dataset/source_datasets/"
else
  export USE_SOURCE_DATASETS=0
fi

echo "running Multi-SLM Medical AI Judge pipeline"
echo ""

echo "[step 0] tests"
$PYTHON tests/test_core.py
echo ""

if $TESTS_ONLY; then
  echo "done."
  exit 0
fi

echo "[step 1] build benchmark CSV"
$PYTHON benchmark_dataset/build_agreement_dataset.py
echo ""

echo "[step 2] build adrd_questions.json"
$PYTHON benchmark_dataset/build_adrd_questions.py
echo ""

echo "[step 3] exp1: dataset analysis"
$PYTHON experiments/exp1_dataset_analysis.py
echo ""

if $NO_LLM; then
  echo "--no-llm: skipping exp2 and exp3"
  echo "run without --no-llm to get exp2 results, then: $PYTHON experiments/exp4_boxplot_agreement.py"
  exit 0
fi

echo "[step 4] exp2: agreement analysis"
$PYTHON experiments/exp2_agreement_analysis.py
echo ""

echo "[step 5] exp3: rubric sensitivity"
$PYTHON experiments/exp3_rubric_sensitivity.py
echo ""

echo "[step 6] exp4: box plot figures"
$PYTHON experiments/exp4_boxplot_agreement.py
echo ""

echo "done."
echo "  results  -> results/"
echo "  figures  -> results/figures/"
echo "  csv      -> benchmark_dataset/agreement_benchmark.csv"
