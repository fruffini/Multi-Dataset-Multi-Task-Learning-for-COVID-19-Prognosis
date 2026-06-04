#!/usr/bin/env bash
# Aggregate fold-level results into summary tables.
#
# Calls Create_Final_Report.py for the three main experiment types:
#   - MDMT (parallel structure)
#   - STL^τ₁ (morbidity baseline)
#   - STL^τ₂ (severity baseline)
#
# Usage
# -----
#   bash scripts/aggregate_results.sh [RELEASE] [ROOT_FLAG]
#
#   RELEASE    AIforCOVID dataset release (1/2/3)   default: 3
#   ROOT_FLAG  Where results are stored: data_root | data_external
#              default: data_root
#
# Examples
#   bash scripts/aggregate_results.sh
#   bash scripts/aggregate_results.sh 3 data_external

set -euo pipefail

RELEASE="${1:-3}"
ROOT_FLAG="${2:-data_root}"

echo "=== Aggregating MDMT results (release $RELEASE) ==="
for STRUCTURE in brixia_Global brixia_Lung regression; do
    python src/postprocessing/Create_Final_Report.py \
        --modality  multi \
        --name_exp  "MULTI_${RELEASE}release_${STRUCTURE}" \
        --structure parallel \
        --root      "$ROOT_FLAG"
done

echo ""
echo "=== Aggregating STL^τ₁ (morbidity baseline) ==="
python src/postprocessing/Create_Final_Report.py \
    --modality  morbidity \
    --name_exp  "BASELINE_${RELEASE}release" \
    --structure none \
    --root      "$ROOT_FLAG"

echo ""
echo "=== Statistical analysis ==="
python src/postprocessing/statistical_analysis/compute_statistical_scores.py

echo ""
echo "Done. Summary tables written to reports/."
