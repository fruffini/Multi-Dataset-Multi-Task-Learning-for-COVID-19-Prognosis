#!/usr/bin/env bash
# Preprocess the BRIXIA dataset (two-step pipeline).
#
# Step 1 — convert DICOMs to TIFF images (parallelised with multiprocessing).
# Step 2 — run U-Net lung segmentation and extract bounding boxes.
#
# Prerequisites
# -------------
#   • Raw DICOMs placed under data/BRIXIA/dicom_clean/  (see README for layout)
#   • metadata_global_v2.csv under data/BRIXIA/
#   • Pre-trained U-Net at models/segmentation_brixia/segmentation_brixia-model.h5
#
# Usage
# -----
#   bash scripts/preprocess_brixia.sh
#   bash scripts/preprocess_brixia.sh --data_dir /path/to/BRIXIA \
#       --model_path /path/to/unet.h5 --workers 8

set -euo pipefail

DATA_DIR="data/BRIXIA"
MODEL_PATH="models/segmentation_brixia/segmentation_brixia-model.h5"
WORKERS=4

# Parse optional overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir)   DATA_DIR="$2";   shift 2 ;;
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --workers)    WORKERS="$2";    shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "=== Step 1: DICOM → TIFF conversion (workers=$WORKERS) ==="
python src/preprocessing/BRIXIA/convertDicom_brixia.py \
    --data_dir "$DATA_DIR" \
    --workers  "$WORKERS"

echo ""
echo "=== Step 2: Lung segmentation and bounding-box extraction ==="
python src/preprocessing/BRIXIA/segmentation_BX.py \
    --data_dir   "$DATA_DIR" \
    --model_path "$MODEL_PATH"

echo ""
echo "BRIXIA preprocessing complete."
echo "  Images      : $DATA_DIR/images/"
echo "  Masks       : $DATA_DIR/processed/masks/"
echo "  BBox file   : $DATA_DIR/processed/box_data_BX.xlsx"
