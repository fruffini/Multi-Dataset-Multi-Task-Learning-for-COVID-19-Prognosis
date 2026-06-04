#!/usr/bin/env bash
# Preprocess the AIforCOVID dataset (two-step pipeline).
#
# Step 1 — convert DICOMs to TIFF images and write a patient manifest.
# Step 2 — run U-Net lung segmentation and extract bounding boxes.
#
# Prerequisites
# -------------
#   • Raw DICOMs placed under data/AIforCOVID/  (see README for layout)
#   • Pre-trained U-Net at models/segmentation_brixia/trained_model.hdf5
#
# Usage
# -----
#   bash scripts/preprocess_aiforcovid.sh
#   bash scripts/preprocess_aiforcovid.sh --data_dir /path/to/AIforCOVID \
#       --model_path /path/to/unet.hdf5

set -euo pipefail

DATA_DIR="data/AIforCOVID"
MODEL_PATH="models/segmentation_brixia/trained_model.hdf5"

# Parse optional overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir)   DATA_DIR="$2";   shift 2 ;;
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "=== Step 1: DICOM → TIFF conversion ==="
python src/preprocessing/AFC/ClinicalDataCreation_AFC.py \
    --data_dir   "$DATA_DIR" \
    --output_dir "$DATA_DIR/processed"

echo ""
echo "=== Step 2: Lung segmentation and bounding-box extraction ==="
python src/preprocessing/AFC/segmentation_AFC.py \
    --data_dir   "$DATA_DIR" \
    --model_path "$MODEL_PATH"

echo ""
echo "AIforCOVID preprocessing complete."
echo "  Images      : $DATA_DIR/processed/images/"
echo "  Masks       : $DATA_DIR/processed/masks/"
echo "  BBox file   : $DATA_DIR/processed/box_data_AXF123.xlsx"
