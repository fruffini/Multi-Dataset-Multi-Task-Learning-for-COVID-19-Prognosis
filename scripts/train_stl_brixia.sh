#!/usr/bin/env bash
# Train STL^τ₂ — Single-Task Learning on BRIXIA (severity score assessment).
#
# Usage
# -----
#   bash scripts/train_stl_brixia.sh [MODEL] [CFG] [STRUCTURE] [ID_EXP] [-c]
#
# Positional arguments (all optional, with defaults shown)
#   MODEL      CNN backbone name          default: resnet18
#   CFG        Path to YAML config        default: configs/5/severity/bx_config_singletask_cv5.yaml
#   STRUCTURE  Brixia score representation
#              brixia_Global — global summed score (4 classes)
#              brixia_Lung   — per-lung score
#              regression    — direct regression of the raw score
#              default: brixia_Global
#   ID_EXP     Experiment identifier / checkpoint prefix   default: BASELINE
#
# Flags
#   -c         Resume from the latest checkpoint
#
# Examples
#   # Default run (resnet18, 5-fold CV, global score)
#   bash scripts/train_stl_brixia.sh
#
#   # EfficientNet-b0, LOCO, per-lung score
#   bash scripts/train_stl_brixia.sh \
#       efficientnet_b0 \
#       configs/loCo/severity/bx_config_singletask_loCo.yaml \
#       brixia_Lung BASELINE_Lung
#
#   # Resume a previous run
#   bash scripts/train_stl_brixia.sh resnet18 configs/5/severity/bx_config_singletask_cv5.yaml brixia_Global BASELINE -c

set -euo pipefail

MODEL="${1:-resnet18}"
CFG="${2:-configs/5/severity/bx_config_singletask_cv5.yaml}"
STRUCTURE="${3:-brixia_Global}"
ID_EXP="${4:-BASELINE}"
CHECKPOINT="${5:-}"   # pass '-c' as 5th argument to resume

echo "Training STL^τ₂ (BRIXIA)"
echo "  Model     : $MODEL"
echo "  Config    : $CFG"
echo "  Structure : $STRUCTURE"
echo "  Exp ID    : $ID_EXP"
[[ -n "$CHECKPOINT" ]] && echo "  Mode      : resume from checkpoint"

python src/models/train_severity_SingleTask.py \
    --model_name "$MODEL" \
    --cfg_file   "$CFG" \
    --structure  "$STRUCTURE" \
    --id_exp     "$ID_EXP" \
    $CHECKPOINT
