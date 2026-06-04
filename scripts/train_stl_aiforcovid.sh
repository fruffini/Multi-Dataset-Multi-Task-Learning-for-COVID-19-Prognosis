#!/usr/bin/env bash
# Train STL^τ₁ — Single-Task Learning on AIforCOVID (severity prognosis).
#
# Usage
# -----
#   bash scripts/train_stl_aiforcovid.sh [MODEL] [CFG] [RELEASE] [ID_EXP] [-c]
#
# Positional arguments (all optional, with defaults shown)
#   MODEL    CNN backbone name       default: resnet18
#   CFG      Path to YAML config     default: configs/5/morbidity/afc_config_singletask_cv5.yaml
#   RELEASE  Dataset release (1/2/3) default: 3
#   ID_EXP   Experiment identifier   default: 1
#
# Flags
#   -c       Resume from the latest checkpoint
#
# Examples
#   # Default run (resnet18, 5-fold CV, release 3)
#   bash scripts/train_stl_aiforcovid.sh
#
#   # DenseNet-121, LOCO validation, release 3
#   bash scripts/train_stl_aiforcovid.sh \
#       densenet121 \
#       configs/loCo/morbidity/afc_config_singletask_loCo.yaml \
#       3 my_run
#
#   # Resume a previous run
#   bash scripts/train_stl_aiforcovid.sh resnet50 configs/5/morbidity/afc_config_singletask_cv5.yaml 3 1 -c

set -euo pipefail

MODEL="${1:-resnet18}"
CFG="${2:-configs/5/morbidity/afc_config_singletask_cv5.yaml}"
RELEASE="${3:-3}"
ID_EXP="${4:-1}"
CHECKPOINT="${5:-}"   # pass '-c' as 5th argument to resume

echo "Training STL^τ₁ (AIforCOVID)"
echo "  Model   : $MODEL"
echo "  Config  : $CFG"
echo "  Release : $RELEASE"
echo "  Exp ID  : $ID_EXP"
[[ -n "$CHECKPOINT" ]] && echo "  Mode    : resume from checkpoint"

python src/models/train_morbidity_SingleTask.py \
    --model_name "$MODEL" \
    --cfg_file   "$CFG" \
    --release    "$RELEASE" \
    --id_exp     "$ID_EXP" \
    $CHECKPOINT
