#!/usr/bin/env bash
# Train MDMT — Multi-Dataset Multi-Task model (proposed method).
#
# Trains a shared CNN backbone jointly on AIforCOVID (τ₁, severity prognosis)
# and BRIXIA (τ₂, severity score assessment) using the indicator-function loss.
#
# Usage
# -----
#   bash scripts/train_mdmt.sh [MODEL] [CFG] [STRUCTURE] [RELEASE] [ID_EXP] [-c]
#
# Positional arguments (all optional, with defaults shown)
#   MODEL      CNN backbone name             default: resnet18
#   CFG        Path to YAML config           default: configs/5/multi/parallel_config_multitask_cv5.yaml
#   STRUCTURE  Brixia score representation for τ₂
#              brixia_Global — global summed score (4 classes)
#              brixia_Lung   — per-lung score
#              regression    — direct regression
#              default: brixia_Global
#   RELEASE    AIforCOVID dataset release (1/2/3)   default: 3
#   ID_EXP     Experiment identifier                 default: 1
#
# Flags
#   -c         Resume from the latest checkpoint
#
# Examples
#   # Default run (resnet18, 5-fold CV, global score, release 3)
#   bash scripts/train_mdmt.sh
#
#   # DenseNet-121, LOCO, release 3
#   bash scripts/train_mdmt.sh \
#       densenet121 \
#       configs/loCo/multi/parallel_config_multitask_loCo18.yaml \
#       brixia_Global 3 my_run
#
#   # Resume a previous run
#   bash scripts/train_mdmt.sh resnet18 configs/5/multi/parallel_config_multitask_cv5.yaml brixia_Global 3 1 -c

set -euo pipefail

MODEL="${1:-resnet18}"
CFG="${2:-configs/5/multi/parallel_config_multitask_cv5.yaml}"
STRUCTURE="${3:-brixia_Global}"
RELEASE="${4:-3}"
ID_EXP="${5:-1}"
CHECKPOINT="${6:-}"   # pass '-c' as 6th argument to resume

echo "Training MDMT (AIforCOVID + BRIXIA)"
echo "  Model     : $MODEL"
echo "  Config    : $CFG"
echo "  Structure : $STRUCTURE"
echo "  Release   : $RELEASE"
echo "  Exp ID    : $ID_EXP"
[[ -n "$CHECKPOINT" ]] && echo "  Mode      : resume from checkpoint"

python src/models/train_MultiObjectiveModel.py \
    --model_name "$MODEL" \
    --cfg_file   "$CFG" \
    --structure  "$STRUCTURE" \
    --release    "$RELEASE" \
    --id_exp     "$ID_EXP" \
    $CHECKPOINT
