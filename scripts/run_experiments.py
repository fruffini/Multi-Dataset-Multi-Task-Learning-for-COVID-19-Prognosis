#!/usr/bin/env python
"""
Run multiple training experiments locally (single or multi-GPU machine).

Reads a JSON experiment configuration (same format used by launch_bash.py),
spawns one subprocess per backbone, and limits concurrency to --max_jobs
parallel processes. Outputs from each job are written to logs/<job_name>.log.

This script replaces the SLURM-based launch_bash.py workflow.

Usage
-----
    # MDMT, 5-fold CV, all 18 backbones, 1 job at a time
    python scripts/run_experiments.py -k multi -e 5 -id 1

    # STL^τ₁, LOCO validation
    python scripts/run_experiments.py -k morbidity -e L -id 1

    # STL^τ₂, 5-fold CV
    python scripts/run_experiments.py -k severity -e 5 -id BASELINE

    # Run 4 backbones in parallel (requires 4 GPUs or time-multiplexing)
    python scripts/run_experiments.py -k multi -e 5 -id 1 --max_jobs 4

    # Resume from checkpoints
    python scripts/run_experiments.py -k multi -e 5 -id 1 -c
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

CONFIG_SELECTOR = {
    "morbidity": {
        "5":  "configs/bash_experiments/experiment_setups_morbidity_5.json",
        "10": "configs/bash_experiments/experiment_setups_morbidity_10.json",
        "L":  "configs/bash_experiments/experiment_setups_morbidity_loCo.json",
    },
    "severity": {
        "5": "configs/bash_experiments/experiment_setups_severity_5.json",
        "L": "configs/bash_experiments/experiment_setups_severity_loCo.json",
    },
    "multi": {
        "5":  "configs/bash_experiments/experiment_setups_multitask_parallel_5.json",
        "L6": "configs/bash_experiments/experiment_setups_multitask_parallel_loCo_6.json",
    },
}

TRAINING_SCRIPT = {
    "morbidity": "src/models/train_morbidity_SingleTask.py",
    "severity":  "src/models/train_severity_SingleTask.py",
    "multi":     "src/models/train_MultiObjectiveModel.py",
}

# Backbone groups (mirrors the JSON definitions)
BACKBONE_GROUPS = {
    "diff": [
        "densenet121_CXR", "densenet121", "efficientnet_lite0", "efficientnet_b0",
        "efficientnet_es", "efficientnet_es_pruned", "efficientnet_b1_pruned",
        "densenet201", "densenet161", "densenet169",
        "mobilenet_v2", "resnet101", "resnext50_32x4d", "wide_resnet50_2",
    ],
    "easy": [
        "googlenet", "resnet18", "resnet34", "resnet50",
        "resnet50_ChestX-ray14", "resnet50_ChexPert",
        "resnet50_ImageNet_ChestX-ray14", "resnet50_ImageNet_ChexPert",
        "shufflenet_v2_x0_5", "shufflenet_v2_x1_0",
    ],
    "all": [
        "googlenet", "resnet18", "resnet34", "resnet50",
        "resnet50_ChestX-ray14", "resnet50_ChexPert",
        "resnet50_ImageNet_ChestX-ray14", "resnet50_ImageNet_ChexPert",
        "shufflenet_v2_x0_5", "shufflenet_v2_x1_0",
        "densenet121_CXR", "densenet121", "efficientnet_lite0", "efficientnet_b0",
        "efficientnet_es", "efficientnet_es_pruned", "efficientnet_b1_pruned",
        "densenet201", "densenet161", "densenet169",
        "mobilenet_v2", "resnet101", "resnext50_32x4d", "wide_resnet50_2",
    ],
}


def resolve_models(models_field: str) -> list[str]:
    if models_field in BACKBONE_GROUPS:
        return BACKBONE_GROUPS[models_field]
    # Treat as a Python list literal for ad-hoc selections
    try:
        result = eval(models_field)  # noqa: S307 — trusted JSON input
        if isinstance(result, list):
            return result
    except Exception:
        pass
    raise ValueError(f"Unknown models field: {models_field!r}")


def build_command(modality: str, exp: dict, model: str, exp_id: str, checkpoint: bool) -> list[str]:
    script = TRAINING_SCRIPT[modality]
    cfg_path = "configs/{}/{}/{}".format(exp["fold"], modality, exp["config_file"])

    cmd = [sys.executable, str(ROOT / script),
           "--model_name", model,
           "--cfg_file", cfg_path,
           "--id_exp", str(exp_id),
           "--structure", exp["structure"]]

    if modality in ("morbidity", "multi") and exp.get("release"):
        cmd += ["--release", str(exp["release"])]

    if checkpoint:
        cmd.append("-c")

    return cmd


def run_all(modality: str, experiment_key: str, exp_id: str,
            checkpoint: bool, max_jobs: int) -> None:
    config_path = ROOT / CONFIG_SELECTOR[modality][experiment_key]
    with open(config_path) as f:
        experiment_list = json.load(f)

    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    pending = []
    for exp in experiment_list:
        for model in resolve_models(exp["models"]):
            pending.append((exp, model))

    logging.info("Total jobs: %d  |  max_jobs: %d", len(pending), max_jobs)

    running: list[tuple[subprocess.Popen, str]] = []

    def wait_for_slot():
        while len(running) >= max_jobs:
            for proc, tag in list(running):
                ret = proc.poll()
                if ret is not None:
                    running.remove((proc, tag))
                    status = "OK" if ret == 0 else f"FAILED (exit {ret})"
                    logging.info("[%s] %s", tag, status)
            if len(running) >= max_jobs:
                time.sleep(5)

    for exp, model in pending:
        wait_for_slot()

        cmd = build_command(modality, exp, model, exp_id, checkpoint)
        tag = f"{modality}_{model}_{exp['structure']}_{exp.get('release', '')}"
        log_path = logs_dir / f"{tag}.log"

        logging.info("Launching: %s", tag)
        with open(log_path, "w") as log_fh:
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
        running.append((proc, tag))

    # Wait for remaining jobs
    for proc, tag in running:
        ret = proc.wait()
        status = "OK" if ret == 0 else f"FAILED (exit {ret})"
        logging.info("[%s] %s", tag, status)

    logging.info("All jobs finished. Logs in %s/", logs_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run training experiments locally (no SLURM required).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-k", "--modality",
        choices=list(TRAINING_SCRIPT),
        default="morbidity",
        help="Which training mode to run: morbidity | severity | multi.",
    )
    parser.add_argument(
        "-e", "--experiment_key",
        default="5",
        help=(
            "Experiment configuration key (see CONFIG_SELECTOR in this script). "
            "morbidity/multi: '5', '10', 'L', 'L6'. severity: '5', 'L'."
        ),
    )
    parser.add_argument(
        "-id", "--exp_id",
        default="1",
        help="Experiment identifier used as checkpoint/report prefix.",
    )
    parser.add_argument(
        "-c", "--checkpoint",
        action="store_true",
        help="Resume each experiment from its latest checkpoint.",
    )
    parser.add_argument(
        "--max_jobs",
        type=int,
        default=1,
        help=(
            "Maximum number of experiments to run in parallel. "
            "Use 1 (default) for a single GPU; increase for multi-GPU machines."
        ),
    )
    args = parser.parse_args()

    if args.experiment_key not in CONFIG_SELECTOR.get(args.modality, {}):
        valid = list(CONFIG_SELECTOR[args.modality])
        parser.error(f"Invalid -e key {args.experiment_key!r} for modality {args.modality!r}. "
                     f"Valid keys: {valid}")

    run_all(args.modality, args.experiment_key, args.exp_id,
            args.checkpoint, args.max_jobs)


if __name__ == "__main__":
    main()
