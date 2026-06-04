"""
Step 1 of AIforCOVID preprocessing — DICOM to image conversion.

Reads all DICOM files from the three AIforCOVID releases, converts them to
normalised TIFF images, and writes a manifest (saved_patients.csv) that maps
each patient ID to its saved image path.

Expected input layout
---------------------
data/AIforCOVID/
    imgs/           ← release 1 DICOMs
    imgs_r2/        ← release 2 DICOMs
    imgs_r3/        ← release 3 DICOMs
    AIforCOVID.xlsx       ← release 1 clinical metadata
    AIforCOVID_r2.xlsx    ← release 2 clinical metadata
    AIforCOVID_r3.xlsx    ← release 3 clinical metadata

Output
------
data/AIforCOVID/processed/images/   ← converted TIFF images
data/AIforCOVID/processed/saved_patients.csv

Usage
-----
    python src/preprocessing/AFC/ClinicalDataCreation_AFC.py
    python src/preprocessing/AFC/ClinicalDataCreation_AFC.py \
        --data_dir data/AIforCOVID --output_dir data/AIforCOVID/processed
"""

import argparse
import os

import numpy as np
import pandas as pd
import pydicom as dicom
from PIL import Image
from pydicom.errors import InvalidDicomError
from tqdm import tqdm


def convert_dicom(image_path: str, output_dir: str, saved: pd.DataFrame) -> pd.DataFrame:
    """Convert a single DICOM file to a normalised TIFF and record its path."""
    try:
        dcm = dicom.dcmread(image_path)
        img_array = dcm.pixel_array.astype(float)
        min_val, max_val = img_array.min(), img_array.max()

        # Invert MONOCHROME1 images so bright pixels = high intensity tissue
        if dcm.PhotometricInterpretation == "MONOCHROME1":
            img_array = np.interp(img_array, (min_val, max_val), (max_val, min_val))
            min_val, max_val = img_array.min(), img_array.max()

        # Collapse any accidental colour channel
        if img_array.ndim > 2:
            img_array = img_array.mean(axis=2)

        # Normalise to [0, 65535] (16-bit TIFF preserves dynamic range)
        if max_val > min_val:
            img_array = (img_array - min_val) / (max_val - min_val) * 65535
        img_pil = Image.fromarray(img_array.astype(np.uint16))

        dest_dir = os.path.join(output_dir, "images")
        os.makedirs(dest_dir, exist_ok=True)

        patient_id = os.path.basename(image_path).replace(".dcm", "")
        save_path = os.path.join(dest_dir, patient_id + ".tiff")
        img_pil.save(save_path)

        saved.loc[len(saved)] = [patient_id, save_path, image_path]

    except (AttributeError, RuntimeError, InvalidDicomError, ValueError) as err:
        print(f"  Skipping {image_path}: {err}")

    return saved


def preprocess_aiforcovid(data_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Collect all DICOM paths from the three releases
    release_dirs = [
        os.path.join(data_dir, "imgs"),
        os.path.join(data_dir, "imgs_r2"),
        os.path.join(data_dir, "imgs_r3"),
    ]
    all_paths = sorted(
        os.path.join(d, f)
        for d in release_dirs
        if os.path.isdir(d)
        for f in os.listdir(d)
        if f.endswith(".dcm")
    )

    if not all_paths:
        raise FileNotFoundError(
            f"No .dcm files found under {data_dir}. "
            "Check that imgs/, imgs_r2/, imgs_r3/ subdirectories are present."
        )

    print(f"Converting {len(all_paths)} DICOM files...")
    saved = pd.DataFrame(columns=["ID", "Path", "OriginalPath"])
    for path in tqdm(all_paths):
        saved = convert_dicom(path, output_dir, saved)

    manifest_path = os.path.join(output_dir, "saved_patients.csv")
    saved.to_csv(manifest_path, index=False)
    print(f"Done. Converted {len(saved)} images.")
    print(f"  Images  → {os.path.join(output_dir, 'images')}/")
    print(f"  Manifest → {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert AIforCOVID DICOMs to TIFF images."
    )
    parser.add_argument(
        "--data_dir",
        default="data/AIforCOVID",
        help="Root directory of the raw AIforCOVID dataset (default: data/AIforCOVID).",
    )
    parser.add_argument(
        "--output_dir",
        default="data/AIforCOVID/processed",
        help="Directory where converted images and the manifest are saved.",
    )
    args = parser.parse_args()
    preprocess_aiforcovid(args.data_dir, args.output_dir)
