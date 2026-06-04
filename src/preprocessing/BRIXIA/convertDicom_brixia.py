"""
Step 1 of BRIXIA preprocessing — DICOM to image conversion.

Reads all DICOM files from data/BRIXIA/dicom_clean/, applies photometric
correction (MONOCHROME1 inversion), and saves each image as a TIFF under
data/BRIXIA/images/ using a multiprocessing pool.

Expected input layout
---------------------
data/BRIXIA/
    dicom_clean/                 ← cleaned DICOM files
    metadata_global_v2.csv       ← official BRIXIA metadata (separator: ';')
                                   must contain columns: Filename, PhotometricInterpretation

Output
------
data/BRIXIA/images/   ← converted TIFF images (one per DICOM)

Usage
-----
    python src/preprocessing/BRIXIA/convertDicom_brixia.py
    python src/preprocessing/BRIXIA/convertDicom_brixia.py \
        --data_dir data/BRIXIA --workers 8
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pydicom as dicom
from PIL import Image
from tqdm import tqdm


def convert_single(filename: str, meta: pd.DataFrame, src_dir: str, dst_dir: str) -> None:
    """Convert one DICOM file to a normalised 16-bit TIFF."""
    dcm = dicom.dcmread(os.path.join(src_dir, filename))
    img_array = dcm.pixel_array.astype(np.float32)
    min_val, max_val = img_array.min(), img_array.max()

    # Invert MONOCHROME1 so that bright = high tissue intensity
    if meta.loc[filename, "PhotometricInterpretation"] == "MONOCHROME1":
        img_array = np.interp(img_array, (min_val, max_val), (max_val, min_val))
        min_val, max_val = img_array.min(), img_array.max()

    # Normalise to [0, 65535]
    if max_val > min_val:
        img_array = (img_array - min_val) / (max_val - min_val) * 65535

    out_name = filename.replace(".dcm", ".tiff")
    Image.fromarray(img_array.astype(np.uint16)).save(os.path.join(dst_dir, out_name))


def convert_brixia(data_dir: str, workers: int) -> None:
    src_dir = os.path.join(data_dir, "dicom_clean")
    dst_dir = os.path.join(data_dir, "images")
    meta_path = os.path.join(data_dir, "metadata_global_v2.csv")

    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"DICOM directory not found: {src_dir}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    os.makedirs(dst_dir, exist_ok=True)

    meta = pd.read_csv(meta_path, sep=";", dtype={"BrixiaScore": str}, index_col="Filename")
    filenames = sorted(f for f in os.listdir(src_dir) if f.endswith(".dcm"))

    print(f"Converting {len(filenames)} DICOM files with {workers} workers...")

    worker_fn = partial(convert_single, meta=meta, src_dir=src_dir, dst_dir=dst_dir)
    with Pool(processes=workers) as pool:
        list(tqdm(pool.imap_unordered(worker_fn, filenames), total=len(filenames)))

    print(f"Done. Images saved to {dst_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert BRIXIA DICOM files to TIFF images."
    )
    parser.add_argument(
        "--data_dir",
        default="data/BRIXIA",
        help="Root directory of the BRIXIA dataset (default: data/BRIXIA).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4).",
    )
    args = parser.parse_args()
    convert_brixia(args.data_dir, args.workers)
