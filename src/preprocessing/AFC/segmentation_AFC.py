"""
Step 2 of AIforCOVID preprocessing — lung segmentation and bounding-box extraction.

Runs a pre-trained U-Net (Keras/TensorFlow) on every converted TIFF image,
produces binary lung masks, and saves a bounding-box manifest that the
training DataLoader consumes.

Expected input layout (output of ClinicalDataCreation_AFC.py)
-------------------------------------------------------------
data/AIforCOVID/processed/images/  ← TIFF images
data/AIforCOVID/
    AIforCOVID.xlsx       ← release 1 clinical metadata  (contains Prognosis label)
    AIforCOVID_r2.xlsx    ← release 2 clinical metadata
    AIforCOVID_r3.xlsx    ← release 3 clinical metadata

U-Net weights
-------------
Download the pre-trained segmentation model from the BSNet repository and place it at:
    models/segmentation_brixia/trained_model.hdf5

Output
------
data/AIforCOVID/processed/masks/        ← binary lung masks (.tif)
data/AIforCOVID/processed/box_data_AXF123.xlsx   ← bounding boxes + labels

Usage
-----
    python src/preprocessing/AFC/segmentation_AFC.py
    python src/preprocessing/AFC/segmentation_AFC.py \
        --data_dir data/AIforCOVID --model_path models/segmentation_brixia/trained_model.hdf5
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure, transform
from tqdm import tqdm


def remove_small_regions(mask: np.ndarray, min_size: float) -> np.ndarray:
    """Zero-out connected components smaller than *min_size* pixels."""
    labels = measure.label(mask)
    for region in measure.regionprops(labels):
        if region.area < min_size:
            mask[labels == region.label] = 0
    return mask


def predict_mask(unet, image_array: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Run U-Net on a single (H, W) float image and return a boolean mask."""
    from keras.preprocessing.image import ImageDataGenerator

    im_shape = (256, 256)
    resized = transform.resize(image_array, im_shape, anti_aliasing=True)
    x = resized[np.newaxis, ..., np.newaxis].astype(np.float32)  # (1, H, W, 1)

    gen = ImageDataGenerator(rescale=1.0)
    pred = unet.predict(x)[..., 0].reshape(im_shape)
    pred = pred > threshold
    pred = remove_small_regions(pred, 0.02 * np.prod(im_shape))
    return pred


def extract_boxes(mask: np.ndarray):
    """Return (left_box, right_box, full_box) from a binary lung mask."""
    lbl = measure.label(mask)
    props = measure.regionprops(lbl)

    if len(props) >= 2:
        b1, b2 = props[0].bbox, props[1].bbox
        if b1[1] < b2[1]:
            left_box, right_box = list(b1), list(b2)
        else:
            left_box, right_box = list(b2), list(b1)
        # Full-lung box from the combined binary mask
        full_props = measure.regionprops(mask.astype("int64"))
        full_box = list(full_props[0].bbox) if len(full_props) == 1 else [0, 0, *mask.shape]
    else:
        left_box = right_box = None
        full_box = [0, 0, *mask.shape]

    return left_box, right_box, full_box


def run_segmentation(data_dir: str, model_path: str) -> None:
    processed_dir = os.path.join(data_dir, "processed")
    images_dir = os.path.join(processed_dir, "images")
    masks_dir = os.path.join(processed_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    # Load clinical metadata to map image IDs to prognosis labels
    dfs = []
    for fname in ("AIforCOVID.xlsx", "AIforCOVID_r2.xlsx", "AIforCOVID_r3.xlsx"):
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            dfs.append(pd.read_excel(path))
    if not dfs:
        raise FileNotFoundError(
            f"No AIforCOVID_*.xlsx metadata files found in {data_dir}."
        )
    meta = pd.concat(dfs).set_index("ImageFile")

    # Load U-Net
    from keras.models import load_model
    print(f"Loading U-Net from {model_path} ...")
    unet = load_model(model_path)

    image_files = sorted(
        f for f in os.listdir(images_dir) if f.lower().endswith(".tiff")
    )
    print(f"Segmenting {len(image_files)} images...")

    records = []
    for fname in tqdm(image_files):
        patient_id = fname.replace(".tiff", "")
        img_path = os.path.join(images_dir, fname)

        img_array = np.array(Image.open(img_path)).astype(np.float32)
        # Normalise to [0, 1] for U-Net
        rng = img_array.max() - img_array.min()
        if rng > 0:
            img_array = (img_array - img_array.min()) / rng

        original_shape = img_array.shape

        mask = predict_mask(unet, img_array)
        # Resize mask back to original image dimensions
        mask_full = transform.resize(mask, original_shape, order=0, anti_aliasing=False)

        # Save mask as TIFF
        mask_pil = Image.fromarray((mask_full * 255).astype(np.uint8))
        mask_pil.save(os.path.join(masks_dir, patient_id + ".tif"))

        left_box, right_box, full_box = extract_boxes(mask)

        label = meta.loc[patient_id, "Prognosis"] if patient_id in meta.index else None
        records.append(
            {
                "img": patient_id,
                "dx": left_box,
                "sx": right_box,
                "all": full_box,
                "label": label,
                "img_path": img_path,
            }
        )

    bbox_df = pd.DataFrame(records).set_index("img")
    bbox_df = bbox_df.dropna(subset=["label"])
    out_path = os.path.join(processed_dir, "box_data_AXF123.xlsx")
    bbox_df.to_excel(out_path, index=True, index_label="img")

    print(f"Done. Results saved to:")
    print(f"  Masks       → {masks_dir}/")
    print(f"  Bounding boxes → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment lungs in AIforCOVID images and extract bounding boxes."
    )
    parser.add_argument(
        "--data_dir",
        default="data/AIforCOVID",
        help="Root directory of the raw AIforCOVID dataset (default: data/AIforCOVID).",
    )
    parser.add_argument(
        "--model_path",
        default="models/segmentation_brixia/trained_model.hdf5",
        help="Path to the pre-trained U-Net weights (.hdf5).",
    )
    args = parser.parse_args()
    run_segmentation(args.data_dir, args.model_path)
