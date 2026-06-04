"""
Step 2 of BRIXIA preprocessing — lung segmentation and bounding-box extraction.

Runs a pre-trained U-Net (Keras/TensorFlow) on every converted TIFF image,
generates binary lung masks, and produces a bounding-box manifest that the
training DataLoader consumes.

Expected input layout (output of convertDicom_brixia.py)
---------------------------------------------------------
data/BRIXIA/
    images/                      ← TIFF images (from convertDicom_brixia.py)
    metadata_global_v2.csv       ← official BRIXIA metadata (separator: ';')
                                   must contain columns: Filename, BrixiaScore

U-Net weights
-------------
Download the pre-trained segmentation model from the BSNet repository and place it at:
    models/segmentation_brixia/segmentation_brixia-model.h5

Output
------
data/BRIXIA/processed/masks/      ← binary lung masks (.tiff)
data/BRIXIA/processed/box_data_BX.xlsx   ← bounding boxes + Brixia scores

Usage
-----
    python src/preprocessing/BRIXIA/segmentation_BX.py
    python src/preprocessing/BRIXIA/segmentation_BX.py \
        --data_dir data/BRIXIA --model_path models/segmentation_brixia/segmentation_brixia-model.h5
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import skimage.filters
from PIL import Image
from skimage import measure
from tqdm import tqdm


def remove_small_regions(mask: np.ndarray, min_size: float) -> np.ndarray:
    """Zero-out connected components smaller than *min_size* pixels."""
    labels = measure.label(mask)
    for region in measure.regionprops(labels):
        if region.area < min_size:
            mask[labels == region.label] = 0
    return mask


def predict_mask(segmenter, image_array: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Run U-Net on a (512, 512) float image and return a refined boolean mask."""
    import tensorflow as tf

    # Normalise to [0, 1] and preprocess
    rng = image_array.max() - image_array.min()
    if rng > 0:
        img = (image_array - image_array.min()) / rng
    else:
        img = image_array.copy()

    # Clip CLAHE-style and median filter (standard preprocessing for BSNet)
    img = np.clip(img, np.percentile(img, 1), np.percentile(img, 99))
    img = cv2.medianBlur((img * 255).astype(np.uint8), 3).astype(np.float32) / 255.0

    x = img[np.newaxis, :, :, np.newaxis]  # (1, H, W, 1)
    input_tensor = tf.convert_to_tensor(x, dtype=tf.float32)

    raw = segmenter.predict(input_tensor)
    seg = (raw > threshold).squeeze()  # (H, W) bool
    seg = remove_small_regions(seg, 0.02 * seg.size)

    # Smooth and re-threshold to fill gaps in the lung contour
    gauss = skimage.filters.gaussian(seg.astype(float), sigma=9)
    thresh = skimage.filters.threshold_otsu(gauss)
    smoothed = (gauss > thresh).astype(np.uint8)
    original = seg.astype(np.uint8)

    # Fill each lung contour separately (keep at most 2)
    canvas_smooth = np.zeros((*seg.shape, 1), dtype=np.float32)
    canvas_orig = np.zeros((*seg.shape, 1), dtype=np.float32)
    for canvas, src in [(canvas_smooth, smoothed), (canvas_orig, original)]:
        contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, c in enumerate(contours):
            if i >= 2:
                break
            cv2.fillPoly(canvas, [c], (255, 255, 255))

    # OR of smoothed and original contour fills
    final = ((canvas_smooth + canvas_orig) > 0).squeeze()
    return final.astype(bool)


def extract_boxes(mask: np.ndarray):
    """Return (left_box, right_box, full_box) from a binary lung mask."""
    lbl = measure.label(mask)
    props = measure.regionprops(lbl)

    if len(props) >= 2:
        b1, b2 = props[0].bbox, props[1].bbox
        left_box, right_box = (list(b1), list(b2)) if b1[1] < b2[1] else (list(b2), list(b1))
        full_props = measure.regionprops(mask.astype("int64"))
        full_box = list(full_props[0].bbox) if len(full_props) == 1 else [0, 0, *mask.shape]
    else:
        left_box = right_box = None
        full_box = [0, 0, *mask.shape]

    return left_box, right_box, full_box


def run_segmentation(data_dir: str, model_path: str) -> None:
    images_dir = os.path.join(data_dir, "images")
    processed_dir = os.path.join(data_dir, "processed")
    masks_dir = os.path.join(processed_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    meta_path = os.path.join(data_dir, "metadata_global_v2.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    meta = pd.read_csv(meta_path, sep=";", dtype={"BrixiaScore": str})

    from keras.models import load_model
    print(f"Loading U-Net from {model_path} ...")
    segmenter = load_model(model_path)

    image_files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(".tiff"))
    print(f"Segmenting {len(image_files)} images...")

    records = []
    for fname in tqdm(image_files):
        dcm_name = fname.replace(".tiff", ".dcm")
        img_path = os.path.join(images_dir, fname)

        img_array = np.array(Image.open(img_path)).astype(np.float32)
        # Resize to 512×512 for the U-Net (BSNet input size)
        from skimage.transform import resize as sk_resize
        img_512 = sk_resize(img_array, (512, 512), anti_aliasing=True)

        mask = predict_mask(segmenter, img_512)

        # Save mask at original image size
        mask_resized = sk_resize(mask.astype(float), img_array.shape, order=0, anti_aliasing=False)
        mask_pil = Image.fromarray((mask_resized * 255).astype(np.uint8))
        mask_pil.save(os.path.join(masks_dir, fname.replace(".tiff", ".tiff")))

        left_box, right_box, full_box = extract_boxes(mask)

        # Look up Brixia score from metadata
        row = meta[meta["Filename"] == dcm_name]
        score = row["BrixiaScore"].values[0] if len(row) > 0 else None

        records.append(
            {
                "img": dcm_name.replace(".dcm", ""),
                "dx": left_box,
                "sx": right_box,
                "all": full_box,
                "label": score,
                "img_path": img_path,
            }
        )

    bbox_df = pd.DataFrame(records).set_index("img")
    bbox_df = bbox_df.dropna(subset=["label"])
    out_path = os.path.join(processed_dir, "box_data_BX.xlsx")
    bbox_df.to_excel(out_path, index=True, index_label="img")

    print(f"Done. Results saved to:")
    print(f"  Masks          → {masks_dir}/")
    print(f"  Bounding boxes → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment lungs in BRIXIA images and extract bounding boxes."
    )
    parser.add_argument(
        "--data_dir",
        default="data/BRIXIA",
        help="Root directory of the BRIXIA dataset (default: data/BRIXIA).",
    )
    parser.add_argument(
        "--model_path",
        default="models/segmentation_brixia/segmentation_brixia-model.h5",
        help="Path to the pre-trained U-Net weights (.h5).",
    )
    args = parser.parse_args()
    run_segmentation(args.data_dir, args.model_path)
