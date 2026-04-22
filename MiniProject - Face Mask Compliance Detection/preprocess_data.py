"""
preprocess_data.py
==================
Preprocessing pipeline for the Face Mask Compliance Detection System.

Steps:
1. Parse Dataset1 Pascal VOC XML annotations → crop faces → label mapping
2. Ingest Dataset2 pre-cropped face images (simple/complex subfolders)
3. Merge both into Data/processed/{proper_mask, incorrect_mask, no_mask}/
4. Augment to balance all classes
5. Resize all crops to 224×224
6. Split into train (80%) / val (20%) with stratification
7. Save Data/processed/stats.json
"""

import os
import json
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Data"

DATASET1_IMAGES = DATA_DIR / "Dataset1" / "images"
DATASET1_ANNOTATIONS = DATA_DIR / "Dataset1" / "annotations"

DATASET2_DIR = DATA_DIR / "Dataset2"

PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"

IMG_SIZE = (224, 224)

LABEL_MAP_DATASET1 = {
    "with_mask": "proper_mask",
    "without_mask": "no_mask",
    "mask_weared_incorrect": "incorrect_mask",
}

LABEL_MAP_DATASET2 = {
    "with_mask": "proper_mask",
    "without_mask": "no_mask",
    "incorrect_mask": "incorrect_mask",
}

CLASSES = ["proper_mask", "no_mask", "incorrect_mask"]

AUGMENTATION_FACTOR = 4  # multiply minority classes


# ──────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────

def ensure_dirs():
    """Create output directory structure."""
    for split in ["train", "val"]:
        for cls in CLASSES:
            (PROCESSED_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    print("[✓] Output directories created.")


def augment_image(img: Image.Image) -> list[Image.Image]:
    """Return a list of augmented PIL images derived from img."""
    augmented = []

    # Horizontal flip
    augmented.append(img.transpose(Image.FLIP_LEFT_RIGHT))

    # Rotation ±15°
    for angle in [-15, 15]:
        augmented.append(img.rotate(angle))

    # Brightness adjustment
    for factor in [0.7, 1.3]:
        enhancer = ImageEnhance.Brightness(img)
        augmented.append(enhancer.enhance(factor))

    # Zoom ~10% (crop center then resize back)
    w, h = img.size
    margin_w = int(w * 0.1)
    margin_h = int(h * 0.1)
    cropped = img.crop((margin_w, margin_h, w - margin_w, h - margin_h))
    augmented.append(cropped.resize(IMG_SIZE))

    return augmented


def save_image(img: Image.Image, dest_path: Path):
    img = img.resize(IMG_SIZE)
    img.save(dest_path, "JPEG", quality=92)


# ──────────────────────────────────────────────
# DATASET 1 — Pascal VOC XML
# ──────────────────────────────────────────────

def parse_dataset1() -> list[tuple[Image.Image, str]]:
    """
    Parse Dataset1: read annotations, crop faces, return (img, label) pairs.
    """
    samples = []

    if not DATASET1_ANNOTATIONS.exists():
        print(f"[WARN] Dataset1 annotations not found at {DATASET1_ANNOTATIONS}")
        return samples

    ann_files = list(DATASET1_ANNOTATIONS.glob("*.xml"))
    print(f"[Dataset1] Found {len(ann_files)} annotation files.")

    for ann_file in ann_files:
        tree = ET.parse(ann_file)
        root = tree.getroot()

        filename = root.findtext("filename")
        img_path = DATASET1_IMAGES / filename
        if not img_path.exists():
            # Try without extension mismatch
            candidates = list(DATASET1_IMAGES.glob(f"{img_path.stem}.*"))
            if not candidates:
                continue
            img_path = candidates[0]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot open {img_path}: {e}")
            continue

        for obj in root.findall("object"):
            label_raw = obj.findtext("name", "").strip()
            label = LABEL_MAP_DATASET1.get(label_raw)
            if label is None:
                continue

            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.findtext("xmin")))
            ymin = int(float(bndbox.findtext("ymin")))
            xmax = int(float(bndbox.findtext("xmax")))
            ymax = int(float(bndbox.findtext("ymax")))

            # Clamp to image bounds
            w, h = img.size
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)

            if xmax <= xmin or ymax <= ymin:
                continue

            face_crop = img.crop((xmin, ymin, xmax, ymax))
            samples.append((face_crop, label))

    print(f"[Dataset1] Extracted {len(samples)} face crops.")
    return samples


# ──────────────────────────────────────────────
# DATASET 2 — Pre-cropped folder structure
# ──────────────────────────────────────────────

def parse_dataset2() -> list[tuple[Image.Image, str]]:
    """
    Parse Dataset2: traverse class/simple and class/complex subfolders.
    """
    samples = []

    if not DATASET2_DIR.exists():
        print(f"[WARN] Dataset2 not found at {DATASET2_DIR}")
        return samples

    for raw_label, mapped_label in LABEL_MAP_DATASET2.items():
        class_dir = DATASET2_DIR / raw_label
        if not class_dir.exists():
            continue

        for subfolder in ["simple", "complex"]:
            sub_dir = class_dir / subfolder
            if not sub_dir.exists():
                continue

            img_files = list(sub_dir.glob("*.jpg")) + list(sub_dir.glob("*.png")) + \
                        list(sub_dir.glob("*.jpeg"))

            for img_path in img_files:
                try:
                    img = Image.open(img_path).convert("RGB")
                    samples.append((img, mapped_label))
                except Exception as e:
                    print(f"[WARN] Cannot open {img_path}: {e}")

    print(f"[Dataset2] Loaded {len(samples)} face images.")
    return samples


# ──────────────────────────────────────────────
# MERGE, AUGMENT, SPLIT
# ──────────────────────────────────────────────

def balance_and_split(samples: list[tuple[Image.Image, str]]):
    """Balance classes via augmentation, resize, split train/val, save to disk."""
    # Group by class
    class_map: dict[str, list[Image.Image]] = {cls: [] for cls in CLASSES}
    for img, label in samples:
        if label in class_map:
            class_map[label].append(img)

    for cls, imgs in class_map.items():
        print(f"  {cls}: {len(imgs)} images before augmentation")

    # Determine target count (max class × AUGMENTATION_FACTOR)
    max_count = max(len(v) for v in class_map.values())
    target = max_count * AUGMENTATION_FACTOR

    all_samples: list[tuple[Image.Image, str]] = []

    for cls, imgs in class_map.items():
        augmented = list(imgs)
        idx = 0
        while len(augmented) < target:
            extra = augment_image(imgs[idx % len(imgs)])
            augmented.extend(extra)
            idx += 1
        augmented = augmented[:target]
        random.shuffle(augmented)
        for img in augmented:
            all_samples.append((img, cls))

    # Stratified train/val split
    images_list = [s[0] for s in all_samples]
    labels_list = [s[1] for s in all_samples]

    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        images_list, labels_list, test_size=0.2, stratify=labels_list, random_state=42
    )

    # Save
    stats = {"train": {}, "val": {}}

    for split, split_imgs, split_labels in [
        ("train", train_imgs, train_labels),
        ("val", val_imgs, val_labels),
    ]:
        for i, (img, label) in enumerate(zip(split_imgs, split_labels)):
            dest = PROCESSED_DIR / split / label / f"{label}_{i:06d}.jpg"
            save_image(img, dest)
        # Count per class
        for cls in CLASSES:
            count = split_labels.count(cls)
            stats[split][cls] = count
            print(f"  [{split}] {cls}: {count}")

    # Save stats JSON
    stats_path = PROCESSED_DIR / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[✓] Stats saved to {stats_path}")

    return stats


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Face Mask Compliance — Data Preprocessing Pipeline")
    print("=" * 60)

    random.seed(42)
    ensure_dirs()

    print("\n[1/3] Parsing Dataset1 (Pascal VOC XML)…")
    d1_samples = parse_dataset1()

    print("\n[2/3] Parsing Dataset2 (pre-cropped folders)…")
    d2_samples = parse_dataset2()

    all_samples = d1_samples + d2_samples
    print(f"\n[3/3] Total raw samples: {len(all_samples)}")
    print("Balancing, augmenting, resizing, and splitting…\n")

    stats = balance_and_split(all_samples)

    total_train = sum(stats["train"].values())
    total_val = sum(stats["val"].values())

    print("\n" + "=" * 60)
    print(f"  Preprocessing complete!")
    print(f"  Train: {total_train} images | Val: {total_val} images")
    print(f"  Output: {PROCESSED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
