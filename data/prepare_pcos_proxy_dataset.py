"""Dataset preparation pipeline for the PCOS pelvic ultrasound proxy dataset.

This script does the following:
1. Locates a raw pelvic-ultrasound dataset under data/raw/pcos_ultrasound/
2. Maps labels:
   - infected -> 1 (endometriosis suspected)
   - not_infected -> 0 (normal)
3. Applies preprocessing and augmentation
4. Creates a stratified 70/15/15 split
5. Generates a simulated symptom feature table aligned with the image labels
6. Saves summary metadata for class balance and data split tracking

Important note:
This is a proxy dataset, not a verified endometriosis dataset. The project
uses the PCOS pelvic-ultrasound dataset as a same-organ-system, same-modality
reference to maintain clinical relevance without making unsupported claims.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


DATA_ROOT = Path(__file__).resolve().parent
RAW_DIR = DATA_ROOT / "raw" / "pcos_ultrasound"
PROCESSED_DIR = DATA_ROOT / "processed"
SYMPTOM_DIR = DATA_ROOT / "symptom_features"
MANIFEST_PATH = DATA_ROOT / "dataset_manifest.csv"
SUMMARY_PATH = DATA_ROOT / "class_balance_summary.json"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LABEL_MAP = {"infected": 1, "not_infected": 0}


def discover_images() -> List[Dict[str, str | int]]:
    """Collect image files from the expected raw dataset structure."""
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Expected structure: data/raw/pcos_ultrasound/{infected,not_infected}/"
        )

    samples: List[Dict[str, str | int]] = []

    for folder_name, label in LABEL_MAP.items():
        folder_path = RAW_DIR / folder_name
        if not folder_path.exists():
            continue

        for image_path in sorted(folder_path.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(
                    {
                        "path": str(image_path),
                        "label": label,
                        "source_label": folder_name,
                    }
                )

    if not samples:
        raise FileNotFoundError(
            "No images found in the raw dataset folders. Please add images to "
            "data/raw/pcos_ultrasound/infected and /not_infected."
        )

    return samples


def stratified_split(samples: List[Dict[str, str | int]]) -> Dict[str, List[Dict[str, str | int]]]:
    """Create a stratified 70/15/15 split by label."""
    by_label: Dict[int, List[Dict[str, str | int]]] = {0: [], 1: []}
    for sample in samples:
        by_label[int(sample["label"])].append(sample)

    for label in by_label:
        random.Random(42).shuffle(by_label[label])

    splits: Dict[str, List[Dict[str, str | int]]] = {"train": [], "val": [], "test": []}

    for label, items in by_label.items():
        total = len(items)
        train_count = int(total * 0.70)
        val_count = int(total * 0.15)
        test_count = total - train_count - val_count

        if test_count < 1:
            test_count = 1
            val_count = max(0, total - train_count - test_count)
            if val_count == 0 and train_count > 0:
                train_count = total - val_count - test_count

        splits["train"].extend(items[:train_count])
        splits["val"].extend(items[train_count : train_count + val_count])
        splits["test"].extend(items[train_count + val_count : train_count + val_count + test_count])

    return splits


def resize_and_normalize(image: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize image to the model input size and normalize to [0, 1]."""
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def augment_image(image: np.ndarray) -> np.ndarray:
    """Light augmentation suitable for training images only."""
    augmented = image.copy()

    if random.random() < 0.5:
        augmented = cv2.flip(augmented, 1)
    if random.random() < 0.5:
        augmented = cv2.flip(augmented, 0)

    angle = random.uniform(-20, 20)
    height, width = augmented.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    augmented = cv2.warpAffine(augmented, matrix, (width, height), flags=cv2.INTER_LINEAR)

    brightness_factor = random.uniform(0.7, 1.3)
    augmented = cv2.convertScaleAbs(augmented, alpha=brightness_factor, beta=0)

    return augmented


def write_csv_manifest(split_map: Dict[str, List[Dict[str, str | int]]]) -> None:
    """Save a dataset manifest with split membership for each file."""
    fieldnames = ["file_path", "source_label", "label", "split"]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for split_name, samples in split_map.items():
            for sample in samples:
                writer.writerow(
                    {
                        "file_path": sample["path"],
                        "source_label": sample["source_label"],
                        "label": sample["label"],
                        "split": split_name,
                    }
                )


def save_processed_split(split_name: str, samples: List[Dict[str, str | int]], augment: bool = False) -> None:
    """Resize, normalize, and save processed images to the output folders."""
    output_dir = PROCESSED_DIR / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for label in [0, 1]:
        (output_dir / str(label)).mkdir(parents=True, exist_ok=True)

    for sample in samples:
        label = int(sample["label"])
        image = cv2.imread(sample["path"])
        if image is None:
            continue

        processed = resize_and_normalize(image)
        output_path = output_dir / str(label) / Path(sample["path"]).name
        cv2.imwrite(str(output_path), cv2.cvtColor((processed * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        if augment and split_name == "train":
            augmented = augment_image(image)
            augmented_processed = resize_and_normalize(augmented)
            augmented_path = output_dir / str(label) / f"aug_{Path(sample['path']).stem}.png"
            cv2.imwrite(
                str(augmented_path),
                cv2.cvtColor((augmented_processed * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            )


def summarize_class_balance(split_map: Dict[str, List[Dict[str, str | int]]]) -> Dict[str, Dict[str, int]]:
    """Report counts by split and label for balancing and reporting purposes."""
    summary: Dict[str, Dict[str, int]] = {}
    for split_name, samples in split_map.items():
        counts = {"total": len(samples), "label_0": 0, "label_1": 0}
        for sample in samples:
            if int(sample["label"]) == 0:
                counts["label_0"] += 1
            else:
                counts["label_1"] += 1
        summary[split_name] = counts

    with SUMMARY_PATH.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)

    return summary


def generate_simulated_symptom_dataset(split_map: Dict[str, List[Dict[str, str | int]]]) -> str:
    """Generate a noisy, overlapping symptom dataset aligned to image labels.

    This version preserves the project's proxy-data framing while ensuring the
    symptom model must learn a pattern rather than simply reading the label.
    Mild symptoms occur in some suspected cases and several symptom features can
    appear in some normal cases, creating a realistic overlap between classes.
    """
    SYMPTOM_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SYMPTOM_DIR / "simulated_symptom_dataset.csv"

    fieldnames = [
        "sample_id",
        "pain_level",
        "cycle_irregularity",
        "pain_during_intercourse",
        "pelvic_pressure",
        "heavy_bleeding",
        "infertility_history",
        "fatigue",
        "label",
        "is_simulated",
    ]

    rows = []
    rng = random.Random(42)
    sample_id = 1

    for split_name, samples in split_map.items():
        for sample in samples:
            label = int(sample["label"])

            if label == 1:
                pain_level = int(np.clip(rng.gauss(7.0, 1.8), 1, 10))
                cycle_irregularity = int(rng.random() < 0.72)
                pain_during_intercourse = int(np.clip(rng.gauss(6.5, 2.1), 1, 10))
                pelvic_pressure = int(np.clip(rng.gauss(6.0, 2.0), 1, 10))
                heavy_bleeding = int(rng.random() < 0.55)
                infertility_history = int(rng.random() < 0.35)
                fatigue = int(np.clip(rng.gauss(5.8, 2.2), 1, 10))

                if rng.random() < 0.18:
                    pain_level = int(np.clip(rng.gauss(4.2, 1.7), 1, 10))
                    pain_during_intercourse = int(np.clip(rng.gauss(4.0, 1.8), 1, 10))
                    pelvic_pressure = int(np.clip(rng.gauss(3.8, 1.9), 1, 10))
                    fatigue = int(np.clip(rng.gauss(3.5, 2.0), 1, 10))
            else:
                pain_level = int(np.clip(rng.gauss(3.0, 2.1), 1, 10))
                cycle_irregularity = int(rng.random() < 0.18)
                pain_during_intercourse = int(np.clip(rng.gauss(3.0, 2.0), 1, 10))
                pelvic_pressure = int(np.clip(rng.gauss(3.2, 2.1), 1, 10))
                heavy_bleeding = int(rng.random() < 0.28)
                infertility_history = int(rng.random() < 0.22)
                fatigue = int(np.clip(rng.gauss(3.0, 2.2), 1, 10))

                if rng.random() < 0.16:
                    pain_level = int(np.clip(rng.gauss(6.4, 1.8), 1, 10))
                    cycle_irregularity = int(rng.random() < 0.55)
                    pain_during_intercourse = int(np.clip(rng.gauss(6.0, 2.1), 1, 10))
                    pelvic_pressure = int(np.clip(rng.gauss(5.5, 2.0), 1, 10))
                    heavy_bleeding = int(rng.random() < 0.6)
                    infertility_history = int(rng.random() < 0.4)
                    fatigue = int(np.clip(rng.gauss(5.7, 2.2), 1, 10))

            rows.append(
                {
                    "sample_id": sample_id,
                    "pain_level": pain_level,
                    "cycle_irregularity": cycle_irregularity,
                    "pain_during_intercourse": pain_during_intercourse,
                    "pelvic_pressure": pelvic_pressure,
                    "heavy_bleeding": heavy_bleeding,
                    "infertility_history": infertility_history,
                    "fatigue": fatigue,
                    "label": label,
                    "is_simulated": True,
                }
            )
            sample_id += 1

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return str(output_path)


if __name__ == "__main__":
    try:
        samples = discover_images()
        split_map = stratified_split(samples)
        write_csv_manifest(split_map)
        for split_name, split_samples in split_map.items():
            save_processed_split(split_name, split_samples, augment=(split_name == "train"))
        summary = summarize_class_balance(split_map)
        symptom_path = generate_simulated_symptom_dataset(split_map)

        print("Dataset preparation complete.")
        print(f"Manifest saved to: {MANIFEST_PATH}")
        print(f"Class summary saved to: {SUMMARY_PATH}")
        print(f"Simulated symptom dataset saved to: {symptom_path}")
        print(json.dumps(summary, indent=2))
    except FileNotFoundError as exc:
        print(f"Dataset pipeline warning: {exc}")
        print("Please add the raw PCOS ultrasound dataset to data/raw/pcos_ultrasound/ before training.")
        print("Expected folders: data/raw/pcos_ultrasound/infected and data/raw/pcos_ulsoft/ not_infected")
