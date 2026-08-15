"""Download the public PCOS pelvic ultrasound dataset into the expected raw folder structure.

This script is intentionally defensive. It does NOT fabricate the dataset slug.
It expects a Kaggle dataset slug to be supplied either as:

- a command-line argument: --slug <user>/<dataset>
- or an environment variable: KAGGLE_DATASET_SLUG

Example:
    python data/download_pcos_dataset.py --slug <user>/<dataset-name>

The downloaded data is then organized into:
    data/raw/pcos_ultrasound/infected/
    data/raw/pcos_ultrasound/not_infected/

This project uses the PCOS ultrasound dataset as a same-organ-system proxy for
endometriosis screening support, and it documents that limitation clearly.
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

import kagglehub


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "pcos_ultrasound"


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the Kaggle PCOS ultrasound dataset into the expected raw folder structure.")
    parser.add_argument(
        "--slug",
        default=os.getenv("KAGGLE_DATASET_SLUG"),
        help="Kaggle dataset slug as user/dataset-name (e.g. user/dataset-name)",
    )
    return parser


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def organize_download(download_dir: Path, destination: Path) -> None:
    """Move the downloaded files into infected/not_infected folders if they already exist."""
    ensure_directory(destination)

    # If KaggleHub extracted a folder with nested content, search for 'infected' and 'not_infected'.
    for candidate in sorted(download_dir.rglob("*")):
        if candidate.is_dir() and candidate.name.lower() in {"infected", "not_infected"}:
            target = destination / candidate.name.lower()
            ensure_directory(target)
            for item in candidate.iterdir():
                if item.is_file():
                    shutil.copy2(item, target / item.name)

    # If there are direct files or subfolders without the expected labels, keep the downloaded folder as-is.
    if not any((destination / name).exists() for name in ["infected", "not_infected"]):
        copy_all_contents(download_dir, destination)


def copy_all_contents(source: Path, destination: Path) -> None:
    for item in sorted(source.iterdir()):
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def download_dataset(slug: str) -> Path:
    if not slug:
        raise ValueError(
            "No Kaggle dataset slug provided. Use --slug or set KAGGLE_DATASET_SLUG. "
            "Example: KAGGLE_DATASET_SLUG=user/dataset-name python data/download_pcos_dataset.py"
        )

    print(f"Downloading dataset slug: {slug}")
    download_path = kagglehub.dataset_download(slug)
    print(f"Dataset downloaded to: {download_path}")
    return Path(download_path)


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()

    try:
        download_dir = download_dataset(args.slug)
        organize_download(download_dir, RAW_ROOT)
        print(f"Raw dataset is ready in {RAW_ROOT}")
        print("Expected structure:")
        print("data/raw/pcos_ultrasound/infected/")
        print("data/raw/pcos_ultrasound/not_infected/")
    except Exception as exc:
        print(f"Dataset download failed: {exc}")
        print("Please provide a valid Kaggle dataset slug and ensure your Kaggle credentials are configured.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
