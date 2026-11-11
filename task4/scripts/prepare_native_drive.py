"""Prepare the competition images for the native-resolution external trainer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drive_vessel.data import estimate_fov, list_drive_samples, read_rgb_native


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        destination.write_bytes(source.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="2026task-4/segmentation/train"
    )
    parser.add_argument(
        "--output", default="external_data/drive_native"
    )
    args = parser.parse_args()
    root = Path(args.output)
    image_dir = root / "training" / "images"
    label_dir = root / "training" / "1st_manual"
    mask_dir = root / "training" / "mask"
    for directory in (image_dir, label_dir, mask_dir):
        directory.mkdir(parents=True, exist_ok=True)

    samples = list_drive_samples(Path(args.source).parent, "train")
    for sample in samples:
        number = int(sample.image_id)
        if number > 20:
            raise ValueError(f"Unexpected training image id: {sample.image_id}")
        image_target = image_dir / f"{number:02d}_training.tif"
        label_target = label_dir / f"{number:02d}_manual1.gif"
        link_or_copy(sample.image_path, image_target)
        if sample.mask_path is None:
            raise FileNotFoundError(f"Missing mask for {sample.image_id}")
        link_or_copy(sample.mask_path, label_target)

        rgb = read_rgb_native(sample.image_path)
        fov = estimate_fov(rgb)
        Image.fromarray(np.asarray(fov, dtype=np.uint8) * 255).save(
            mask_dir / f"{number:02d}_training_mask.gif"
        )
    print(f"Prepared {len(samples)} native-resolution training images in {root}")


if __name__ == "__main__":
    main()
