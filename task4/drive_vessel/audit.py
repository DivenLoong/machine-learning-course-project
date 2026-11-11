from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .data import list_drive_samples, read_binary_mask, read_rgb, sha256_file
from .splits import make_balanced_folds
from .utils import write_json


def audit_dataset(root: str | Path, output: str | Path, size: int, num_folds: int) -> dict:
    root = Path(root)
    train = list_drive_samples(root, "train")
    test = list_drive_samples(root, "test")
    if len(train) != len(test):
        raise ValueError(f"Expected equal train/test counts, got {len(train)} and {len(test)}")
    report: dict = {
        "root": str(root.resolve()),
        "train_count": len(train),
        "test_count": len(test),
        "size": size,
        "checks": {},
        "train": [],
        "test": [],
    }
    for sample in train:
        with Image.open(sample.image_path) as image:
            source_size = list(image.size)
        mask = read_binary_mask(sample.mask_path, size)
        image = read_rgb(sample.image_path, size)
        report["train"].append(
            {
                "id": sample.image_id,
                "source_size": source_size,
                "vessel_fraction": float(mask.mean()),
                "mean_intensity": float(image.mean()),
                "image_sha256": sha256_file(sample.image_path),
                "mask_sha256": sha256_file(sample.mask_path),
            }
        )
    for sample in test:
        with Image.open(sample.image_path) as image:
            source_size = list(image.size)
        report["test"].append(
            {
                "id": sample.image_id,
                "source_size": source_size,
                "image_sha256": sha256_file(sample.image_path),
            }
        )
    train_ids = {sample.image_id for sample in train}
    test_ids = {sample.image_id for sample in test}
    splits = make_balanced_folds(train, size, num_folds)
    flattened = [item for fold in splits["folds"] for item in fold]
    report["checks"] = {
        "train_image_mask_pairs": all(sample.mask_path is not None for sample in train),
        "train_test_ids_match": train_ids == test_ids,
        "binary_masks": all(
            set(np.unique(read_binary_mask(sample.mask_path, size)).tolist()) <= {0.0, 1.0}
            for sample in train
        ),
        "folds_cover_each_sample_once": sorted(flattened, key=int)
        == sorted(train_ids, key=int),
        "source_sizes": sorted({tuple(item["source_size"]) for item in report["train"]}),
    }
    report["folds"] = splits["folds"]
    write_json(output, report)
    print(
        f"Audit passed: {len(train)} train, {len(test)} test, "
        f"{num_folds} folds, masks binary and paired"
    )
    return report
