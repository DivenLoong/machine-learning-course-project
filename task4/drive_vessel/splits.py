from __future__ import annotations

import json
from pathlib import Path

from .data import Sample, read_binary_mask, read_rgb, sha256_file


def make_balanced_folds(samples: list[Sample], size: int, num_folds: int = 5) -> dict:
    if num_folds < 2:
        raise ValueError("num_folds must be at least two")
    records = []
    for sample in samples:
        if sample.mask_path is None:
            raise ValueError("Every fold sample requires a mask")
        mask = read_binary_mask(sample.mask_path, size)
        image = read_rgb(sample.image_path, size)
        records.append(
            {
                "id": sample.image_id,
                "vessel_fraction": float(mask.mean()),
                "mean_intensity": float(image.mean()),
                "sha256": sha256_file(sample.image_path),
            }
        )
    vessel_range = max(max(r["vessel_fraction"] for r in records) - min(r["vessel_fraction"] for r in records), 1e-6)
    intensity_range = max(max(r["mean_intensity"] for r in records) - min(r["mean_intensity"] for r in records), 1e-6)
    for record in records:
        record["balance_score"] = (
            record["vessel_fraction"] / vessel_range
            + record["mean_intensity"] / intensity_range
        )
    records.sort(key=lambda record: record["balance_score"], reverse=True)

    folds = [[] for _ in range(num_folds)]
    for index, record in enumerate(records):
        block = index // num_folds
        within = index % num_folds
        fold_index = within if block % 2 == 0 else num_folds - within - 1
        folds[fold_index].append(record["id"])

    all_ids = {record["id"] for record in records}
    validation_ids = [image_id for fold in folds for image_id in fold]
    assert len(validation_ids) == len(all_ids), "Folds must cover all samples exactly once"
    return {
        "num_folds": num_folds,
        "seed_note": "deterministic vessel-fraction/intensity snake balancing",
        "folds": folds,
        "records": records,
    }


def load_or_create_folds(path: Path, samples: list[Sample], size: int, num_folds: int) -> dict:
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            split = json.load(handle)
        if split.get("num_folds") != num_folds:
            raise ValueError(f"Existing fold file has {split.get('num_folds')} folds, expected {num_folds}")
        return split
    split = make_balanced_folds(samples, size, num_folds)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2)
    return split
