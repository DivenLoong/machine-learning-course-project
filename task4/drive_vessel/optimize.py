from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import list_drive_samples, read_binary_mask
from .inference import load_probability_archive, postprocess_mask
from .metrics import binary_metrics


def combine_oof_archives(paths: list[str | Path]) -> dict[str, np.ndarray]:
    combined: dict[str, np.ndarray] = {}
    for path in paths:
        archive = load_probability_archive(path)
        overlap = set(combined).intersection(archive)
        if overlap:
            raise ValueError(f"OOF folds overlap on image IDs: {sorted(overlap, key=int)}")
        combined.update(archive)
    return combined


def optimize_threshold_and_area(
    config: dict,
    probabilities: dict[str, np.ndarray],
    thresholds: list[float] | None = None,
    min_areas: list[int] | None = None,
) -> dict:
    thresholds = thresholds or [round(value, 2) for value in np.arange(0.20, 0.81, 0.05)]
    min_areas = min_areas or [0, 8, 16, 32, 64]
    dti_options = [False]
    if bool(config["eval"].get("dti_search", False)):
        dti_options = [False, True]
    samples = {
        sample.image_id: sample
        for sample in list_drive_samples(config["data"]["root"], "train")
    }
    missing = set(probabilities) - set(samples)
    if missing:
        raise ValueError(f"Probability archive contains unknown IDs: {sorted(missing)}")

    targets = {}
    for image_id in probabilities:
        targets[image_id] = (
            read_binary_mask(
                samples[image_id].mask_path, int(config["data"]["size"])
            )
            > 0.5
        )

    candidates = []
    for threshold in thresholds:
        for min_area in min_areas:
            for dti in dti_options:
                rows = []
                low_threshold = threshold - 0.15 if dti else None
                for image_id, probability in probabilities.items():
                    prediction = postprocess_mask(
                        probability,
                        samples[image_id].image_path,
                        int(config["data"]["size"]),
                        float(threshold),
                        min_area=min_area,
                        fov_mask=bool(config["eval"].get("fov_mask", True)),
                        threshold_low=low_threshold,
                        dti=dti,
                    )
                    rows.append(binary_metrics(targets[image_id], prediction))
                dice = float(np.mean([row["dice"] for row in rows]))
                candidates.append(
                    {
                        "threshold": float(threshold),
                        "threshold_low": low_threshold,
                        "dti": dti,
                        "min_area": int(min_area),
                        "dice": dice,
                        "sensitivity": float(np.mean([row["sensitivity"] for row in rows])),
                        "specificity": float(np.mean([row["specificity"] for row in rows])),
                        "worst_dice": float(min(row["dice"] for row in rows)),
                    }
                )
    best = max(candidates, key=lambda row: (row["dice"], row["worst_dice"]))
    return {"best": best, "candidates": candidates, "image_count": len(probabilities)}
