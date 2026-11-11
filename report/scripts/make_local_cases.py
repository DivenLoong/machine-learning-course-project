"""Generate case figures for the local report from the original course workspace."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from shutil import copy2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPORT = Path(__file__).resolve().parents[1]
PRIVATE = REPORT / "private"
TEAL, AMBER = "#23D4AC", "#FFAF44"
FER_LABELS = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    for ax in fig.axes:
        ax.set_axis_off()
    fig.savefig(PRIVATE / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fer(root):
    run = root / "outputs/runs/cns224_f0"
    images = root / "2026task2/fer_data/fer_data/train"
    paths = [path for label in FER_LABELS for path in sorted((images / label).glob("*.jpg"))]
    idx = np.load(run / "val_idx.npy")
    prob = np.load(run / "val_probs.npy")
    truth = np.array([FER_LABELS.index(paths[i].parent.name) for i in idx])
    pred = np.argmax(prob, axis=1)
    conf = np.max(prob, axis=1)
    groups = [(truth == pred) & (truth == label) for label in (2, 4)]
    groups += [(truth != pred) & (truth == label) for label in (1, 3)]
    fig, axes = plt.subplots(1, 4, figsize=(8.5, 2.5), layout="constrained")
    for ax, mask in zip(axes, groups):
        ids = np.flatnonzero(mask)
        i = ids[np.argmax(conf[ids])]
        with Image.open(paths[idx[i]]) as image:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"True: {FER_LABELS[truth[i]]}\nPred: {FER_LABELS[pred[i]]} ({conf[i]:.2f})",
                     fontsize=9)
    save(fig, "fer_case.png")


def fovea(root):
    train = root / "2026task3/detection/train"
    run = root / "solution/runs/exp030_vessel_10fold_seed123"

    def coords(row):
        return np.array([float(row["pred_x"]), float(row["pred_y"])]), np.array([float(row["gt_x"]), float(row["gt_y"])])

    def error(row):
        pred, truth = coords(row)
        return np.sqrt(np.mean((pred - truth)**2))

    all_cases = [row for path in run.glob("oof_f*.csv") for row in rows(path)]
    assert all(1 <= int(r["id"]) <= 80 for r in all_cases)
    selected = [min(all_cases, key=error), max(all_cases, key=error)]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), layout="constrained")
    for ax, row in zip(axes, selected):
        pred, truth = coords(row)
        with Image.open(train / f"{int(row['id']):04d}.jpg") as image:
            ax.imshow(image.convert("RGB"))
        ax.scatter(*truth, s=60, color=TEAL, marker="o", facecolors="none", linewidths=1.5)
        ax.scatter(*pred, s=54, color=AMBER, marker="+", linewidths=1.5)
        center = (pred + truth) / 2
        half = max(160., float(np.max(np.abs(pred-truth))) * .83)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] + half, center[1] - half)
        ax.set_title(f"Validation coordinate RMSE: {error(row):.2f} px", fontsize=10)
    save(fig, "fovea_case.png")


def drive(root):
    import sys
    sys.path.insert(0, str(REPORT.parent / "task4"))
    from drive_vessel.data import list_drive_samples, read_rgb
    from drive_vessel.rle import decode_mask
    snap = root / "snapshots/best_public_0.81766_fr_official"
    sample = list_drive_samples(root / "2026task-4/segmentation", "test")[0]
    prediction = next(r for r in rows(snap / "submission_fr_official.csv") if int(r["Id"]) == int(sample.image_id))
    with np.load(snap / "probabilities.npz") as archive:
        probability = archive[str(int(sample.image_id))].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.9), layout="constrained")
    axes[0].imshow(read_rgb(sample.image_path, 512))
    axes[1].imshow(probability, cmap="magma", vmin=0, vmax=1)
    axes[2].imshow(decode_mask(prediction["Predicted"]), cmap="gray", vmin=0, vmax=1)
    for ax, title in zip(axes, ("Course test input", "Frozen probability", "Submitted binary mask")):
        ax.set_title(title, fontsize=9)
    save(fig, "drive_case.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, help="Parent of the four original task directories")
    base = parser.parse_args().workspace
    PRIVATE.mkdir(exist_ok=True)
    copy2(base / "机器学习1/2026task1/figure_defect.png", PRIVATE / "plant_case.png")
    fer(base / "机器学习2")
    fovea(base / "机器学习3")
    drive(base / "机器学习4")
