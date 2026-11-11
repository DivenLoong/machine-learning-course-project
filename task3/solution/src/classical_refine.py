"""Evaluate a local fovea-darkness refinement around model predictions."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CACHE, pixels_rmse


def load_oof(path: Path) -> pd.DataFrame:
    parts = [pd.read_csv(item) for item in sorted(path.glob("oof_f*.csv"))]
    return pd.concat(parts).drop_duplicates("id").sort_values("id").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    predictions = load_oof(Path(args.run))
    metadata = pd.read_csv(CACHE / "canon_meta.csv").set_index("id")
    images = np.load(CACHE / "canon_2000.npy", mmap_mode="r")

    parameters = list(itertools.product(
        (5.0, 10.0, 20.0, 30.0),
        (0.0, 0.25, 0.5, 1.0),
        (0.0, 0.25, 0.5, 1.0),
    ))
    results = {parameter: [] for parameter in parameters}
    targets = []

    for row in predictions.itertuples():
        i = int(row.id)
        meta = metadata.loc[i]
        px = (float(row.pred_x) - meta.x0) * meta.scale
        py = (float(row.pred_y) - meta.y0) * meta.scale
        gx = (float(row.gt_x) - meta.x0) * meta.scale
        gy = (float(row.gt_y) - meta.y0) * meta.scale
        targets.append([gx, gy])

        radius = 150
        x0 = int(np.clip(px - radius - 24, 0, 2000))
        x1 = int(np.clip(px + radius + 24, 0, 2000))
        y0 = int(np.clip(py - radius - 24, 0, 2000))
        y1 = int(np.clip(py + radius + 24, 0, 2000))
        patch = np.asarray(images[i, y0:y1, x0:x1, 1], dtype=np.float32)
        ys, xs = np.mgrid[y0:y1, x0:x1]
        distance2 = (xs - px) ** 2 + (ys - py) ** 2
        valid = distance2 <= radius ** 2

        vessel_path = CACHE / "vessel" / f"{i:04d}.png"
        vessel = cv2.imdecode(np.fromfile(vessel_path, np.uint8), cv2.IMREAD_GRAYSCALE)
        vessel = cv2.resize(vessel, (2000, 2000), interpolation=cv2.INTER_LINEAR)
        vessel = vessel[y0:y1, x0:x1].astype(np.float32) / 255.0

        for sigma, vessel_weight, distance_weight in parameters:
            smooth = cv2.GaussianBlur(patch, (0, 0), sigma)
            if vessel_weight:
                vessel_smooth = cv2.GaussianBlur(vessel, (0, 0), sigma * 1.5)
            else:
                vessel_smooth = 0.0
            score = (smooth - patch) / 32.0
            score -= vessel_weight * vessel_smooth
            score -= distance_weight * distance2 / (2 * radius ** 2)
            score[~valid] = -np.inf
            index = int(np.argmax(score))
            y, x = divmod(index, score.shape[1])
            results[(sigma, vessel_weight, distance_weight)].append(
                [x + x0 + 0.5, y + y0 + 0.5])

    truth = np.asarray(targets)
    current = np.asarray([
        [(row.pred_x - metadata.loc[int(row.id)].x0) * metadata.loc[int(row.id)].scale,
         (row.pred_y - metadata.loc[int(row.id)].y0) * metadata.loc[int(row.id)].scale]
        for row in predictions.itertuples()
    ])
    print("current canonical RMSE", pixels_rmse(current, truth))
    ranked = []
    for parameter, values in results.items():
        predicted = np.asarray(values)
        ranked.append((pixels_rmse(predicted, truth), parameter, predicted))
    ranked.sort(key=lambda item: item[0])
    for score, parameter, _ in ranked[:15]:
        print(f"{score:9.3f} sigma={parameter[0]:4.1f} "
              f"vessel={parameter[1]:.2f} distance={parameter[2]:.2f}")

    best_score, best_parameter, best_pred = ranked[0]
    ids = predictions["id"].to_numpy(int)
    rows = []
    for i, pred, target in zip(ids, best_pred, truth):
        meta = metadata.loc[i]
        rows.append({
            "id": int(i),
            "pred_x": float(pred[0] / meta.scale + meta.x0),
            "pred_y": float(pred[1] / meta.scale + meta.y0),
            "gt_x": float(target[0] / meta.scale + meta.x0),
            "gt_y": float(target[1] / meta.scale + meta.y0),
        })
    output = pd.DataFrame(rows)
    out_path = Path(args.run) / "classical_refine.csv"
    output.to_csv(out_path, index=False)
    print("best", best_parameter, best_score, "wrote", out_path)


if __name__ == "__main__":
    main()
