"""Bagged cross-fitted ensemble weights for the final test prediction."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT, SAMPLE_SUB


def load_oof(run_dir: Path) -> pd.DataFrame:
    frame = pd.concat([
        pd.read_csv(path) for path in sorted(run_dir.glob("oof_f*.csv"))
    ]).drop_duplicates("id").sort_values("id").reset_index(drop=True)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds-run", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--test-raw", nargs="+", required=True)
    parser.add_argument("--vascx-train", required=True)
    parser.add_argument("--vascx-test", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    folds_run = Path(args.folds_run)
    train_frames = [load_oof(Path(run)) for run in args.runs]
    test_frames = [
        pd.read_csv(Path(path)).sort_values("id").reset_index(drop=True)
        for path in args.test_raw
    ]
    train_base = np.stack([
        frame[["pred_x", "pred_y"]].to_numpy(float) for frame in train_frames
    ], axis=1)
    test_base = np.stack([
        frame[["pred_x", "pred_y"]].to_numpy(float) for frame in test_frames
    ], axis=1)
    gt = train_frames[0][["gt_x", "gt_y"]].to_numpy(float)
    vascx_train = pd.read_csv(args.vascx_train).sort_values(
        "id")[["pred_x", "pred_y"]].to_numpy(float)
    vascx_test = pd.read_csv(args.vascx_test).sort_values(
        "id")[["pred_x", "pred_y"]].to_numpy(float)

    fold = np.zeros(len(gt), dtype=int)
    for path in sorted(folds_run.glob("oof_f*.csv")):
        frame = pd.read_csv(path)
        fold[frame["id"].to_numpy(int) - 1] = int(path.stem.split("_f")[-1])

    weights = []
    for a in range(11):
        for b in range(11 - a):
            weights.append(np.array([a, b, 10 - a - b], dtype=float) / 10)
    # VascX direct blending did not generalize in nested validation; retain the
    # hook but use a zero weight by default.
    parameters = [(weight, 0.0, 20) for weight in weights]

    def blend(base, vascx, weight, vascx_weight, threshold):
        prediction = sum(w * base[:, i] for i, w in enumerate(weight))
        if vascx_weight > 0:
            distance = np.linalg.norm(prediction - vascx, axis=1)
            mask = distance <= threshold
            prediction[mask] = (
                (1 - vascx_weight) * prediction[mask]
                + vascx_weight * vascx[mask])
        return prediction

    nested = np.zeros_like(gt)
    test_predictions = []
    selected = []
    for k in range(10):
        train_mask = fold != k
        val_mask = ~train_mask
        best = None
        for weight, vascx_weight, threshold in parameters:
            prediction = blend(
                train_base[train_mask], vascx_train[train_mask],
                weight, vascx_weight, threshold)
            score = float(np.sqrt(np.mean((prediction - gt[train_mask]) ** 2)))
            if best is None or score < best[0]:
                best = (score, weight, vascx_weight, threshold)
        _, weight, vascx_weight, threshold = best
        nested[val_mask] = blend(
            train_base[val_mask], vascx_train[val_mask],
            weight, vascx_weight, threshold)
        test_predictions.append(blend(
            test_base, vascx_test, weight, vascx_weight, threshold))
        selected.append((weight, vascx_weight, threshold))
        print(f"fold {k}: {weight.tolist()} vascx={vascx_weight} "
              f"threshold={threshold}", flush=True)

    print(f"nested OOF RMSE: {np.sqrt(np.mean((nested - gt) ** 2)):.3f}")
    prediction = np.mean(test_predictions, axis=0)
    ids = test_frames[0]["id"].to_numpy(int)
    raw = pd.DataFrame({
        "id": ids, "pred_x": prediction[:, 0], "pred_y": prediction[:, 1]})
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output.with_name(output.stem + "_raw.csv"), index=False)
    submission = pd.read_csv(SAMPLE_SUB)
    values = {}
    for row in raw.itertuples():
        values[f"{int(row.id)}_Fovea_X"] = float(row.pred_x)
        values[f"{int(row.id)}_Fovea_Y"] = float(row.pred_y)
    submission["value"] = submission["ImageID"].map(values)
    if submission["value"].isna().any():
        raise RuntimeError("Missing test ID in bagged prediction")
    submission.to_csv(output, index=False)
    print("wrote", output)


if __name__ == "__main__":
    main()
