"""Evaluate a lightweight fovea head on frozen fundus-domain embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import ToTensor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT, load_or_make_folds, load_train_df, pixels_rmse, seed_everything
from fundus_image_toolbox import load_fovea_od_model


def load_rgb(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@torch.inference_mode()
def extract_embeddings(df, canonical: bool, size: int, device: torch.device):
    fit_model, _ = load_fovea_od_model("default", device=str(device))
    encoder = fit_model.model
    encoder.eval()
    rows = []
    to_tensor = ToTensor()
    for row in df.itertuples():
        image = Image.fromarray(load_rgb(row.image_path))
        if canonical:
            side = min(row.width, row.height)
            x0 = (row.width - side) // 2
            y0 = (row.height - side) // 2
            image = image.crop((x0, y0, x0 + side, y0 + side))
        else:
            side = min(row.width, row.height)
            x0 = (row.width - side) // 2
            y0 = (row.height - side) // 2

        image = image.resize((size, size), Image.Resampling.BILINEAR)
        tensor = to_tensor(image).unsqueeze(0).to(device)
        feat = encoder.features(tensor)
        feat = encoder.avgpool(feat).flatten(1)
        raw = encoder(tensor)
        rows.append(torch.cat([feat, raw], dim=1).cpu().numpy()[0])

    x = (df["Fovea_X"].to_numpy(float) - x0) / side
    y = (df["Fovea_Y"].to_numpy(float) - y0) / side
    # x0/y0/side are identical for all currently supported input shapes except
    # the five near-square images. Recompute exact targets per row.
    targets = []
    for row in df.itertuples():
        side = min(row.width, row.height)
        x0 = (row.width - side) // 2
        y0 = (row.height - side) // 2
        targets.append([(row.Fovea_X - x0) / side, (row.Fovea_Y - y0) / side])
    return np.stack(rows), np.asarray(targets, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=350)
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    seed_everything(20261027)
    device = torch.device("cuda")
    df = load_train_df()
    folds = load_or_make_folds(df)
    features, targets = extract_embeddings(df, args.canonical, args.size, device)
    ids = df["id"].to_numpy(int)
    fold_of = {}
    for fold in folds:
        for i in fold["val_ids"]:
            fold_of[int(i)] = int(fold["fold"])

    best = None
    for alpha in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
        pred = np.zeros_like(targets)
        for fold in folds:
            va_ids = np.asarray(fold["val_ids"], int)
            tr_ids = np.asarray(fold["train_ids"], int)
            tr_mask = np.isin(ids, tr_ids)
            va_mask = np.isin(ids, va_ids)
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(features[tr_mask], targets[tr_mask])
            pred[va_mask] = model.predict(features[va_mask])
        score = pixels_rmse(pred * 2000.0, targets * 2000.0)
        print(f"alpha={alpha:7.2f} normalized_rmse={score:8.3f}", flush=True)
        if best is None or score < best[0]:
            best = (score, alpha, pred.copy())

    score, alpha, pred = best
    rows = []
    for row_index, i in enumerate(ids):
        side = min(df.iloc[row_index]["width"], df.iloc[row_index]["height"])
        x0 = (df.iloc[row_index]["width"] - side) // 2
        y0 = (df.iloc[row_index]["height"] - side) // 2
        pred_px = np.array([x0, y0]) + pred[row_index] * side
        rows.append({
            "fold": fold_of[int(i)],
            "id": int(i),
            "pred_x": float(pred_px[0]),
            "pred_y": float(pred_px[1]),
            "gt_x": float(df.iloc[row_index]["Fovea_X"]),
            "gt_y": float(df.iloc[row_index]["Fovea_Y"]),
        })
    import pandas as pd
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    for fold in folds:
        frame[frame["fold"] == fold["fold"]].drop(columns="fold").to_csv(
            out / f"oof_f{fold['fold']}.csv", index=False)
    gt = frame[["gt_x", "gt_y"]].to_numpy(float)
    xy = frame[["pred_x", "pred_y"]].to_numpy(float)
    print(f"BEST alpha={alpha} canonical={args.canonical} "
          f"OOF={pixels_rmse(xy, gt):.3f}px")


if __name__ == "__main__":
    main()
