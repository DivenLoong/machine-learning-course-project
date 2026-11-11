"""Run FIT's zero-shot regressor in the competition's canonical square crop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_test_df, load_train_df
from fundus_image_toolbox import load_fovea_od_model


def read_rgb(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@torch.inference_mode()
def predict_frame(df, device, batch_size=16):
    wrapper, _ = load_fovea_od_model("default", device=str(device))
    model = wrapper.model
    model.eval()
    outputs = []
    for start in range(0, len(df), batch_size):
        part = df.iloc[start:start + batch_size]
        images = []
        for row in part.itertuples():
            image = read_rgb(row.image_path)
            side = min(row.width, row.height)
            x0 = (row.width - side) // 2
            y0 = (row.height - side) // 2
            image = image[y0:y0 + side, x0:x0 + side]
            image = cv2.resize(image, (350, 350), interpolation=cv2.INTER_AREA)
            images.append(to_tensor(Image.fromarray(image)))
        images = torch.stack(images).to(device)
        acc = torch.zeros(images.shape[0], 4, device=device)
        for flip in ("none", "h", "v", "hv"):
            x = images
            if "h" in flip:
                x = torch.flip(x, dims=[3])
            if "v" in flip:
                x = torch.flip(x, dims=[2])
            out = model(x)
            if "h" in flip:
                out[:, 0] = 1 - out[:, 0]
                out[:, 2] = 1 - out[:, 2]
            if "v" in flip:
                out[:, 1] = 1 - out[:, 1]
                out[:, 3] = 1 - out[:, 3]
            acc += out
        outputs.append((acc / 4).cpu().numpy())
    return np.concatenate(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    df = load_test_df() if args.test else load_train_df()
    out = predict_frame(df, device)
    rows = []
    for row, prediction in zip(df.itertuples(), out):
        side = min(row.width, row.height)
        x0 = (row.width - side) // 2
        y0 = (row.height - side) // 2
        values = np.array([
            x0 + prediction[0] * side,
            y0 + prediction[1] * side,
            x0 + prediction[2] * side,
            y0 + prediction[3] * side,
        ])
        record = {
            "id": int(row.id),
            "pred_x": float(values[0]), "pred_y": float(values[1]),
            "od_x": float(values[2]), "od_y": float(values[3]),
        }
        if not args.test:
            record["gt_x"] = float(row.Fovea_X)
            record["gt_y"] = float(row.Fovea_Y)
        rows.append(record)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out, index=False)
    if not args.test:
        gt = frame[["gt_x", "gt_y"]].to_numpy(float)
        pred = frame[["pred_x", "pred_y"]].to_numpy(float)
        print("fovea RMSE",
              float(np.sqrt(np.mean((pred - gt) ** 2))))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
