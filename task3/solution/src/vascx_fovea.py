"""Evaluate VascX's public ConvNeXtV2 fovea heatmap model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "external" / "vascx_src"))

from rtnls_fundusprep.mask_extraction import get_cfi_bounds

from common import load_test_df, load_train_df

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_rgb(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess(image: np.ndarray, size: int, legacy: bool = False):
    bounds = get_cfi_bounds(image)
    transform, warped_bounds = bounds.crop(size)
    processed = warped_bounds.image
    encoded = (processed.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(encoded.transpose(2, 0, 1)).unsqueeze(0)
    if legacy:
        enhanced = warped_bounds.contrast_enhanced_5
        encoded_ce = (enhanced.astype(np.float32) / 255.0 - MEAN) / STD
        ce_tensor = torch.from_numpy(
            encoded_ce.transpose(2, 0, 1)).unsqueeze(0)
        tensor = torch.cat([tensor, ce_tensor], dim=1)
    return tensor, transform


def heatmap_coord(heatmap: torch.Tensor, window: int = 9) -> np.ndarray:
    heatmap = torch.sigmoid(heatmap.float())
    aggregate = heatmap.mean(1)[0, 0]
    height, width = aggregate.shape
    index = int(aggregate.argmax())
    peak_y, peak_x = index // width, index % width
    half = window // 2
    y0, y1 = max(0, peak_y - half), min(height, peak_y + half + 1)
    x0, x1 = max(0, peak_x - half), min(width, peak_x + half + 1)
    local = aggregate[y0:y1, x0:x1].clamp_min(1e-8)
    local = local / local.sum()
    xs = torch.arange(x0, x1, device=aggregate.device, dtype=aggregate.dtype) + 0.5
    ys = torch.arange(y0, y1, device=aggregate.device, dtype=aggregate.dtype) + 0.5
    x = float((local.sum(0) * xs).sum())
    y = float((local.sum(1) * ys).sum())
    return np.array([x, y], dtype=np.float64)


def infer_heatmap(model, tensor: torch.Tensor, size: int, sliding: bool):
    if not sliding:
        return model(tensor)
    patch = 512
    stride = patch // 2
    positions = list(range(0, size - patch + 1, stride))
    if positions[-1] != size - patch:
        positions.append(size - patch)
    weight_1d = torch.hann_window(
        patch, periodic=False, device=tensor.device, dtype=tensor.dtype
    ).clamp_min(1e-4)
    weight_2d = weight_1d[:, None] * weight_1d[None, :]
    output = torch.zeros(1, 1, 1, size, size, device=tensor.device, dtype=tensor.dtype)
    normalization = torch.zeros_like(output)
    for y0 in positions:
        for x0 in positions:
            tile = tensor[:, :, y0:y0 + patch, x0:x0 + patch]
            prediction = model(tile)
            output[:, :, :, y0:y0 + patch, x0:x0 + patch] += prediction * weight_2d
            normalization[:, :, :, y0:y0 + patch, x0:x0 + patch] += weight_2d
    return output / normalization.clamp_min(1e-6)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--image-dir", default="")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument("--sliding", action="store_true")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", type=Path, help="Local VascX TorchScript file")
    args = parser.parse_args()
    device = torch.device("cuda")
    # TorchScript's C++ file loader does not accept non-ASCII Windows paths.
    model_path = Path(
        args.checkpoint or (
            PROJECT / "external" / "vascx_models" / "fovea" /
            ("fovea_july24.pt" if args.legacy else "fovea_may26.pt")
        )
    )
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    if args.image_dir:
        rows = []
        for index, path in enumerate(sorted(Path(args.image_dir).rglob("*.jpg"))):
            from PIL import Image
            with Image.open(path) as image:
                width, height = image.size
            rows.append({
                "id": 2001 + index, "image_path": str(path),
                "width": width, "height": height,
            })
        frame = pd.DataFrame(rows)
    else:
        frame = load_test_df() if args.test else load_train_df()
    if args.limit:
        frame = frame.iloc[:args.limit].copy()

    rows = []
    for row in frame.itertuples():
        image = read_rgb(row.image_path)
        tensor, transform = preprocess(image, args.size, args.legacy)
        tensor = tensor.to(device)
        heatmap = infer_heatmap(model, tensor, args.size, args.sliding)
        point = heatmap_coord(heatmap, args.window)
        point = transform.apply_inverse(point[None, :])[0]
        record = {
            "id": int(row.id),
            "pred_x": float(np.clip(point[0], 0, row.width - 1)),
            "pred_y": float(np.clip(point[1], 0, row.height - 1)),
        }
        if not args.test and not args.image_dir:
            record["gt_x"] = float(row.Fovea_X)
            record["gt_y"] = float(row.Fovea_Y)
        rows.append(record)
        print(f"{int(row.id):04d} {record['pred_x']:.1f} {record['pred_y']:.1f}",
              flush=True)

    output = pd.DataFrame(rows)
    output.to_csv(args.out, index=False)
    if not args.test and not args.image_dir:
        d = output[["pred_x", "pred_y"]].to_numpy(float) - output[
            ["gt_x", "gt_y"]].to_numpy(float)
        print(f"OOF RMSE {np.sqrt(np.mean(d**2)):.3f}px")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
