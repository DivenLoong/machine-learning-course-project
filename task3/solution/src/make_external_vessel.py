"""Generate vessel probability maps for the public REFUGE corpus."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import argparse
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CACHE, canonical_crop


def main():
    import fundus_image_toolbox as fit

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--csv", default=str(CACHE / "external_refuge.csv"))
    parser.add_argument("--out", default=str(CACHE / "external_refuge_vessel"))
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = frame[
        ~frame["id"].map(lambda value: (out / f"{int(value):05d}.png").exists())
    ].reset_index(drop=True)
    if args.limit:
        frame = frame.iloc[:args.limit].copy()
    print(f"generating {len(frame)} missing maps", flush=True)
    ensemble = fit.load_segmentation_ensemble(device="cuda:0")
    ensemble = ensemble[:args.models]
    done = 0
    for _, group in frame.groupby(["width", "height"], sort=False):
        for start in range(0, len(group), args.batch):
            part = group.iloc[start:start + args.batch]
            images = []
            for row in part.itertuples():
                raw = np.fromfile(row.image_path, dtype=np.uint8)
                image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            masks = fit.ensemble_predict_segmentation(
                ensemble, images, threshold=0.5, size=(512, 512))
            for row, mask in zip(part.itertuples(), masks):
                x0, y0, side = canonical_crop(row.width, row.height)
                crop = mask[y0:y0 + side, x0:x0 + side]
                crop = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
                Image.fromarray(np.clip(crop * 255, 0, 255).astype(np.uint8)).save(
                    out / f"{int(row.id):05d}.png")
                done += 1
            print(f"done {done}/{len(frame)}", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
