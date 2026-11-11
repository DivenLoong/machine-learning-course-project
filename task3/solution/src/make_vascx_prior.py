"""Convert VascX fovea predictions into an aligned prior input channel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CACHE, canonical_crop, load_test_df, load_train_df
from data import gaussian_heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-csv", required=True)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--sigma", type=float, default=8.0)
    parser.add_argument("--out", default=str(CACHE / "vascx_prior"))
    args = parser.parse_args()
    frame = load_test_df() if args.test else load_train_df()
    predictions = pd.read_csv(args.pred_csv).set_index("id")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    for row in frame.itertuples():
        x0, y0, side = canonical_crop(row.width, row.height)
        pred = predictions.loc[int(row.id)]
        u = (float(pred["pred_x"]) - x0) / side
        v = (float(pred["pred_y"]) - y0) / side
        heatmap = gaussian_heatmap(512, u, v, args.sigma)[0].numpy()
        Image.fromarray(np.clip(heatmap * 255, 0, 255).astype(np.uint8)).save(
            output / f"{int(row.id):04d}.png")
    print(f"wrote {len(frame)} prior maps to {output}")


if __name__ == "__main__":
    main()
