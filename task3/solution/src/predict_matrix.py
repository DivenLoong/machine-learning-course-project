"""Collect per-fold TTA predictions on the test set for robust aggregation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT, load_test_df
from data import FoveaDataset
from infer import coords_to_pixels, predict_loader
from model import HeatmapNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", nargs="+", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="../submissions/test_model_predictions.csv")
    p.add_argument("--use-vessel", action="store_true")
    p.add_argument("--vessel-dir", default=str(PROJECT / "cache" / "vessel"))
    args = p.parse_args()
    device = torch.device("cuda")
    df = load_test_df()
    ds = FoveaDataset(
        df.assign(Fovea_X=df.width / 2, Fovea_Y=df.height / 2),
        df["id"].tolist(), args.size, 2.0, False, False,
        vessel_dir=args.vessel_dir if args.use_vessel else None)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
    rows = []
    for run in args.run:
        name = Path(run).name
        for ckpt in sorted(Path(run).glob("best_f*.pt")):
            ck = torch.load(ckpt, map_location=device, weights_only=False)
            model = HeatmapNet(ck["args"].get("backbone", "efficientnet_b0"),
                               pretrained=False,
                               in_chans=4 if args.use_vessel else 3).cuda()
            model.load_state_dict(ck["model"])
            ids, coords, metas = predict_loader(model, dl, device, tta=True, window=17)
            pred = coords_to_pixels(coords, metas)
            for i, xy in zip(ids, pred):
                rows.append({"run": name, "fold": int(ck["fold"]), "id": int(i),
                             "pred_x": float(xy[0]), "pred_y": float(xy[1])})
            print(f"{name} fold {ck['fold']} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print("wrote", args.out, len(out))


if __name__ == "__main__":
    main()
