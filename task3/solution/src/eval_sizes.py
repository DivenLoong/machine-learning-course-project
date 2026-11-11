"""Evaluate existing fold checkpoints at multiple input resolutions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_or_make_folds, load_train_df, pixels_rmse
from data import FoveaDataset
from infer import coords_to_pixels, predict_loader
from model import HeatmapNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--sizes", default="448,512,576,640,768")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    df = load_train_df()
    folds = load_or_make_folds(df)
    device = torch.device("cuda")
    sizes = [int(s) for s in args.sizes.split(",")]
    rows = []
    raw = []
    for fold in folds:
        k = fold["fold"]
        ckpt_path = Path(args.run) / f"best_f{k}.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = HeatmapNet(ckpt["args"].get("backbone", "efficientnet_b0"),
                           pretrained=False).to(device)
        model.load_state_dict(ckpt["model"])

        ids = fold["val_ids"]
        gt = df.set_index("id").loc[ids, ["Fovea_X", "Fovea_Y"]].values
        for size in sizes:
            ds = FoveaDataset(df, ids, size, 2.0, train=False, augment=False)
            dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
            out_ids, coords, metas = predict_loader(model, dl, device, tta=True)
            pred = coords_to_pixels(coords, metas)
            order = np.argsort(out_ids)
            rmse = pixels_rmse(pred[order], gt)
            rows.append({"fold": k, "size": size, "rmse": rmse})
            for i, pp, gg in zip(np.asarray(out_ids)[order], pred[order], gt):
                raw.append({"size": size, "id": int(i), "pred_x": pp[0],
                            "pred_y": pp[1], "gt_x": gg[0], "gt_y": gg[1]})
            print(f"fold {k} size {size}: {rmse:.3f} px", flush=True)

    res = pd.DataFrame(rows)
    raw_df = pd.DataFrame(raw)
    print("\nPer-size overall:")
    for size in sizes:
        g = raw_df[raw_df["size"] == size]
        overall = pixels_rmse(g[["pred_x", "pred_y"]].values, g[["gt_x", "gt_y"]].values)
        fold_rmse = res[res["size"] == size].rmse.values
        print(f"  {size}: overall {overall:.3f} px  folds {np.round(fold_rmse,2).tolist()}")
    res.to_csv(Path(args.run) / "size_eval.csv", index=False)
    raw_df.to_csv(Path(args.run) / "size_eval_raw.csv", index=False)


if __name__ == "__main__":
    main()
