"""Per-image diagnostics: error vs mask confidence/size and flip disagreement."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_or_make_folds, load_train_df
from data import FoveaDataset
from model import HeatmapNet, heatmap_to_coord


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--batch", type=int, default=2)
    args = p.parse_args()
    device = torch.device("cuda")
    df = load_train_df()
    folds = load_or_make_folds(df)
    rows = []

    for fold in folds:
        k = fold["fold"]
        ck = torch.load(Path(args.run) / f"best_f{k}.pt", map_location=device,
                        weights_only=False)
        model = HeatmapNet(ck["args"].get("backbone", "efficientnet_b0"),
                           pretrained=False).cuda()
        model.load_state_dict(ck["model"])
        model.eval()
        ds = FoveaDataset(df, fold["val_ids"], 512, 2.0, train=False, augment=False)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)
        with torch.no_grad():
            for b in dl:
                x = b["image"].cuda()
                hm = model(x)
                base = heatmap_to_coord(hm)
                flips = []
                for fl in range(4):
                    xx = x
                    if fl & 1:
                        xx = torch.flip(xx, [3])
                    if fl & 2:
                        xx = torch.flip(xx, [2])
                    cc = heatmap_to_coord(model(xx))
                    if fl & 1:
                        cc[:, 0] = 1 - cc[:, 0]
                    if fl & 2:
                        cc[:, 1] = 1 - cc[:, 1]
                    flips.append(cc.cpu().numpy())
                stacked = np.stack(flips, 0)
                mean_c = stacked.mean(0)
                spread = stacked.std(0)
                for j, i in enumerate(b["id"].tolist()):
                    row = df[df["id"] == i].iloc[0]
                    meta = b["meta"][j].numpy()
                    p_base = base[j].cpu().numpy() * meta[2] + meta[:2]
                    p_tta = mean_c[j] * meta[2] + meta[:2]
                    gt = np.array([row.Fovea_X, row.Fovea_Y])
                    h = hm[j, 0]
                    prob = h.flatten() / h.sum().clamp_min(1e-8)
                    entropy = float(-(prob * (prob + 1e-8).log()).sum())
                    rows.append({
                        "fold": k, "id": int(i),
                        "err_base": float(np.hypot(*(p_base - gt))),
                        "err_tta": float(np.hypot(*(p_tta - gt))),
                        "dx_tta": float(p_tta[0] - gt[0]),
                        "dy_tta": float(p_tta[1] - gt[1]),
                        "peak": float(h.max()),
                        "area05": float((h > 0.5).sum()),
                        "area03": float((h > 0.3).sum()),
                        "entropy": entropy,
                        "spread_x": float(spread[j, 0] * meta[2]),
                        "spread_y": float(spread[j, 1] * meta[2]),
                        "spread": float(np.hypot(spread[j, 0], spread[j, 1]) * meta[2]),
                        "gt_x": float(gt[0]), "gt_y": float(gt[1]),
                        "pred_x": float(p_tta[0]), "pred_y": float(p_tta[1]),
                        "width": int(row.width), "height": int(row.height),
                    })

    out = pd.DataFrame(rows).sort_values("id")
    path = Path(args.out) if args.out else Path(args.run) / "diagnostics.csv"
    out.to_csv(path, index=False)
    print(out[["id", "err_tta", "peak", "area05", "entropy", "spread"]]
          .sort_values("err_tta", ascending=False).head(12).round(2).to_string(index=False))
    print("\ncorrelation with error:")
    for c in ["peak", "area05", "area03", "entropy", "spread", "width"]:
        print(f"  {c}: {out['err_tta'].corr(out[c]):.3f}")


if __name__ == "__main__":
    main()
