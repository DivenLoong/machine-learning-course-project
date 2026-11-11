"""Ensemble fold checkpoints with flip TTA and write a submission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT, SAMPLE_SUB, load_test_df
from data import FoveaDataset
from infer import coords_to_pixels, predict_loader
from model import HeatmapNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, nargs="+", required=True,
                   help="one or more run directories with best_f*.pt")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", type=str, default=str(PROJECT / "submissions" / "submission.csv"))
    p.add_argument("--save-raw", action="store_true")
    p.add_argument("--padding-mode", type=str, default="zeros")
    p.add_argument("--window", type=int, default=17)
    p.add_argument("--pattern", type=str, default="best_f*.pt")
    p.add_argument("--use-vessel", action="store_true")
    p.add_argument("--vessel-dir", type=str, default=str(PROJECT / "cache" / "vessel"))
    p.add_argument("--use-vascx-prior", action="store_true")
    p.add_argument("--vascx-prior-dir", type=str,
                   default=str(PROJECT / "cache" / "vascx_prior"))
    p.add_argument("--image-dir", type=str, default="",
                   help="Optional external image directory; IDs are assigned 2001+.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = []
    for run in args.run:
        ckpts.extend(sorted(Path(run).glob(args.pattern)))
    if not ckpts:
        raise FileNotFoundError(f"No best_f*.pt in {run}")

    if args.image_dir:
        rows = []
        for index, path in enumerate(sorted(Path(args.image_dir).rglob("*.jpg"))):
            with Image.open(path) as image:
                width, height = image.size
            rows.append({
                "id": 2001 + index, "image_path": str(path),
                "width": width, "height": height,
                "Fovea_X": width / 2.0, "Fovea_Y": height / 2.0,
            })
        df = pd.DataFrame(rows)
    else:
        df = load_test_df()
    ds = FoveaDataset(df.assign(Fovea_X=df["width"] / 2.0, Fovea_Y=df["height"] / 2.0),
                      df["id"].tolist(),
                      args.size, 2.0, train=False, augment=False,
                      vessel_dir=args.vessel_dir if args.use_vessel else None,
                      prior_dir=args.vascx_prior_dir
                      if args.use_vascx_prior else None)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    all_preds = []
    meta_ref = None
    id_ref = None
    for path in ckpts:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["args"]
        model = HeatmapNet(cfg.get("backbone", "efficientnet_b0"), pretrained=False,
                           padding_mode=args.padding_mode,
                           in_chans=3 + int(args.use_vessel)
                           + int(args.use_vascx_prior)).to(device)
        model.load_state_dict(ckpt["model"])
        ids, coords, metas = predict_loader(model, dl, device, tta=True,
                                            window=args.window)
        all_preds.append(coords_to_pixels(coords, metas))
        id_ref, meta_ref = ids, metas
        print(f"{path.name}: done", flush=True)

    pred_px = np.mean(np.stack(all_preds), axis=0)
    order = np.argsort(id_ref)
    ids = np.asarray(id_ref)[order]
    pred_px = pred_px[order]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame({"id": ids, "pred_x": pred_px[:, 0], "pred_y": pred_px[:, 1]})
    if args.image_dir:
        raw.to_csv(out, index=False)
        print(f"wrote {out} ({len(raw)} rows)")
        return

    sub = pd.read_csv(SAMPLE_SUB)
    value_map = {}
    for i, (x, y) in zip(ids, pred_px):
        value_map[f"{int(i)}_Fovea_X"] = float(x)
        value_map[f"{int(i)}_Fovea_Y"] = float(y)
    missing = [k for k in sub["ImageID"] if k not in value_map]
    if missing:
        raise RuntimeError(f"Missing predictions: {missing[:5]}")
    sub["value"] = sub["ImageID"].map(value_map)
    sub.loc[sub["value"].isna(), "value"] = 0.0
    sub.to_csv(out, index=False)

    raw.to_csv(out.with_name(out.stem + "_raw.csv"), index=False)
    print(f"wrote {out} ({len(sub)} rows)")
    print(raw.to_string(index=False))


if __name__ == "__main__":
    main()
