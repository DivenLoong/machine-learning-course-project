"""Train the second-stage ROI refinement model for one or all folds."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT, load_or_make_folds, load_train_df, pixels_rmse, seed_everything
from infer import predict_loader
from model import HeatmapNet, soft_argmax
from refine_data import RefineDataset, roi_coords_to_pixels


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--coarse-run", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--jitter", type=float, default=40.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=str(PROJECT / "runs" / "refine001"))
    return p.parse_args()


def run_fold(args, df, fold, device):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    k = int(fold["fold"])
    # Use global out-of-fold coarse predictions for both train and val crops.
    # Each id's centre comes from a model that never saw that id.
    coarse_oof = pd.concat([
        pd.read_csv(Path(args.coarse_run) / f"oof_f{j}.csv") for j in range(5)
    ]).drop_duplicates("id")
    tr_ids, va_ids = fold["train_ids"], fold["val_ids"]

    ds_tr = RefineDataset(df, tr_ids, coarse_oof, args.crop, args.size, True,
                          args.samples, args.jitter)
    ds_va = RefineDataset(df, va_ids, coarse_oof, args.crop, args.size, False,
                          1, args.jitter)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, drop_last=False,
                       num_workers=args.workers, pin_memory=True,
                       persistent_workers=args.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=8, shuffle=False, num_workers=args.workers,
                       pin_memory=True, persistent_workers=args.workers > 0)

    model = HeatmapNet("efficientnet_b0", pretrained=True, drop_path_rate=0.05,
                       padding_mode="replicate").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lam(ep):
        if ep < 2:
            return (ep + 1) / 2
        prog = (ep - 2) / max(1, args.epochs - 2)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    scaler = torch.amp.GradScaler("cuda")
    gt = df.set_index("id").loc[va_ids, ["Fovea_X", "Fovea_Y"]].values
    gt_map = {int(i): gt[j] for j, i in enumerate(va_ids)}

    best = float("inf")
    hist = []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0; n = 0
        for b in dl_tr:
            images = b["image"].to(device, non_blocking=True)
            target = b["heatmap"].to(device, non_blocking=True)
            coord = b["coord"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred = model(images)
            with torch.amp.autocast("cuda", enabled=False):
                pred = pred.float(); target = target.float()
                loss = F.mse_loss(pred, target) + 0.1 * F.l1_loss(soft_argmax(pred), coord)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tot += float(loss.item()) * images.size(0); n += images.size(0)
        sched.step()

        ids, coords, metas = predict_loader(model, dl_va, device, tta=True, window=9)
        pred_px = roi_coords_to_pixels(coords, metas)
        rmse = pixels_rmse(pred_px, np.stack([gt_map[i] for i in ids]))
        hist.append({"epoch": ep + 1, "loss": tot / max(1, n), "val_rmse_px": rmse})
        print(f"[refine f{k}] ep {ep+1:03d}/{args.epochs} loss {tot/max(1,n):.5f} "
              f"val {rmse:.3f}px {time.time()-t0:.1f}s", flush=True)
        state = {"model": model.state_dict(), "args": vars(args), "fold": k,
                 "epoch": ep + 1, "val_rmse_px": rmse}
        torch.save(state, out / f"last_f{k}.pt")
        if rmse < best:
            best = rmse
            torch.save(state, out / f"best_f{k}.pt")
            print(f"[refine f{k}] new best {best:.3f}px", flush=True)

    pd.DataFrame(hist).to_csv(out / f"history_f{k}.csv", index=False)
    ck = torch.load(out / f"best_f{k}.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    ids, coords, metas = predict_loader(model, dl_va, device, tta=True, window=9)
    pred_px = roi_coords_to_pixels(coords, metas)
    rows = []
    for i, p in zip(ids, pred_px):
        g = gt_map[i]
        rows.append({"id": int(i), "pred_x": float(p[0]), "pred_y": float(p[1]),
                     "gt_x": float(g[0]), "gt_y": float(g[1])})
    pd.DataFrame(rows).sort_values("id").to_csv(out / f"oof_f{k}.csv", index=False)
    print(f"[refine f{k}] BEST {best:.3f}px", flush=True)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda")
    df = load_train_df()
    folds = load_or_make_folds(df)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "config.json").write_text(json.dumps(vars(args), indent=1))
    targets = folds if args.fold < 0 else [folds[args.fold]]
    for f in targets:
        run_fold(args, df, f, device)


if __name__ == "__main__":
    main()
