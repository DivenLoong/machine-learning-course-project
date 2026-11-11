"""Domain pretraining on public REFUGE fovea annotations."""

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

from common import CACHE, PROJECT, pixels_rmse, seed_everything
from data import FoveaDataset
from infer import coords_to_pixels, predict_loader
from model import HeatmapNet, mask_centroid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone", default="efficientnet_b1")
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--coord-weight", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda")
    frame = pd.read_csv(CACHE / "external_refuge.csv")
    # Use all 800 public REFUGE images. A deterministic 80-image tail is
    # withheld only to monitor external-domain learning.
    frame = frame.iloc[:800].reset_index(drop=True)
    train_ids = frame["id"].iloc[:720].tolist()
    val_ids = frame["id"].iloc[720:].tolist()
    vessel_dir = CACHE / "external_refuge_vessel"
    missing = [i for i in train_ids + val_ids
               if not (vessel_dir / f"{int(i):05d}.png").exists()]
    if missing:
        raise FileNotFoundError(f"Missing vessel maps, first ids: {missing[:10]}")

    train_ds = FoveaDataset(
        frame, train_ids, args.size, target="circle", radius=args.radius,
        aug_level="strong", vessel_dir=str(vessel_dir), train=True, augment=True)
    val_ds = FoveaDataset(
        frame, val_ids, args.size, target="circle", radius=args.radius,
        vessel_dir=str(vessel_dir), train=False, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0)

    model = HeatmapNet(
        args.backbone, pretrained=True, drop_path_rate=args.drop_path,
        in_chans=4).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    gt = frame.set_index("id")[["Fovea_X", "Fovea_Y"]].to_dict("index")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=1))
    history = []
    best = float("inf")

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        count = 0
        started = time.time()
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            heatmap = batch["heatmap"].to(device, non_blocking=True).float()
            coords = batch["coord"].to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred = model(images)
            with torch.amp.autocast("cuda", enabled=False):
                pred = pred.float()
                bce = F.binary_cross_entropy(
                    pred.clamp(1e-4, 1 - 1e-4), heatmap)
                intersection = (pred * heatmap).sum((2, 3))
                dice = 1 - (2 * intersection + 1) / (
                    pred.sum((2, 3)) + heatmap.sum((2, 3)) + 1)
                loss = bce + dice.mean() + args.coord_weight * F.l1_loss(
                    mask_centroid(pred), coords)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * images.size(0)
            count += images.size(0)

        ids, coordinates, metas = predict_loader(
            model, val_loader, device, tta=True, window=17)
        pred_px = coords_to_pixels(coordinates, metas)
        gt_px = np.asarray([
            [gt[int(i)]["Fovea_X"], gt[int(i)]["Fovea_Y"]] for i in ids
        ])
        score = pixels_rmse(pred_px, gt_px)
        progress = (epoch + 1) / args.epochs
        learning_rate = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        history.append({
            "epoch": epoch + 1, "loss": running / count,
            "external_val_rmse_px": score, "lr": learning_rate,
            "seconds": time.time() - started,
        })
        state = {
            "model": model.state_dict(), "args": vars(args),
            "epoch": epoch + 1, "external_val_rmse_px": score,
        }
        torch.save(state, output / "last.pt")
        if score < best:
            best = score
            torch.save(state, output / "best.pt")
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(f"ep {epoch+1:03d}/{args.epochs} loss {running/count:.5f} "
              f"REFUGE val {score:.3f}px {time.time()-started:.1f}s",
              flush=True)
    print(f"BEST external validation {best:.3f}px; wrote {output}")


if __name__ == "__main__":
    main()
