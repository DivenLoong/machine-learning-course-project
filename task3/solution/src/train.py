"""Train one or all folds of the fovea heatmap model."""

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
from torch.utils.data import WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (PROJECT, load_or_make_folds, load_train_df, pixels_rmse,
                    seed_everything)
from data import FoveaDataset
from infer import coords_to_pixels, predict_dual_loader, predict_loader
from losses import AdaptiveWingLoss
from model import HeatmapNet, mask_centroid, soft_argmax


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0, help="-1 runs all folds")
    p.add_argument("--only-folds", type=str, default="",
                   help="Comma-separated fold indices to run when --fold -1.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--backbone", type=str, default="efficientnet_b0")
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--target", type=str, default="gaussian", choices=["gaussian", "circle"])
    p.add_argument("--radius", type=float, default=6.0)
    p.add_argument("--loss", type=str, default="mse", choices=["mse", "bce"])
    p.add_argument("--heatmap-loss", type=str, default="auto",
                   choices=["auto", "bce_dice", "awing"])
    p.add_argument("--aug", type=str, default="base", choices=["base", "strong"])
    p.add_argument("--instance-norm", action="store_true")
    p.add_argument("--coord-weight", type=float, default=0.05)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--out", type=str, default=str(PROJECT / "runs" / "exp001"))
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--save-last-n", type=int, default=0,
                   help="save and ensemble the last N epoch checkpoints")
    p.add_argument("--window", type=int, default=17)
    p.add_argument("--reg-head", action="store_true")
    p.add_argument("--reg-weight", type=float, default=0.5)
    p.add_argument("--fuse-threshold", type=float, default=0.015,
                   help="normalized distance above which regression is used")
    p.add_argument("--folds-file", type=str, default="")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--aux-od", action="store_true")
    p.add_argument("--aux-weight", type=float, default=0.3)
    p.add_argument("--aux-sigma", type=float, default=3.0)
    p.add_argument("--mask-dir", type=str, default=str(PROJECT / "cache" / "masks"))
    p.add_argument("--use-vessel", action="store_true")
    p.add_argument("--vessel-dir", type=str, default=str(PROJECT / "cache" / "vessel"))
    p.add_argument("--use-vascx-prior", action="store_true")
    p.add_argument("--vascx-prior-dir", type=str,
                   default=str(PROJECT / "cache" / "vascx_prior"))
    p.add_argument("--difficulty-csv", type=str, default="",
                   help="Optional ID-level prediction errors used for weighted sampling.")
    p.add_argument("--difficulty-weight", type=float, default=0.0)
    p.add_argument("--difficulty-scale", type=float, default=30.0)
    p.add_argument("--difficulty-clip", type=float, default=4.0)
    p.add_argument("--pseudo-csv", type=str, default="",
                   help="Optional pseudo-labelled external images used in every training fold.")
    p.add_argument("--pseudo-weight", type=float, default=0.25)
    p.add_argument("--init-checkpoint", type=str, default="",
                   help="Optional compatible checkpoint for domain-pretrained initialization.")
    return p.parse_args()


def run_fold(args, df, fold_info: dict, device: torch.device,
             pseudo_df: pd.DataFrame | None = None):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold = int(fold_info["fold"])
    tr_ids = list(fold_info["train_ids"])
    va_ids = list(fold_info["val_ids"])
    if args.limit_train:
        tr_ids = tr_ids[: args.limit_train]
    if args.limit_val:
        va_ids = va_ids[: args.limit_val]

    pseudo_ids = pseudo_df["id"].tolist() if pseudo_df is not None else []
    train_source = (
        pd.concat([df, pseudo_df], ignore_index=True)
        if pseudo_df is not None else df)
    ds_tr = FoveaDataset(train_source, tr_ids + pseudo_ids, args.size, args.sigma,
                         train=True, augment=True,
                         target=args.target, radius=args.radius, aug_level=args.aug,
                         instance_norm=args.instance_norm, aux_od=args.aux_od,
                         mask_dir=args.mask_dir, aux_sigma=args.aux_sigma,
                         vessel_dir=args.vessel_dir if args.use_vessel else None,
                         prior_dir=args.vascx_prior_dir
                         if args.use_vascx_prior else None)
    ds_va = FoveaDataset(df, va_ids, args.size, args.sigma, train=False, augment=False,
                         target=args.target, radius=args.radius,
                         instance_norm=args.instance_norm, aux_od=args.aux_od,
                         mask_dir=args.mask_dir, aux_sigma=args.aux_sigma,
                         vessel_dir=args.vessel_dir if args.use_vessel else None,
                         prior_dir=args.vascx_prior_dir
                         if args.use_vascx_prior else None)
    sampler = None
    if (args.difficulty_csv and args.difficulty_weight > 0) or pseudo_ids:
        difficulty = (
            pd.read_csv(args.difficulty_csv).set_index("id")
            if args.difficulty_csv and args.difficulty_weight > 0 else None)
        weights = []
        for i in tr_ids + pseudo_ids:
            if int(i) in pseudo_ids:
                weights.append(float(args.pseudo_weight))
                continue
            weight = 1.0
            if difficulty is not None:
                row = difficulty.loc[int(i)]
                error = float(np.hypot(
                    row["pred_x"] - row["gt_x"], row["pred_y"] - row["gt_y"]))
                weight += args.difficulty_weight * np.clip(
                    error / args.difficulty_scale, 0.0,
                    args.difficulty_clip)
            weights.append(weight)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights), replacement=True)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=sampler is None,
                       sampler=sampler, num_workers=args.workers,
                       pin_memory=True, drop_last=False,
                       persistent_workers=args.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                       pin_memory=True, persistent_workers=args.workers > 0)

    model = HeatmapNet(args.backbone, pretrained=True, drop_path_rate=args.drop_path,
                       use_reg=args.reg_head, aux_head=args.aux_od,
                       in_chans=3 + int(args.use_vessel)
                       + int(args.use_vascx_prior)).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(
            args.init_checkpoint, map_location=device, weights_only=False)
        initial_state = checkpoint.get("model", checkpoint)
        current_state = model.state_dict()
        compatible = {}
        for key, value in initial_state.items():
            if key not in current_state or current_state[key].shape == value.shape:
                compatible[key] = value
                continue
            if (key.endswith("conv_stem.weight") and value.ndim == 4
                    and current_state[key].ndim == 4
                    and value.shape[0] == current_state[key].shape[0]):
                adapted = torch.zeros_like(current_state[key])
                channels = min(value.shape[1], current_state[key].shape[1])
                adapted[:, :channels] = value[:, :channels]
                if current_state[key].shape[1] > channels:
                    adapted[:, channels:] = value[:, :3].mean(1, keepdim=True)
                compatible[key] = adapted
        model.load_state_dict(compatible, strict=False)
        print(f"initialized from {args.init_checkpoint} "
              f"({len(compatible)} tensors)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(ep):
        if ep < args.warmup:
            return (ep + 1) / max(1, args.warmup)
        prog = (ep - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")
    awing = AdaptiveWingLoss().to(device)

    gt_px = {int(r.id): np.array([r.Fovea_X, r.Fovea_Y], dtype=np.float64)
             for r in df.itertuples() if int(r.id) in va_ids}

    best = float("inf")
    history = []
    snapshots = []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        total, n = 0.0, 0
        for batch in dl_tr:
            images = batch["image"].to(device, non_blocking=True)
            target = batch["heatmap"].to(device, non_blocking=True)
            coord = batch["coord"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                reg_pred = None
                aux_pred = None
                if args.reg_head and args.aux_od:
                    pred, reg_pred, aux_pred = model(
                        images, return_reg=True, return_aux=True)
                elif args.reg_head:
                    pred, reg_pred = model(images, return_reg=True)
                elif args.aux_od:
                    pred, aux_pred = model(images, return_aux=True)
                else:
                    pred = model(images)
            # Losses are computed in fp32; BCE is unsafe under autocast.
            with torch.amp.autocast("cuda", enabled=False):
                pred = pred.float()
                target = target.float()
                use_awing = args.heatmap_loss == "awing"
                if use_awing:
                    loss_hm = awing(pred, target)
                    loss_xy = F.l1_loss(soft_argmax(pred), coord)
                elif args.target == "circle":
                    bce = F.binary_cross_entropy(pred.clamp(1e-4, 1 - 1e-4), target)
                    inter = (pred * target).sum(dim=(2, 3))
                    dice = 1.0 - (2 * inter + 1.0) / (
                        pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1.0)
                    loss_hm = bce + args.dice_weight * dice.mean()
                    loss_xy = F.l1_loss(mask_centroid(pred), coord)
                elif args.loss == "bce":
                    loss_hm = F.binary_cross_entropy(pred.clamp(1e-4, 1 - 1e-4), target)
                    loss_xy = F.l1_loss(soft_argmax(pred), coord)
                else:
                    loss_hm = F.mse_loss(pred, target)
                    loss_xy = F.l1_loss(soft_argmax(pred), coord)
                loss = loss_hm + args.coord_weight * loss_xy
                if reg_pred is not None:
                    loss = loss + args.reg_weight * F.smooth_l1_loss(
                        reg_pred.float(), coord.float())
                if args.aux_od:
                    od_t = batch["od_heatmap"].to(device, non_blocking=True).float()
                    od_c = batch["od_coord"].to(device, non_blocking=True).float()
                    aux_pred = aux_pred.float()
                    aux_loss = F.mse_loss(aux_pred, od_t) + 0.1 * F.l1_loss(
                        soft_argmax(aux_pred), od_c)
                    loss = loss + args.aux_weight * aux_loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.item()) * images.size(0)
            n += images.size(0)
        sched.step()

        if args.reg_head:
            ids, hm_c, reg_c, fused_c, metas = predict_dual_loader(
                model, dl_va, device, tta=True, window=args.window,
                fuse_threshold=args.fuse_threshold)
            gt_val = np.stack([gt_px[i] for i in ids])
            rmse = pixels_rmse(coords_to_pixels(hm_c, metas), gt_val)
            reg_rmse = pixels_rmse(coords_to_pixels(reg_c, metas), gt_val)
            fused_rmse = pixels_rmse(coords_to_pixels(fused_c, metas), gt_val)
        else:
            ids, coords, metas = predict_loader(model, dl_va, device, tta=True,
                                                window=args.window)
            pred_px = coords_to_pixels(coords, metas)
            rmse = pixels_rmse(pred_px, np.stack([gt_px[i] for i in ids]))
            reg_rmse = fused_rmse = float("nan")
        train_loss = total / max(1, n)
        history.append({"epoch": ep + 1, "loss": train_loss, "val_rmse_px": rmse,
                        "reg_rmse_px": reg_rmse, "fused_rmse_px": fused_rmse,
                        "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0})
        print(f"[f{fold}] ep {ep+1:03d}/{args.epochs} loss {train_loss:.5f} "
              f"val_rmse {rmse:.3f}px reg {reg_rmse:.3f} fused {fused_rmse:.3f} "
              f"lr {opt.param_groups[0]['lr']:.2e} "
              f"{time.time()-t0:.1f}s", flush=True)

        state = {
            "model": model.state_dict(),
            "args": vars(args),
            "fold": fold,
            "epoch": ep + 1,
            "val_rmse_px": rmse,
        }
        torch.save(state, out_dir / f"last_f{fold}.pt")
        if rmse < best:
            best = rmse
            torch.save(state, out_dir / f"best_f{fold}.pt")
            print(f"[f{fold}] new best {best:.3f}px", flush=True)
        if args.save_last_n and ep >= args.epochs - args.save_last_n:
            sp = out_dir / f"snap_f{fold}_e{ep+1:03d}.pt"
            torch.save(state, sp)
            snapshots.append(sp)

    pd.DataFrame(history).to_csv(out_dir / f"history_f{fold}.csv", index=False)

    # OOF predictions with flip TTA using the best checkpoint.
    ckpt = torch.load(out_dir / f"best_f{fold}.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if args.reg_head:
        ids, hm_c, reg_c, fused_c, metas = predict_dual_loader(
            model, dl_va, device, tta=True, window=args.window,
            fuse_threshold=args.fuse_threshold)
        hm_px = coords_to_pixels(hm_c, metas)
        reg_px = coords_to_pixels(reg_c, metas)
        fused_px = coords_to_pixels(fused_c, metas)
        gt = np.stack([gt_px[i] for i in ids])
        rows = [{"id": int(i), "pred_x": float(h[0]), "pred_y": float(h[1]),
                 "reg_x": float(r[0]), "reg_y": float(r[1]),
                 "fused_x": float(f[0]), "fused_y": float(f[1]),
                 "gt_x": float(g[0]), "gt_y": float(g[1])}
                for i, h, r, f, g in zip(ids, hm_px, reg_px, fused_px, gt)]
    else:
        ids, coords, metas = predict_loader(model, dl_va, device, tta=True,
                                            window=args.window)
        pred_px = coords_to_pixels(coords, metas)
        gt = np.stack([gt_px[i] for i in ids])
        rows = [{"id": int(i), "pred_x": float(p[0]), "pred_y": float(p[1]),
                 "gt_x": float(g[0]), "gt_y": float(g[1])}
                for i, p, g in zip(ids, pred_px, gt)]
    oof = pd.DataFrame(rows).sort_values("id")
    oof.to_csv(out_dir / f"oof_f{fold}.csv", index=False)
    oof_hm = pixels_rmse(oof[["pred_x", "pred_y"]].values,
                         oof[["gt_x", "gt_y"]].values)
    if args.reg_head:
        oof_reg = pixels_rmse(oof[["reg_x", "reg_y"]].values,
                              oof[["gt_x", "gt_y"]].values)
        oof_fused = pixels_rmse(oof[["fused_x", "fused_y"]].values,
                                oof[["gt_x", "gt_y"]].values)
        print(f"[f{fold}] best {best:.3f}px, OOF hm {oof_hm:.3f} "
              f"reg {oof_reg:.3f} fused {oof_fused:.3f}", flush=True)
    else:
        print(f"[f{fold}] best {best:.3f}px, OOF+TTA {oof_hm:.3f}px", flush=True)

    if snapshots:
        coord_list = []
        for sp in snapshots:
            ck_s = torch.load(sp, map_location=device, weights_only=False)
            model.load_state_dict(ck_s["model"])
            ids_s, coords_s, metas_s = predict_loader(model, dl_va, device, tta=True,
                                                      window=args.window)
            coord_list.append(coords_s)
        coords_mean = np.mean(np.stack(coord_list, axis=0), axis=0)
        pred_px = coords_to_pixels(coords_mean, metas_s)
        rows = []
        for i, p in zip(ids_s, pred_px):
            g = gt_px[int(i)]
            rows.append({"id": int(i), "pred_x": float(p[0]), "pred_y": float(p[1]),
                         "gt_x": float(g[0]), "gt_y": float(g[1])})
        snap_df = pd.DataFrame(rows).sort_values("id")
        snap_df.to_csv(out_dir / f"oof_snap_f{fold}.csv", index=False)
        print(f"[f{fold}] snapshot-{len(snapshots)} OOF "
              f"{pixels_rmse(snap_df[['pred_x','pred_y']].values, snap_df[['gt_x','gt_y']].values):.3f}px",
              flush=True)
    return best


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.heatmap_loss == "awing" and args.target != "gaussian":
        raise ValueError("Adaptive Wing Loss expects --target gaussian.")
    df = load_train_df()
    folds_path = Path(args.folds_file) if args.folds_file else None
    folds = load_or_make_folds(df, folds_path, n_splits=args.n_splits, seed=args.seed)
    (Path(args.out)).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "config.json").write_text(json.dumps(vars(args), indent=1))

    targets = folds if args.fold < 0 else [folds[args.fold]]
    if args.only_folds:
        selected = {int(value) for value in args.only_folds.split(",") if value}
        targets = [fold for fold in targets if int(fold["fold"]) in selected]
    pseudo_df = pd.read_csv(args.pseudo_csv) if args.pseudo_csv else None
    for f in targets:
        run_fold(args, df, f, device, pseudo_df=pseudo_df)


if __name__ == "__main__":
    main()
