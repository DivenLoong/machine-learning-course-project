"""Fine-tune FIT's fundus-domain fovea/OD regressor on the competition images."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT, load_or_make_folds, load_train_df, pixels_rmse, seed_everything
from data import build_train_aug
from fundus_image_toolbox import load_fovea_od_model


def read_rgb(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class FitDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, ids: list[int], size: int,
                 train: bool, aug_level: str = "strong"):
        self.frame = frame.set_index("id")
        self.ids = [int(i) for i in ids]
        self.size = int(size)
        self.aug = build_train_aug(self.size, aug_level) if train else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        i = self.ids[index]
        row = self.frame.loc[i]
        image = read_rgb(row["image_path"])
        side = min(int(row["width"]), int(row["height"]))
        x0 = (int(row["width"]) - side) // 2
        y0 = (int(row["height"]) - side) // 2
        image = image[y0:y0 + side, x0:x0 + side]
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        u = (float(row["Fovea_X"]) - x0) / side
        v = (float(row["Fovea_Y"]) - y0) / side
        if self.aug is not None:
            result = self.aug(
                image=image,
                keypoints=[(u * self.size, v * self.size)],
            )
            image = result["image"]
            x, y = result["keypoints"][0]
            u, v = x / self.size, y / self.size
        tensor = to_tensor(image)
        return {
            "image": tensor,
            "coord": torch.tensor([u, v], dtype=torch.float32),
            "id": i,
            "meta": torch.tensor([x0, y0, side], dtype=torch.float32),
        }


def make_model(device: torch.device):
    wrapper, _ = load_fovea_od_model("default", device=str(device))
    model = wrapper.model
    for parameter in model.parameters():
        parameter.requires_grad = False
    for block in model.features[7:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return model


@torch.inference_mode()
def predict(model, loader, device: torch.device):
    model.eval()
    ids, coords, metas = [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        acc = torch.zeros(images.shape[0], 2, device=device)
        for flip in ("none", "h", "v", "hv"):
            x = images
            if "h" in flip:
                x = torch.flip(x, dims=[3])
            if "v" in flip:
                x = torch.flip(x, dims=[2])
            c = model(x)[:, :2]
            if "h" in flip:
                c[:, 0] = 1 - c[:, 0]
            if "v" in flip:
                c[:, 1] = 1 - c[:, 1]
            acc += c
        ids.extend(batch["id"].tolist())
        coords.append((acc / 4).cpu().numpy())
        metas.append(batch["meta"].numpy())
    return np.asarray(ids), np.concatenate(coords), np.concatenate(metas)


def run_fold(args, df, fold, device):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    k = int(fold["fold"])
    train_ds = FitDataset(df, fold["train_ids"], args.size, True, args.aug)
    val_ds = FitDataset(df, fold["val_ids"], args.size, False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0)
    model = make_model(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    gt = df.set_index("id").loc[fold["val_ids"], ["Fovea_X", "Fovea_Y"]].to_dict("index")
    best = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        count = 0
        started = time.time()
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            coords = batch["coord"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(images)
                loss = F.smooth_l1_loss(output[:, :2].float(), coords)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 2.0)
            scaler.step(opt)
            scaler.update()
            train_loss += float(loss.item()) * images.size(0)
            count += images.size(0)

        ids, coords, metas = predict(model, val_loader, device)
        pred_px = coords * metas[:, 2:3] + metas[:, :2]
        gt_px = np.stack([
            [gt[int(i)]["Fovea_X"], gt[int(i)]["Fovea_Y"]] for i in ids
        ])
        score = pixels_rmse(pred_px, gt_px)
        history.append({"epoch": epoch + 1, "loss": train_loss / count,
                        "val_rmse_px": score, "lr": opt.param_groups[0]["lr"],
                        "seconds": time.time() - started})
        print(f"[fit f{k}] ep {epoch+1:03d}/{args.epochs} "
              f"loss {train_loss/count:.6f} val {score:.3f}px "
              f"{time.time()-started:.1f}s", flush=True)
        state = {"model": model.state_dict(), "fold": k, "epoch": epoch + 1,
                 "val_rmse_px": score, "args": vars(args)}
        torch.save(state, out / f"last_f{k}.pt")
        if score < best:
            best = score
            torch.save(state, out / f"best_f{k}.pt")
            print(f"[fit f{k}] new best {best:.3f}px", flush=True)
        if epoch < math.ceil(args.epochs * 0.1):
            opt.param_groups[0]["lr"] *= 1.08
        else:
            progress = (epoch - args.epochs * 0.1) / max(
                1, args.epochs * 0.9)
            opt.param_groups[0]["lr"] = args.lr * 0.5 * (
                1 + math.cos(math.pi * min(1.0, progress)))
        pd.DataFrame(history).to_csv(out / f"history_f{k}.csv", index=False)

    checkpoint = torch.load(out / f"best_f{k}.pt", map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["model"])
    ids, coords, metas = predict(model, val_loader, device)
    pred_px = coords * metas[:, 2:3] + metas[:, :2]
    rows = []
    for i, pred in zip(ids, pred_px):
        target = gt[int(i)]
        rows.append({
            "id": int(i), "pred_x": float(pred[0]), "pred_y": float(pred[1]),
            "gt_x": float(target["Fovea_X"]), "gt_y": float(target["Fovea_Y"]),
        })
    pd.DataFrame(rows).sort_values("id").to_csv(out / f"oof_f{k}.csv", index=False)
    print(f"[fit f{k}] FINAL {best:.3f}px", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--size", type=int, default=350)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--aug", default="strong")
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda")
    df = load_train_df()
    folds = load_or_make_folds(df)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    targets = folds if args.fold < 0 else [folds[args.fold]]
    for fold in targets:
        run_fold(args, df, fold, device)


if __name__ == "__main__":
    main()
