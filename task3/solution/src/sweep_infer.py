"""Sweep coordinate decoding methods using existing fold checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_or_make_folds, load_train_df
from data import FoveaDataset
from infer import coords_to_pixels
from model import HeatmapNet


def decode(window, beta, power, heatmap):
    b, _, h, w = heatmap.shape
    xs = (torch.arange(w, device=heatmap.device, dtype=heatmap.dtype) + 0.5) / w
    ys = (torch.arange(h, device=heatmap.device, dtype=heatmap.dtype) + 0.5) / h
    if window is not None:
        flat = heatmap.reshape(b, -1)
        index = flat.argmax(1)
        result = torch.empty(b, 2, device=heatmap.device, dtype=heatmap.dtype)
        half = window // 2
        for k in range(b):
            peak_y, peak_x = int(index[k] // h), int(index[k] % w)
            y0, y1 = max(0, peak_y - half), min(h, peak_y + half + 1)
            x0, x1 = max(0, peak_x - half), min(w, peak_x + half + 1)
            local = heatmap[k, 0, y0:y1, x0:x1].clamp_min(1e-8)
            local = local / local.sum()
            result[k, 0] = (
                local.sum(0) * xs[x0:x1]).sum()
            result[k, 1] = (
                local.sum(1) * ys[y0:y1]).sum()
        return result
    if beta is not None:
        p = torch.softmax(heatmap.reshape(b, -1) * beta, 1).reshape_as(heatmap)
    else:
        p = heatmap.clamp_min(0).pow(power)
        p /= p.sum((1, 2, 3), keepdim=True).clamp_min(1e-8)
    return torch.stack([
        (p.sum(2) * xs).sum(2).squeeze(1),
        (p.sum(3) * ys).sum(2).squeeze(1),
    ], dim=1)


@torch.inference_mode()
def infer_run(run, df, folds, device, pattern: str = "best_f*.pt"):
    configs = {
        "window_5": lambda hm: decode(5, None, None, hm),
        "window_9": lambda hm: decode(9, None, None, hm),
        "window_13": lambda hm: decode(13, None, None, hm),
        "window_17": lambda hm: decode(17, None, None, hm),
        "window_21": lambda hm: decode(21, None, None, hm),
        "soft_5": lambda hm: decode(None, 5.0, None, hm),
        "soft_10": lambda hm: decode(None, 10.0, None, hm),
        "soft_25": lambda hm: decode(None, 25.0, None, hm),
        "soft_50": lambda hm: decode(None, 50.0, None, hm),
        "power_1": lambda hm: decode(None, None, 1.0, hm),
        "power_2": lambda hm: decode(None, None, 2.0, hm),
        "power_4": lambda hm: decode(None, None, 4.0, hm),
    }
    result = {name: [] for name in configs}
    ids_out, gt_out = [], []
    for fold in folds:
        matches = sorted(run.glob(pattern))
        checkpoint_by_fold = {
            int(path.stem.split("_f")[-1]): path for path in matches
        }
        ckpt_path = checkpoint_by_fold[int(fold["fold"])]
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = checkpoint["args"]
        model = HeatmapNet(
            cfg.get("backbone", "efficientnet_b0"), pretrained=False,
            in_chans=4 if cfg.get("use_vessel") else 3,
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        vessel_dir = Path(cfg.get("vessel_dir", "")) if cfg.get("use_vessel") else None
        if vessel_dir is not None and not vessel_dir.exists():
            vessel_dir = Path(__file__).resolve().parents[1] / "cache" / "vessel"
        dataset = FoveaDataset(
            df, fold["val_ids"], cfg.get("size", 512), cfg.get("sigma", 2.0),
            train=False, augment=False, target=cfg.get("target", "circle"),
            radius=cfg.get("radius", 6.0), instance_norm=cfg.get("instance_norm", False),
            vessel_dir=str(vessel_dir) if vessel_dir is not None else None,
        )
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
        fold_ids = []
        fold_gt = []
        fold_pred = {name: [] for name in configs}
        for batch in loader:
            images = batch["image"].to(device)
            accumulated = {name: torch.zeros(images.shape[0], 2, device=device)
                           for name in configs}
            for flip in ("none", "h", "v", "hv"):
                x = images
                if "h" in flip:
                    x = torch.flip(x, dims=[3])
                if "v" in flip:
                    x = torch.flip(x, dims=[2])
                heatmap = model(x)
                for name, function in configs.items():
                    coord = function(heatmap)
                    if coord.shape != (images.shape[0], 2):
                        raise RuntimeError(
                            f"{name} produced {tuple(coord.shape)}, expected "
                            f"{(images.shape[0], 2)}; heatmap {tuple(heatmap.shape)}"
                        )
                    if "h" in flip:
                        coord[:, 0] = 1 - coord[:, 0]
                    if "v" in flip:
                        coord[:, 1] = 1 - coord[:, 1]
                    accumulated[name] += coord
            for name in configs:
                result[name].append(coords_to_pixels(
                    (accumulated[name] / 4).cpu().numpy(), batch["meta"].numpy()))
            ids_out.extend(batch["id"].tolist())
            batch_gt = np.stack([
                df.set_index("id").loc[int(i), ["Fovea_X", "Fovea_Y"]].to_numpy(float)
                for i in batch["id"].tolist()
            ])
            gt_out.extend(batch_gt)
            fold_ids.extend(batch["id"].tolist())
            fold_gt.extend(batch_gt)
            for name in configs:
                fold_pred[name].append(result[name][-1])
        fold_rmse = np.sqrt(np.mean((np.concatenate(fold_pred["window_17"])
                                     - np.asarray(fold_gt)) ** 2))
        print(f"fold {fold['fold']} {ckpt_path.name}: {fold_rmse:.3f}px",
              flush=True)
    ids = np.asarray(ids_out)
    order = np.argsort(ids)
    gt = np.asarray(gt_out)[order]
    return {name: np.concatenate(values)[order] for name, values in result.items()}, ids[order], gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs="+", required=True)
    parser.add_argument("--pattern", default="best_f*.pt")
    parser.add_argument("--folds-file", default="")
    args = parser.parse_args()
    df = load_train_df()
    folds = load_or_make_folds(
        df, Path(args.folds_file) if args.folds_file else None)
    device = torch.device("cuda")
    outputs = {}
    ids = gt = None
    for name in args.run:
        outputs[name], ids, gt = infer_run(
            Path(name), df, folds, device, pattern=args.pattern)
        print(name)
        for method, pred in outputs[name].items():
            print(f"  {method:12s} {np.sqrt(np.mean((pred - gt) ** 2)):8.3f}")
    if len(outputs) > 1:
        print("EQUAL ENSEMBLE")
        for method in outputs[args.run[0]]:
            pred = np.mean([outputs[name][method] for name in args.run], axis=0)
            print(f"  {method:12s} {np.sqrt(np.mean((pred - gt) ** 2)):8.3f}")


if __name__ == "__main__":
    main()
