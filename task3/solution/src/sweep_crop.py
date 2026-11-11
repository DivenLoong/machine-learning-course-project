"""Evaluate crop-shift TTA on fixed cross-validation folds."""

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
from infer import coords_to_pixels, predict_loader
from model import HeatmapNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--folds-file", required=True)
    parser.add_argument("--shifts", default="-64,0,64")
    args = parser.parse_args()
    shifts = [int(value) for value in args.shifts.split(",")]
    df = load_train_df()
    folds = load_or_make_folds(df, Path(args.folds_file))
    run = Path(args.run)
    device = torch.device("cuda")
    all_predictions = []
    ids_ref = gt_ref = None
    for fold in folds:
        checkpoint = torch.load(
            run / f"best_f{fold['fold']}.pt", map_location=device,
            weights_only=False)
        cfg = checkpoint["args"]
        model = HeatmapNet(
            cfg.get("backbone", "efficientnet_b0"), pretrained=False,
            in_chans=3 + int(cfg.get("use_vessel", False))
            + int(cfg.get("use_vascx_prior", False))).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        shift_predictions = []
        for shift in shifts:
            dataset = FoveaDataset(
                df, fold["val_ids"], cfg.get("size", 512), cfg.get("sigma", 2.0),
                train=False, augment=False, target=cfg.get("target", "circle"),
                radius=cfg.get("radius", 6.0),
                vessel_dir=str(Path(cfg["vessel_dir"]))
                if cfg.get("use_vessel") else None,
                crop_shift=(shift, 0))
            loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
            ids, coords, metas = predict_loader(
                model, loader, device, tta=True, window=17)
            shift_predictions.append(coords_to_pixels(coords, metas))
            gt = np.stack([
                df.set_index("id").loc[int(i), ["Fovea_X", "Fovea_Y"]].to_numpy(float)
                for i in ids
            ])
        for index in range(len(shift_predictions)):
            subset = np.mean(shift_predictions[:index + 1], axis=0)
            label = "shift_" + ",".join(str(value) for value in shifts[:index + 1])
            score = float(np.sqrt(np.mean((subset - gt) ** 2)))
            print(f"fold {fold['fold']} {label}: {score:.3f}", flush=True)
        all_predictions.append(np.mean(shift_predictions, axis=0))
        ids_ref, gt_ref = ids, gt
    prediction = np.concatenate(all_predictions)
    truth = np.concatenate([
        np.stack([
            df.set_index("id").loc[int(i), ["Fovea_X", "Fovea_Y"]].to_numpy(float)
            for i in fold["val_ids"]
        ]) for fold in folds
    ])
    print("combined OOF",
          float(np.sqrt(np.mean((prediction - truth) ** 2))))


if __name__ == "__main__":
    main()
