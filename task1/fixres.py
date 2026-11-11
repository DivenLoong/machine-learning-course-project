"""Check whether scoring at a different resolution than training helps (FixRes).

Random-resized-crop during training makes objects appear larger than they do at the
deterministic centre-crop used for evaluation; evaluating at a higher resolution
compensates for that train/test discrepancy.

Nothing is retrained here - only the already-saved fold checkpoints are re-scored.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cnx320p")
    ap.add_argument("--sizes", default="320,352,384,416")
    ap.add_argument("--scales", default="1.0,1.15")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


@torch.no_grad()
def score(tag_dir: str, tag: str, fold: int, arch: str, val_idx, samples,
          eval_size: int, scales, crop_cfg, boxes, cache_dir, mean, std,
          device) -> np.ndarray:
    model = P.build_model(arch, len(P.CLASSES), pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(tag_dir, f"fold{fold}_model.pt"),
                                     map_location=device))
    views = P.tta_views(eval_size, mean, std,
                        scales=[float(s) for s in scales.split(",")], rotations=4)
    ds = P.InferDS([(samples[i][0], i) for i in val_idx], views, crop_cfg, boxes,
                   cache_dir=cache_dir)
    loader = P.make_loader(ds, 64, False, 2)
    probs, order = P.logits_for(model, loader, device)
    out = np.zeros((len(val_idx), len(P.CLASSES)), dtype=np.float32)
    for row, ds_idx in enumerate(order):
        out[ds_idx] = probs[row]
    del model
    torch.cuda.empty_cache()
    return out


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(P.OUT_DIR, args.tag)
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as fh:
        summary = json.load(fh)
    run_args = summary["args"]
    arch, train_size = run_args["arch"], run_args["size"]
    mean = tuple(summary["mean"])
    std = tuple(summary["std"])

    crop_cfg = P.CropConfig(enabled=not run_args.get("no_crop", False),
                            mask_background=bool(run_args.get("mask_bg", False)),
                            cache_name=("crop_boxes_masked.json"
                                        if run_args.get("mask_bg") else "crop_boxes.json"))
    samples = P.scan_train()
    labels = np.array([y for _, y in samples])
    all_files = [p for p, _ in samples] + [p for p, _ in P.scan_test()]
    boxes = P.build_crop_cache(all_files, crop_cfg)
    cache_dir = P.build_image_cache(all_files, crop_cfg, side=448)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    splits = list(skf.split(np.zeros(len(labels)), labels))
    folds = sorted(int(f[4:-9]) for f in os.listdir(run_dir)
                   if f.startswith("fold") and f.endswith("_model.pt"))

    print(f"{args.tag}: arch={arch} trained at {train_size}, "
          f"{len(folds)} folds, n_pseudo={summary.get('n_pseudo', 0)}")
    results = {}
    for size in [int(s) for s in args.sizes.split(",")]:
        oof = np.zeros((len(labels), len(P.CLASSES)), dtype=np.float32)
        for fold in folds:
            oof[splits[fold][1]] = score(run_dir, args.tag, fold, arch, splits[fold][1],
                                         samples, size, args.scales, crop_cfg, boxes,
                                         cache_dir, mean, std, device)
        acc = accuracy_score(labels, oof.argmax(axis=1))
        results[size] = float(acc)
        print(f"  eval size {size:>4d}: OOF accuracy {acc:.4f}")
    best = max(results, key=results.get)
    print(f"best eval size: {best} ({results[best]:.4f}), "
          f"gain over {train_size}: {results[best] - results.get(train_size, float('nan')):+.4f}")


if __name__ == "__main__":
    main()
