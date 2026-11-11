"""Step 3 - ensemble the fold checkpoints with multi-view TTA and write submission.csv."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="b0_288", help="comma-separated run tags to ensemble")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--scales", default="1.0,1.15")
    ap.add_argument("--rotations", type=int, default=4)
    ap.add_argument("--tta-flip", action="store_true",
                    help="add mirrored TTA views (doubles the view count)")
    ap.add_argument("--prior-json", default="",
                    help="blend.json from blend.py; applies the estimated class prior to "
                         "the decision rule (accuracy-optimal under prior shift)")
    ap.add_argument("--prior-scale", type=float, default=1.0,
                    help="0 = ignore the prior (uniform rule), 1 = full correction")
    ap.add_argument("--weights-json", default="",
                    help="weights.json from weights.py; each run is scaled by its share")
    return ap.parse_args()


def load_run(tag: str) -> dict:
    run_dir = os.path.join(P.OUT_DIR, tag)
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as fh:
        summary = json.load(fh)
    folds = sorted(int(f[4:-9]) for f in os.listdir(run_dir)
                   if f.startswith("fold") and f.endswith("_model.pt"))
    return {"tag": tag, "dir": run_dir, "summary": summary, "folds": folds}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tests = P.scan_test()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    total_probs = None
    run_weights = {}
    if args.weights_json:
        with open(args.weights_json, encoding="utf-8") as fh:
            cfg = json.load(fh)
        run_weights = dict(zip(cfg["tags"], cfg["weights"]))
        print("run weights:", {k: round(v, 3) for k, v in run_weights.items()})
    total_weight = 0.0
    n_models = 0

    for tag in tags:
        run = load_run(tag)
        run_args = run["summary"]["args"]
        run_classes = run["summary"].get("classes", P.CLASSES)
        size, arch = run_args["size"], run_args["arch"]
        mean, std = (tuple(run["summary"]["mean"]), tuple(run["summary"]["std"])) \
            if "mean" in run["summary"] else (P.IMAGENET_MEAN, P.IMAGENET_STD)
        print(f"  {tag}: arch={arch} size={size} norm={list(np.round(mean, 3))}",
              flush=True)
        masked = bool(run_args.get("mask_bg", False))
        crop_cfg = P.CropConfig(enabled=not run_args.get("no_crop", False),
                                mask_background=masked,
                                cache_name=("crop_boxes_masked.json" if masked
                                            else "crop_boxes.json"))
        all_files = [p for p, _ in P.scan_train()] + [p for p, _ in tests]
        boxes = P.build_crop_cache(all_files, crop_cfg)
        cache_dir = P.build_image_cache(all_files, crop_cfg, side=max(448, size))
        views = ([P.eval_transform(size, mean, std)] if args.no_tta else
                 P.tta_views(size, mean, std,
                             scales=[float(s) for s in args.scales.split(",")],
                             rotations=args.rotations, flips=args.tta_flip))
        ds = P.InferDS(tests, views, crop_cfg, boxes, cache_dir=cache_dir)
        # A TTA batch holds ``views`` images per sample, so keep it small and avoid
        # page-locked host memory to stay clear of host and device limits.
        loader = P.make_loader(ds, 12, False, 2, pin_memory=False)
        run_weight = run_weights.get(tag, 1.0)
        run_total = None
        for fold in run["folds"]:
            model = P.build_model(arch, len(run_classes), img_size=size).to(device)
            model.load_state_dict(torch.load(os.path.join(run["dir"], f"fold{fold}_model.pt"),
                                             map_location=device))
            probs, order = P.logits_for(model, loader, device)
            assert order.tolist() == list(range(len(tests))), "loader order changed"
            if run_total is None:
                run_total = np.zeros_like(probs)
            run_total += probs
            n_models += 1
            print(f"  {tag} fold{fold}: done ({n_models} models)", flush=True)
            del model
            torch.cuda.empty_cache()
        if run_total is None:
            continue
        if total_probs is None:
            total_probs = np.zeros_like(run_total)
        total_probs += run_weight * (run_total / max(len(run["folds"]), 1))
        total_weight += run_weight

    total_probs /= max(total_weight, 1e-9)

    prior = np.ones(total_probs.shape[1])
    if args.prior_json:
        with open(args.prior_json, encoding="utf-8") as fh:
            blend = json.load(fh)
        estimated = np.asarray(blend["estimated_prior"], dtype=np.float64)
        if len(estimated) != total_probs.shape[1]:
            raise SystemExit("prior has a different number of classes than the ensemble")
        prior = estimated ** args.prior_scale
        print(f"prior correction (scale={args.prior_scale}): "
              f"{np.round(estimated, 3).tolist()}")
    preds = (total_probs * prior).argmax(axis=1)

    order = [line.split(",")[0] for line in
             open(P.SAMPLE_SUBMISSION, encoding="utf-8").read().strip().splitlines()[1:]]
    id_to_row = {name: i for i, (_, name) in enumerate(tests)}
    out_path = os.path.join(P.ROOT, args.out)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("ID,Category\n")
        for name in order:
            fh.write(f"{name},{P.CLASSES[preds[id_to_row[name]]]}\n")

    np.savez(os.path.join(P.OUT_DIR, "test_probs.npz"), probs=total_probs,
             ids=np.array([n for _, n in tests]))
    print(f"wrote {os.path.relpath(out_path, P.ROOT)} with {len(order)} rows "
          f"from {n_models} models")
    print("prediction distribution:", Counter(P.CLASSES[p] for p in preds).most_common())
    conf = total_probs.max(axis=1)
    print(f"mean confidence={conf.mean():.3f}  "
          f"below 0.5 conf: {int((conf < 0.5).sum())}/{len(conf)}")


if __name__ == "__main__":
    main()
