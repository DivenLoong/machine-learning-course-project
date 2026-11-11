"""Step 2 - stratified K-fold training of a pretrained backbone.

Saves per-fold checkpoints, histories, out-of-fold probabilities and a summary JSON
under ``outputs/<tag>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-folds", type=int, default=0, help="0 = all folds")
    ap.add_argument("--head-epochs", type=int, default=4)
    ap.add_argument("--ft-epochs", type=int, default=36)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--cutmix", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay, 0 disables")
    ap.add_argument("--raug", action="store_true", help="use RandAugment instead of ColorJitter")
    ap.add_argument("--raug-ops", type=int, default=2)
    ap.add_argument("--raug-mag", type=int, default=9)
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--mask-bg", action="store_true",
                    help="paint the soil background with a flat colour after cropping")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", default="",
                    help="comma-separated class names for a specialist model, e.g. "
                         "'Black-grass,Loose Silky-bent'")
    ap.add_argument("--resume", action="store_true",
                    help="reuse fold checkpoints that already exist and only score them")
    ap.add_argument("--pseudo", default="",
                    help="npz of test probabilities written by predict.py; its confident "
                         "predictions are added to every training fold")
    ap.add_argument("--pseudo-thresh", type=float, default=0.9)
    return ap.parse_args()


def load_norm(arch: str = "") -> tuple:
    """Normalisation constants: dataset statistics, or the backbone's own ImageNet ones."""
    if "." in arch:  # timm-style name -> respect the weights' expected preprocessing
        try:
            import timm

            cfg = timm.create_model(arch, pretrained=False).pretrained_cfg
            if cfg and cfg.get("mean"):
                return tuple(cfg["mean"]), tuple(cfg["std"])
        except Exception:  # noqa: BLE001 - fall through to the generic options
            pass
    stats_path = os.path.join(P.OUT_DIR, "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, encoding="utf-8") as fh:
            stats = json.load(fh)
        return tuple(stats["mean"]), tuple(stats["std"])
    return P.IMAGENET_MEAN, P.IMAGENET_STD


def main() -> None:
    args = parse_args()
    run_dir = P.ensure_dir(os.path.join(P.OUT_DIR, args.tag))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_norm(args.arch)

    cfg = P.TrainConfig(arch=args.arch, size=args.size, folds=args.folds,
                        batch_size=args.batch_size, head_epochs=args.head_epochs,
                        finetune_epochs=args.ft_epochs, lr=args.lr,
                        weight_decay=args.weight_decay,
                        label_smoothing=args.label_smoothing,
                        mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                        ema_decay=args.ema, raug=args.raug,
                        raug_ops=args.raug_ops, raug_mag=args.raug_mag,
                        tag=args.tag, seed=args.seed)
    crop_cfg = P.CropConfig(enabled=not args.no_crop, mask_background=args.mask_bg,
                            cache_name=("crop_boxes_masked.json" if args.mask_bg
                                        else "crop_boxes.json"))

    classes = list(P.CLASSES)
    if args.subset:
        classes = [c.strip() for c in args.subset.split(",") if c.strip()]
        for c in classes:
            if c not in P.CLASS_TO_IDX:
                raise SystemExit(f"unknown class {c!r}")
    subset_orig = [P.CLASS_TO_IDX[c] for c in classes]

    remap = {orig: new for new, orig in enumerate(subset_orig)}
    samples = [(p, remap[y]) for p, y in P.scan_train() if y in remap]
    n_real = len(samples)
    real_orig = np.array([subset_orig[y] for _, y in samples])

    n_pseudo = 0
    if args.pseudo:
        data = np.load(args.pseudo, allow_pickle=True)
        probs = data["probs"]
        cols = subset_orig if args.subset else list(range(len(P.CLASSES)))
        sub = probs[:, cols]
        sub = sub / np.maximum(sub.sum(axis=1, keepdims=True), 1e-9)
        keep = sub.max(axis=1) >= args.pseudo_thresh
        tests = P.scan_test()
        samples += [(tests[i][0], int(sub[i].argmax())) for i in np.where(keep)[0]]
        n_pseudo = int(keep.sum())
        print(f"pseudo-labelling: kept {n_pseudo}/{len(keep)} test images "
              f"at threshold {args.pseudo_thresh}")
    test_samples = P.scan_test()
    all_files = [p for p, _ in samples] + [p for p, _ in test_samples]
    boxes = P.build_crop_cache(all_files, crop_cfg)
    cache_dir = P.build_image_cache(all_files, crop_cfg, side=max(448, args.size))
    labels = np.array([y for _, y in samples[:n_real]])
    extra_idx = np.arange(n_real, len(samples))

    log_path = os.path.join(run_dir, "train.log")
    log_fh = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log(f"=== {args.tag} | arch={args.arch} size={args.size} folds={args.folds} "
        f"crop={not args.no_crop} device={device} ===")
    log(f"normalisation mean={np.round(mean, 4).tolist()} std={np.round(std, 4).tolist()}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    n_run = args.max_folds or args.folds
    oof_probs = np.zeros((n_real, len(classes)), dtype=np.float32)
    oof_seen = np.zeros(n_real, dtype=bool)
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        if fold >= n_run:
            break
        # Pseudo-labelled test images are never validated on: they join every training
        # fold only, so the out-of-fold estimate stays honest.
        tr_idx = np.concatenate([tr_idx, extra_idx]) if len(extra_idx) else tr_idx
        started = time.time()
        log(f"--- fold {fold + 1}/{args.folds} (train={len(tr_idx)} val={len(va_idx)}) ---")
        tr_ds = P.TrainDS([samples[i] for i in tr_idx],
                          P.train_transform(cfg.size, mean, std, raug=cfg.raug,
                                            raug_ops=cfg.raug_ops,
                                            raug_mag=cfg.raug_mag), crop_cfg, boxes,
                          cache_dir=cache_dir)
        va_ds = P.EvalDS([samples[i] for i in va_idx],
                         P.eval_transform(cfg.size, mean, std), crop_cfg, boxes,
                         cache_dir=cache_dir)
        tr_loader = P.make_loader(tr_ds, cfg.batch_size, True, cfg.num_workers, drop_last=True)
        va_loader = P.make_loader(va_ds, cfg.eval_batch_size, False, cfg.num_workers)

        ckpt_path = os.path.join(run_dir, f"fold{fold}_model.pt")
        hist_path = os.path.join(run_dir, f"fold{fold}_history.json")
        if args.resume and os.path.exists(ckpt_path) and os.path.exists(hist_path):
            log(f"    resuming fold {fold + 1} from {os.path.basename(ckpt_path)}")
            model = P.build_model(cfg.arch, len(classes), pretrained=False,
                                  img_size=cfg.size).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            with open(hist_path, encoding="utf-8") as fh:
                history = json.load(fh)
        else:
            model, history = P.train_one_fold(tr_loader, va_loader, cfg, device,
                                              num_classes=len(classes), log=log)
            torch.save(model.state_dict(), ckpt_path)
            with open(hist_path, "w", encoding="utf-8") as fh:
                json.dump(history, fh, indent=2)

        infer_ds = P.InferDS([(samples[i][0], i) for i in va_idx],
                             P.tta_views(cfg.size, mean, std), crop_cfg, boxes,
                             cache_dir=cache_dir)
        # TTA stacks 16 views per image, so a batch of 64 would need >1 GB of host RAM
        # per batch; keep it small and single-process to stay well clear of the limit.
        infer_loader = P.make_loader(infer_ds, 16, False, 0)
        probs, order = P.logits_for(model, infer_loader, device)
        for row, ds_idx in enumerate(order):
            global_idx = va_idx[ds_idx]
            oof_probs[global_idx] = probs[row]
            oof_seen[global_idx] = True
        y_true = np.array([labels[va_idx[i]] for i in order])
        y_pred = probs.argmax(axis=1)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        fold_metrics.append({"fold": fold, "accuracy": acc, "macro_f1": f1,
                             "best_val_acc": history["best_val_acc"],
                             "minutes": (time.time() - started) / 60})
        log(f"    fold {fold + 1} accuracy={acc:.4f} macro_f1={f1:.4f} "
            f"({(time.time() - started) / 60:.1f} min)")
        del model
        torch.cuda.empty_cache()

    np.savez(os.path.join(run_dir, "oof.npz"), probs=oof_probs, seen=oof_seen,
             labels=labels, orig_labels=real_orig)
    seen_idx = np.where(oof_seen)[0]
    if len(seen_idx):
        oof_acc = accuracy_score(labels[seen_idx], oof_probs[seen_idx].argmax(axis=1))
        oof_f1 = f1_score(labels[seen_idx], oof_probs[seen_idx].argmax(axis=1),
                          average="macro")
    else:
        oof_acc = oof_f1 = float("nan")
    summary = {"args": vars(args), "mean": list(mean), "std": list(std),
               "classes": classes, "n_folds": n_run, "n_pseudo": n_pseudo,
               "fold_metrics": fold_metrics,
               "oof_accuracy": oof_acc, "oof_macro_f1": oof_f1}
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log(f"=== {args.tag}: OOF accuracy={oof_acc:.4f} macro_f1={oof_f1:.4f} ===")
    log_fh.close()


if __name__ == "__main__":
    main()
