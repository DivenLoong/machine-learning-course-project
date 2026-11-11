"""Frozen self-supervised features + linear probe, as a complementary ensemble member.

Every current member is a fine-tuned supervised CNN, and analysis showed 16 images are
wrong in all of them.  A frozen DINOv2 backbone is a mechanically different information
source, so it can plausibly fix images the CNN ensemble cannot.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="dino")
    return ap.parse_args()


@torch.no_grad()
def extract(model, loader, device) -> np.ndarray:
    model.eval()
    feats = []
    for batch, _idx in loader:
        batch = batch.to(device, non_blocking=True)
        if batch.dim() == 5:            # (B, views, C, H, W) from the TTA dataset
            b, v = batch.shape[0], batch.shape[1]
            batch = batch.flatten(0, 1)
            with torch.amp.autocast("cuda", enabled=True):
                out = model(batch)
            out = out.float().reshape(b, v, -1).mean(dim=1)
            feats.append(torch.nn.functional.normalize(out, dim=1).cpu().numpy())
            continue
        with torch.amp.autocast("cuda", enabled=True):
            out = model(batch)
        out = out.float()
        out = torch.nn.functional.normalize(out, dim=1)
        feats.append(out.cpu().numpy())
    return np.concatenate(feats)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import timm

    model = timm.create_model(args.arch, pretrained=True, num_classes=0,
                              img_size=args.size).to(device)
    cfg = model.pretrained_cfg
    mean, std = tuple(cfg["mean"]), tuple(cfg["std"])
    print(f"backbone {args.arch} input {args.size} "
          f"feature dim {model.num_features} device {device}")

    crop_cfg = P.CropConfig()          # plain crop, same as the unmasked members
    train = P.scan_train()
    test = P.scan_test()
    all_files = [p for p, _ in train] + [p for p, _ in test]
    boxes = P.build_crop_cache(all_files, crop_cfg)
    cache_dir = P.build_image_cache(all_files, crop_cfg, side=max(448, args.size))

    views = [
        P.eval_transform(args.size, mean, std, scale=1.0),
        P.eval_transform(args.size, mean, std, scale=1.0, hflip=True),
    ]
    started = time.time()
    tr_ds = P.InferDS(train, views, crop_cfg, boxes, cache_dir=cache_dir)
    tr_loader = P.make_loader(tr_ds, args.batch_size, False, 4)
    tr_feat = extract(model, tr_loader, device)
    te_ds = P.InferDS(test, views, crop_cfg, boxes, cache_dir=cache_dir)
    te_loader = P.make_loader(te_ds, args.batch_size, False, 4)
    te_feat = extract(model, te_loader, device)
    print(f"features extracted in {time.time() - started:.1f}s "
          f"train {tr_feat.shape} test {te_feat.shape}")

    labels = np.array([y for _, y in train])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for C in (0.1, 1.0, 10.0):
        oof = np.zeros((len(labels), len(P.CLASSES)))
        for tr_idx, va_idx in skf.split(np.zeros(len(labels)), labels):
            scaler = StandardScaler().fit(tr_feat[tr_idx])
            clf = LogisticRegression(C=C, max_iter=3000, n_jobs=-1)
            clf.fit(scaler.transform(tr_feat[tr_idx]), labels[tr_idx])
            oof[va_idx] = clf.predict_proba(scaler.transform(tr_feat[va_idx]))
        acc = accuracy_score(labels, oof.argmax(1))
        print(f"  linear probe C={C:<5} OOF accuracy = {acc:.4f}")
        if C == 1.0:
            best_oof = oof

    np.savez(os.path.join(P.OUT_DIR, f"{args.tag}_features.npz"),
             train=tr_feat, test=te_feat, labels=labels)
    np.savez(os.path.join(P.OUT_DIR, f"{args.tag}_oof.npz"), probs=best_oof,
             labels=labels, seen=np.ones(len(labels), dtype=bool))

    # Does the probe fix what the CNN ensemble gets wrong?
    members = ["b0_288", "cnx320", "cnx320p", "cnx320m", "cnx384m", "cnxs320m"]
    cnn = np.mean([np.load(os.path.join(P.OUT_DIR, m, "oof.npz"))["probs"]
                   for m in members], axis=0)
    hard = np.where(cnn.argmax(1) != labels)[0]
    fixed = sum(1 for i in hard if best_oof[i].argmax() == labels[i])
    print(f"\nCNN ensemble errors: {len(hard)}; "
          f"linear probe gets {fixed} of them right")
    hard_shared = [i for i in hard if all(
        np.load(os.path.join(P.OUT_DIR, m, "oof.npz"))["probs"][i].argmax() != labels[i]
        for m in members)]
    fixed_shared = sum(1 for i in hard_shared if best_oof[i].argmax() == labels[i])
    print(f"images wrong in every CNN member: {len(hard_shared)}; "
          f"probe gets {fixed_shared} of them right")


if __name__ == "__main__":
    main()
