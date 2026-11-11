"""Step 1 - dataset reconnaissance: class balance, plant-crop coverage, normalisation stats."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402


def md5(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main() -> None:
    P.ensure_dir(P.OUT_DIR)
    train = P.scan_train()
    test = P.scan_test()
    print(f"train images: {len(train)}   test images: {len(test)}")
    print("train per class:", {c: sum(1 for _, y in train if y == i)
                               for i, c in enumerate(P.CLASSES)})

    cfg = P.CropConfig()
    files = [p for p, _ in train] + [p for p, _ in test]
    boxes = P.build_crop_cache(files, cfg)
    detected = sum(1 for p, _ in train if boxes.get(os.path.relpath(p, P.ROOT)))
    print(f"plant box detected on train: {detected}/{len(train)} "
          f"({detected / len(train):.1%})")
    print(f"plant box detected on test : "
          f"{sum(1 for p, _ in test if boxes.get(os.path.relpath(p, P.ROOT)))}/{len(test)}")

    # duplicate check inside train (and train<->test collisions)
    hashes = defaultdict(list)
    for p, _ in train:
        hashes[md5(p)].append(p)
    dup = {h: v for h, v in hashes.items() if len(v) > 1}
    print(f"exact duplicate groups inside train: {len(dup)}")

    # normalisation statistics over cropped images
    acc = np.zeros(3)
    acc_sq = np.zeros(3)
    n = 0
    for p, _ in train:
        img = P.apply_crop(Image.open(p).convert("RGB"),
                           tuple(boxes[os.path.relpath(p, P.ROOT)])
                           if boxes.get(os.path.relpath(p, P.ROOT)) else None, cfg)
        arr = np.asarray(img.resize((96, 96), Image.BILINEAR), dtype=np.float64) / 255.0
        acc += arr.reshape(-1, 3).mean(axis=0)
        acc_sq += (arr.reshape(-1, 3) ** 2).mean(axis=0)
        n += 1
    mean = acc / n
    std = np.sqrt(np.maximum(acc_sq / n - mean ** 2, 0))
    stats = {"mean": mean.tolist(), "std": std.tolist(), "crop": asdict(cfg)}
    with open(os.path.join(P.OUT_DIR, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"cropped-dataset mean={np.round(mean, 4).tolist()} std={np.round(std, 4).tolist()}")

    # montage: 5 classes x 3 samples, raw on top row and cropped below
    cell = 150
    per_class = 3
    cols = per_class
    canvas = Image.new("RGB", (cols * cell, len(P.CLASSES) * 2 * cell), (30, 30, 30))
    for ci, cls in enumerate(P.CLASSES):
        samples = [s for s in train if s[1] == ci][:per_class]
        for si, (p, _) in enumerate(samples):
            raw = Image.open(p).convert("RGB")
            box = boxes.get(os.path.relpath(p, P.ROOT))
            crop = P.apply_crop(raw, tuple(box) if box else None, cfg)
            for row, im in enumerate((raw, crop)):
                im = im.copy()
                im.thumbnail((cell, cell), Image.BILINEAR)
                tile = Image.new("RGB", (cell, cell), (30, 30, 30))
                tile.paste(im, ((cell - im.width) // 2, (cell - im.height) // 2))
                canvas.paste(tile, (si * cell, (ci * 2 + row) * cell))
    out = os.path.join(P.OUT_DIR, "eda_montage.png")
    canvas.save(out)
    print("montage written to", os.path.relpath(out, P.ROOT))


if __name__ == "__main__":
    main()
