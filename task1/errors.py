"""Visualise the out-of-fold mistakes so we can tell model error from label noise."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cnx320")
    ap.add_argument("--cell", type=int, default=160)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = os.path.join(P.OUT_DIR, args.tag)
    data = np.load(os.path.join(run_dir, "oof.npz"))
    probs, labels = data["probs"], data["labels"]
    samples = P.scan_train()
    pred = probs.argmax(axis=1)
    wrong = np.where(pred != labels)[0]
    print(f"{args.tag}: {len(wrong)}/{len(labels)} out-of-fold errors")

    pairs = {}
    for i in wrong:
        pairs.setdefault((int(labels[i]), int(pred[i])), []).append(i)
    for (t, p), idxs in sorted(pairs.items(), key=lambda kv: -len(kv[1])):
        print(f"  {P.CLASSES[t]:>18s} -> {P.CLASSES[p]:<18s} {len(idxs):3d}")

    cell = args.cell
    cols = 6
    cache = P.imgcache_dir if hasattr(P, "imgcache_dir") else P.image_cache_dir(P.CropConfig(), 448)
    rows = (len(wrong) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * cell), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    for k, i in enumerate(wrong):
        path, _ = samples[i]
        with Image.open(os.path.join(cache, P.cache_key(path))) as im:
            tile = im.convert("RGB").copy()
        tile.thumbnail((cell, cell - 16), Image.BILINEAR)
        x, y = (k % cols) * cell, (k // cols) * cell
        canvas.paste(tile, (x + (cell - tile.width) // 2, y + 16))
        draw.text((x + 4, y + 3), f"{P.CLASSES[labels[i]][:9]}>{P.CLASSES[pred[i]][:9]}",
                  fill=(255, 220, 120))
        draw.text((x + 4, y + 28), f"p={probs[i].max():.2f}", fill=(160, 220, 255))
    out = os.path.join(P.OUT_DIR, f"errors_{args.tag}.png")
    canvas.save(out)
    print("montage written to", os.path.relpath(out, P.ROOT))


if __name__ == "__main__":
    main()
