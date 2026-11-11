"""Visualise the images that every ensemble member gets wrong.

Six fine-tuned CNNs plus a frozen-feature probe disagree in useful ways, but some images
fail everywhere.  Looking at them is the fastest way to tell a modelling limit from a
label or preprocessing problem.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402

CNN_MEMBERS = ["b0_288", "cnx320", "cnx320p", "cnx320m", "cnx384m", "cnxs320m"]


def main() -> None:
    train = P.scan_train()
    labels = np.array([y for _, y in train])
    members = {m: np.load(os.path.join(P.OUT_DIR, m, "oof.npz"))["probs"]
               for m in CNN_MEMBERS}
    cnn = np.mean(list(members.values()), axis=0)
    probe = np.load(os.path.join(P.OUT_DIR, "dino_oof.npz"))["probs"]

    wrong_all = [i for i in range(len(labels))
                 if all(members[m][i].argmax() != labels[i] for m in CNN_MEMBERS)]
    probe_fixes = [i for i in wrong_all if probe[i].argmax() == labels[i]]
    print(f"wrong in every CNN member: {len(wrong_all)}")
    print(f"  of those, the DINOv2 probe is right on {len(probe_fixes)}")
    pairs = {}
    for i in wrong_all:
        pairs.setdefault((int(labels[i]), int(cnn[i].argmax())), []).append(i)
    for (t, p), idx in sorted(pairs.items(), key=lambda kv: -len(kv[1])):
        print(f"  truth {P.CLASSES[t]:<18s} -> predicted {P.CLASSES[p]:<18s} "
              f"{len(idx):3d}")

    cache = P.image_cache_dir(P.CropConfig(), 448)
    cell, cols = 150, 6
    rows = (len(wrong_all) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * cell), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    for k, i in enumerate(wrong_all):
        path, _ = train[i]
        with Image.open(os.path.join(cache, P.cache_key(path))) as im:
            tile = im.convert("RGB").copy()
        tile.thumbnail((cell, cell - 30), Image.BILINEAR)
        x, y = (k % cols) * cell, (k // cols) * cell
        canvas.paste(tile, (x + (cell - tile.width) // 2, y + 30))
        draw.text((x + 4, y + 4), f"truth {P.CLASSES[labels[i]][:10]}", fill=(255, 210, 120))
        draw.text((x + 4, y + 17), f"pred  {P.CLASSES[cnn[i].argmax()][:10]}",
                  fill=(150, 210, 255))
        mark = "PROBE OK" if i in probe_fixes else "probe wrong"
        draw.text((x + 4, cell - 12), mark, fill=(150, 255, 150) if i in probe_fixes
                  else (200, 120, 120))
    out = os.path.join(P.OUT_DIR, "hard_cases.png")
    canvas.save(out)
    print("montage:", os.path.relpath(out, P.ROOT))


if __name__ == "__main__":
    main()
