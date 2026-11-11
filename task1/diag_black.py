"""Why are some training images black? Compare the original with the cached crop."""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402

CNN_MEMBERS = ["b0_288", "cnx320", "cnx320p", "cnx320m", "cnx384m", "cnxs320m"]


def stats(img: Image.Image) -> str:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    dark = (arr.max(axis=2) < 16).mean()
    return (f"size={img.size} mode={img.mode} mean={arr.mean():6.1f} "
            f"dark_frac={dark:5.1%}")


def main() -> None:
    train = P.scan_train()
    labels = np.array([y for _, y in train])
    members = {m: np.load(os.path.join(P.OUT_DIR, m, "oof.npz"))["probs"]
               for m in CNN_MEMBERS}
    wrong_all = [i for i in range(len(labels))
                 if all(members[m][i].argmax() != labels[i] for m in CNN_MEMBERS)]

    cache = P.image_cache_dir(P.CropConfig(), 448)
    print(f"{'file':<18s} {'original':<46s} {'cached crop':<46s}")
    n_dark_orig = n_dark_cache = 0
    for i in wrong_all:
        path, lab = train[i]
        with Image.open(path) as im:
            im.load()
            orig = im.copy()
        with Image.open(os.path.join(cache, P.cache_key(path))) as im2:
            im2.load()
            cached = im2.copy()
        so, sc = stats(orig), stats(cached)
        dark_o = (np.asarray(orig.convert("RGB")).max(axis=2) < 16).mean()
        dark_c = (np.asarray(cached.convert("RGB")).max(axis=2) < 16).mean()
        n_dark_orig += dark_o > 0.5
        n_dark_cache += dark_c > 0.5
        print(f"{os.path.basename(path):<18s} {so:<46s} {sc:<46s}")
    print(f"\nhard cases dominated by black (>50% pixels): "
          f"original {n_dark_orig}/{len(wrong_all)}, cached {n_dark_cache}/{len(wrong_all)}")

    # How widespread is this across the whole training set?
    dark_counts = {"orig": 0, "cache": 0}
    for path, _lab in train:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"))
        if (arr.max(axis=2) < 16).mean() > 0.5:
            dark_counts["orig"] += 1
        with Image.open(os.path.join(cache, P.cache_key(path))) as im:
            arr = np.asarray(im.convert("RGB"))
        if (arr.max(axis=2) < 16).mean() > 0.5:
            dark_counts["cache"] += 1
    print(f"whole training set with >50% black pixels: "
          f"original {dark_counts['orig']}/500, cached {dark_counts['cache']}/500")


if __name__ == "__main__":
    main()
