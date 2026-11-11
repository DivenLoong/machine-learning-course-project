"""Build a uint8 memmap of canonical 2000x2000 fundus crops (ids 1..100)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CACHE, PROJECT, TEST_DIR, TRAIN_DIR, canonical_crop, load_test_df, load_train_df


def main():
    out = CACHE / "canon_2000.npy"
    arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=(101, 2000, 2000, 3))
    meta = []
    train = load_train_df()
    test = load_test_df()
    for row in pd.concat([train[["id", "image_path"]], test[["id", "image_path"]]]).itertuples():
        buf = np.fromfile(row.image_path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        x0, y0, size = canonical_crop(w, h)
        crop = img[y0:y0 + size, x0:x0 + size]
        crop = cv2.resize(crop, (2000, 2000), interpolation=cv2.INTER_AREA)
        arr[int(row.id)] = crop
        meta.append({"id": int(row.id), "x0": x0, "y0": y0, "size": size,
                     "scale": 2000.0 / size})
        print(f"cached {int(row.id):04d}", flush=True)
    arr.flush()
    pd.DataFrame(meta).to_csv(CACHE / "canon_meta.csv", index=False)
    print("wrote", out, out.stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    main()
