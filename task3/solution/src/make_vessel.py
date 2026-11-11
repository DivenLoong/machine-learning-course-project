"""Generate vessel probability maps for all 100 images with FIVES ensemble."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CACHE, PROJECT, canonical_crop, load_test_df, load_train_df


def main():
    import fundus_image_toolbox as fit

    out = CACHE / "vessel"
    out.mkdir(parents=True, exist_ok=True)
    train = load_train_df()
    test = load_test_df()
    df = pd.concat([train[["id", "image_path", "width", "height"]],
                    test[["id", "image_path", "width", "height"]]]).reset_index(drop=True)
    ens = fit.load_segmentation_ensemble(device="cuda:0")

    done = 0
    for (w, h), g in df.groupby(["width", "height"]):
        rows = list(g.itertuples())
        for start in range(0, len(rows), 4):
            chunk = rows[start:start + 4]
            imgs = []
            for r in chunk:
                im = cv2.imdecode(np.fromfile(r.image_path, np.uint8), cv2.IMREAD_COLOR)
                imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            masks = fit.ensemble_predict_segmentation(
                ens, imgs, threshold=0.5, size=(512, 512))
            for r, m in zip(chunk, masks):
                x0, y0, s = canonical_crop(r.width, r.height)
                mc = m[y0:y0 + s, x0:x0 + s]
                mc = cv2.resize(mc, (512, 512), interpolation=cv2.INTER_AREA)
                # PIL handles non-ASCII Windows paths; cv2.imwrite silently fails.
                Image.fromarray(np.clip(mc * 255, 0, 255).astype(np.uint8)).save(
                    out / f"{int(r.id):04d}.png")
                done += 1
            print(f"done {done}/{len(df)}", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
