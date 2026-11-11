"""Pre-decode every image into the cropped cache (plain and masked variants)."""

from __future__ import annotations

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402


def main() -> None:
    files = [p for p, _ in P.scan_train()] + [p for p, _ in P.scan_test()]
    for cfg in (P.CropConfig(),
                P.CropConfig(mask_background=True, cache_name="crop_boxes_masked.json")):
        started = time.time()
        out = P.build_image_cache(files, cfg, side=448)
        label = "masked" if cfg.mask_background else "plain"
        print(f"{label}: {out}  ({time.time() - started:.1f}s for {len(files)} images)",
              flush=True)


if __name__ == "__main__":
    main()
