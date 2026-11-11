"""Diagnostic: are any test images near-duplicates of a training image?

If they are, the task is partly solvable by nearest-neighbour lookup rather than by
learning, which would explain perfectly-scoring leaderboard entries.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402

SIDE = 448


def signature(path: str) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32)
    arr = arr - arr.mean()
    return (arr / (np.linalg.norm(arr) + 1e-6)).ravel()


def main() -> None:
    cache = P.image_cache_dir(P.CropConfig(), SIDE)
    train = [p for p, _ in P.scan_train()]
    test = [p for p, _ in P.scan_test()]

    def load(paths):
        return np.stack([signature(os.path.join(cache, P.cache_key(p))) for p in paths])

    tr = load(train)
    te = load(test)
    print(f"signatures: train={tr.shape} test={te.shape}")

    sim_tt = te @ tr.T                      # test vs train cosine similarity
    best_train = sim_tt.max(axis=1)
    arg_train = sim_tt.argmax(axis=1)

    sim_tr = tr @ tr.T
    np.fill_diagonal(sim_tr, -1)
    best_self = sim_tr.max(axis=1)

    print("\nnear-duplicate check (cosine similarity of 32x32 grayscale crops)")
    for thr in (0.99, 0.98, 0.95, 0.90):
        n_test = int((best_train >= thr).sum())
        n_within = int((best_self >= thr).sum())
        print(f"  >={thr:.2f}: test-with-train-match={n_test:4d}/{len(test)}   "
              f"train-with-train-match={n_within:4d}/{len(train)}")

    order = np.argsort(-best_train)
    print("\nmost test-similar pairs (test id -> nearest train image, similarity):")
    for i in order[:12]:
        j = arg_train[i]
        print(f"  {os.path.basename(test[i]):>16s} -> "
              f"{P.CLASSES[[y for p, y in P.scan_train()][j]]:<18s} "
              f"{os.path.basename(train[j]):>16s}  sim={best_train[i]:.4f}")
    print(f"\nmedian best-match similarity: {np.median(best_train):.4f}")


if __name__ == "__main__":
    main()
