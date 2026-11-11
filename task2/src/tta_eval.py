"""Measure how much each TTA configuration actually buys, on the validation split.

OOF probabilities stored by train.py are TTA-free, so the TTA gain has never been
measured directly -- only inferred from the leaderboard.  This script closes that gap.

    python src/tta_eval.py --tag cns224_f0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds  # noqa: E402
import predict as P  # noqa: E402

CONFIGS = {
    "none      (1.00)": (1.0,),
    "current   (3 scales)": (0.9, 1.0, 1.1),
    "5 scales": (0.85, 0.925, 1.0, 1.075, 1.15),
    "7 scales": (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15),
}


@torch.no_grad()
def probs_for(model, px, res, mean, std, in_ch, scales, batch=256):
    acc = torch.zeros(px.shape[0], ds.N_CLS, device=px.device)
    n = 0
    for s in scales:
        for flip in (False, True):
            n += 1
            for i in range(0, px.shape[0], batch):
                x = P.make_input(px[i : i + batch], res, mean, std, in_ch, s)
                if flip:
                    x = torch.flip(x, dims=[3])
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                acc[i : i + x.shape[0]] += F.softmax(logits.float(), dim=1)
    return acc / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="cns224_f0")
    a = p.parse_args()

    dev = ds.Device("cuda")
    mean, std = ds.norm_stats()
    model, in_ch, cfg = P.load_run(a.tag)
    res = cfg["res"]

    run_dir = os.path.join(ds.RUNS_DIR, a.tag)
    idx = np.load(os.path.join(run_dir, "val_idx.npy"))
    idx_t = torch.from_numpy(idx).to("cuda")
    px = dev.train_px[idx_t]
    y = dev.train_y[idx_t]

    print(f"[{a.tag}] res={res} val_n={len(idx)}")
    base = None
    for name, scales in CONFIGS.items():
        pr = probs_for(model, px, res, mean, std, in_ch, scales)
        acc = float((pr.argmax(1) == y).float().mean())
        if base is None:
            base = acc
        print(f"  {name:<22} n_passes={len(scales)*2:>2}  acc={acc:.4f}  delta={acc-base:+.4f}", flush=True)


if __name__ == "__main__":
    main()
