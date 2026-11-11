"""Re-estimate BatchNorm statistics on the data to be predicted (no training).

The models were trained on folds; their BN running statistics therefore describe
the training population.  Recomputing them on the target batch -- a standard
transductive trick -- can help when the evaluation batch has its own statistics.

Protocol: for fold f, adapt BN on fold f's validation images and evaluate on the
same images.  This mirrors the real submission setting exactly (adapt on the test
set, then predict it).

    python src/bn_adapt.py --tag cns224_f0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds  # noqa: E402
import models as M  # noqa: E402
import predict as P  # noqa: E402


@torch.no_grad()
def run(model, px, res, mean, std, in_ch, batch=128):
    model.eval()
    out = torch.zeros(px.shape[0], ds.N_CLS, device=px.device)
    for i in range(0, px.shape[0], batch):
        x = P.make_input(px[i : i + batch], res, mean, std, in_ch, 1.0)
        with torch.amp.autocast("cuda"):
            out[i : i + x.shape[0]] = F.softmax(model(x).float(), dim=1)
    return out


@torch.no_grad()
def adapt_bn(model, px, res, mean, std, in_ch, batch=128, passes=2):
    bns = [m for m in model.modules() if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm))]
    print(f"    adapting {len(bns)} BN layers over {passes} passes")
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average
    model.train()
    for _ in range(passes):
        for i in range(0, px.shape[0], batch):
            x = P.make_input(px[i : i + batch], res, mean, std, in_ch, 1.0)
            with torch.amp.autocast("cuda"):
                model(x)
    model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="cns224_f0")
    a = p.parse_args()
    dev = ds.Device("cuda")
    mean, std = ds.norm_stats()
    model, in_ch, cfg = P.load_run(a.tag)
    res = cfg["res"]
    idx = np.load(os.path.join(ds.RUNS_DIR, a.tag, "val_idx.npy"))
    it = torch.from_numpy(idx).to("cuda")
    px, y = dev.train_px[it], dev.train_y[it]

    before = run(model, px, res, mean, std, in_ch)
    acc0 = float((before.argmax(1) == y).float().mean())
    print(f"[{a.tag}] no-adapt val acc = {acc0:.4f}")

    adapt_bn(model, px, res, mean, std, in_ch)
    after = run(model, px, res, mean, std, in_ch)
    acc1 = float((after.argmax(1) == y).float().mean())
    print(f"[{a.tag}] BN-adapted val acc = {acc1:.4f}  delta={acc1 - acc0:+.4f}")


if __name__ == "__main__":
    main()
