"""Test-time inference with TTA for a trained run.

Writes outputs/runs/<tag>/test_probs.npy  (softmax averaged over
{scales} x {identity, horizontal flip}).

    python src/predict.py --tag ctn224_f0 [--res 224] [--batch 512]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds  # noqa: E402
import models as M  # noqa: E402


def make_input(px_u8, res, mean, std, in_ch, scale=1.0):
    x = px_u8.view(-1, 1, ds.IMG_SIZE, ds.IMG_SIZE).float().div_(255.0)
    r = int(round(res * scale))
    x = F.interpolate(x, size=(r, r), mode="bilinear", align_corners=False)
    if r != res:
        if r > res:
            o = (r - res) // 2
            x = x[:, :, o : o + res, o : o + res]
        else:
            p = (res - r + 1) // 2
            x = F.pad(x, (p, res - r - p, p, res - r - p), value=0.0)
    x = (x - mean) / std
    if in_ch == 3:
        x = x.repeat(1, 3, 1, 1)
    return x


@torch.no_grad()
def tta_probs(model, px_u8, res, mean, std, in_ch, batch=512, scales=(0.9, 1.0, 1.1)):
    model.eval()
    out = torch.zeros(px_u8.shape[0], ds.N_CLS, device=px_u8.device)
    for s in scales:
        for flip in (False, True):
            for i in range(0, px_u8.shape[0], batch):
                chunk = px_u8[i : i + batch]
                x = make_input(chunk, res, mean, std, in_ch, s)
                if flip:
                    x = torch.flip(x, dims=[3])
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                out[i : i + chunk.shape[0]] += F.softmax(logits.float(), dim=1)
    out /= len(scales) * 2
    return out


def load_run(tag, device="cuda"):
    run_dir = os.path.join(ds.RUNS_DIR, tag)
    if not os.path.exists(os.path.join(run_dir, "best.pt")):
        raise FileNotFoundError(os.path.join(run_dir, "best.pt"))
    with open(os.path.join(run_dir, "config.json")) as fh:
        cfg = json.load(fh)
    model, in_ch, _ = M.build_model(
        cfg["backbone"], num_classes=ds.N_CLS, drop_path=cfg.get("drop_path", 0.1), pretrained=False
    )
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))
    model.to(device)
    model.eval()
    return model, in_ch, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--batch", type=int, default=0, help="0 = auto from resolution")
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    run_dir = os.path.join(ds.RUNS_DIR, a.tag)
    out_path = os.path.join(run_dir, "test_probs.npy")
    if os.path.exists(out_path) and not a.overwrite:
        print(f"[skip] {out_path} exists")
        return

    device = "cuda"
    dev = ds.Device(device)
    mean, std = ds.norm_stats()
    model, in_ch, cfg = load_run(a.tag, device)
    res = cfg["res"]
    batch = a.batch or ds.eval_batch_for(res)
    scales = (1.0,) if a.no_tta else (0.9, 1.0, 1.1)
    probs = tta_probs(model, dev.test_px, res, mean, std, in_ch, batch, scales)
    np.save(out_path, probs.cpu().numpy())
    acc_cols = probs.argmax(1).bincount(minlength=6).cpu().tolist()
    print(f"[{a.tag}] wrote {out_path}  scales={scales}  pred histogram={acc_cols}")


if __name__ == "__main__":
    main()
