"""Test whether weight-space averaging (model soup) beats prediction averaging.

Fair protocol: build the soup from four folds and evaluate on the held-out fifth
fold only, so no evaluated image was seen by any soup member.  Prediction
averaging is the reference (the group's 5-fold OOF accuracy).

    python src/model_soup.py --group convnext_small_224
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
import models as M  # noqa: E402
import predict as P  # noqa: E402


def load_state(run_dir):
    return torch.load(os.path.join(run_dir, "best.pt"), map_location="cuda")


@torch.no_grad()
def evaluate(model, px, res, mean, std, in_ch, batch=128):
    model.eval()
    out = torch.zeros(px.shape[0], ds.N_CLS, device=px.device)
    for i in range(0, px.shape[0], batch):
        x = P.make_input(px[i : i + batch], res, mean, std, in_ch, 1.0)
        with torch.amp.autocast("cuda"):
            out[i : i + x.shape[0]] = F.softmax(model(x).float(), dim=1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="convnext_small_224")
    a = p.parse_args()

    dev = ds.Device("cuda")
    mean, std = ds.norm_stats()
    folds = ds.make_folds(5)
    runs = []
    for f in range(5):
        d = os.path.join(ds.RUNS_DIR, f"{a.group.split('_')[0][:8]}{a.group.split('_')[-1]}_f{f}")
        # resolve the run directory that belongs to this group/fold
        runs.append(None)
    # map by reading summaries
    import ensemble as E
    all_runs = [r for r in E.scan_runs() if r["group"] == a.group]
    by_fold = {r["fold"]: r for r in all_runs}
    if len(by_fold) < 5:
        print(f"group {a.group} has only {len(by_fold)} folds")
        return
    cfg = by_fold[0]["summary"]
    res = cfg["res"]
    model, in_ch, _ = M.build_model(cfg["backbone"], num_classes=ds.N_CLS,
                                    drop_path=cfg.get("drop_path", 0.1), pretrained=False)
    model = model.to("cuda")

    # reference: prediction averaging per fold
    ref_correct = 0
    ref_total = 0
    for f in range(5):
        idx = np.load(os.path.join(by_fold[f]["dir"], "val_idx.npy"))
        vp = np.load(os.path.join(by_fold[f]["dir"], "val_probs.npy"))
        yy = dev.train_y[torch.from_numpy(idx).to("cuda")].cpu().numpy()
        ref_correct += int((vp.argmax(1) == yy).sum())
        ref_total += len(yy)
    print(f"[{a.group}] prediction-averaging OOF = {ref_correct/ref_total:.5f}")

    soup_correct, soup_total = 0, 0
    for h in range(5):
        members = [f for f in range(5) if f != h]
        acc_state = None
        for f in members:
            st = load_state(by_fold[f]["dir"])
            if acc_state is None:
                acc_state = {k: v.float().clone() if v.dtype.is_floating_point else v.clone()
                             for k, v in st.items()}
            else:
                for k, v in st.items():
                    if acc_state[k].dtype.is_floating_point:
                        acc_state[k] += v.float()
        for k in acc_state:
            if acc_state[k].dtype.is_floating_point:
                acc_state[k] /= len(members)
        model.load_state_dict({k: v for k, v in acc_state.items()})
        idx = np.load(os.path.join(by_fold[h]["dir"], "val_idx.npy"))
        it = torch.from_numpy(idx).to("cuda")
        px = dev.train_px[it]
        yy = dev.train_y[it]
        pr = evaluate(model, px, res, mean, std, in_ch)
        soup_correct += int((pr.argmax(1) == yy).sum())
        soup_total += len(idx)
        print(f"  leave-fold-{h}-out soup acc = {float((pr.argmax(1)==yy).float().mean()):.5f}", flush=True)
    print(f"[{a.group}] model soup OOF = {soup_correct/soup_total:.5f}  "
          f"delta vs prediction avg = {soup_correct/soup_total - ref_correct/ref_total:+.5f}")


if __name__ == "__main__":
    main()
