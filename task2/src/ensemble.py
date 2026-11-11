"""Blend runs into a submission.

Runs are grouped by backbone+resolution (each group is expected to cover the whole
training set with its 5 folds). Group weights are chosen by greedy forward selection
on out-of-fold accuracy, so nothing is hand-tuned. An optional prior-correction
exponent is also selected on OOF only.

    python src/ensemble.py --groups ctn224,efs224,swin224 --name stage2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds  # noqa: E402


def scan_runs():
    """Collect finished runs. Ablation / smoke tags are ignored so that single-fold
    experiments can never leak into the blend."""
    SKIP = ("abl_", "aug_", "smoke")
    runs = []
    for d in sorted(glob.glob(os.path.join(ds.RUNS_DIR, "*"))):
        if os.path.basename(d).startswith(SKIP):
            continue
        sp = os.path.join(d, "summary.json")
        vp = os.path.join(d, "val_probs.npy")
        vi = os.path.join(d, "val_idx.npy")
        tp = os.path.join(d, "test_probs.npy")
        if not (os.path.exists(sp) and os.path.exists(vp) and os.path.exists(vi)):
            continue
        s = json.load(open(sp))
        backbone = s["backbone"].split(".")[0]
        runs.append(
            {
                "tag": os.path.basename(d),
                "dir": d,
                "summary": s,
                "group": f"{backbone}_{s['res']}",
                "fold": s.get("fold", 0),
                "has_test": os.path.exists(tp),
            }
        )
    return runs


def build_group_oof(runs, n_train):
    groups = {}
    for r in runs:
        g = groups.setdefault(
            r["group"],
            {"probs": np.zeros((n_train, ds.N_CLS), np.float64), "mask": np.zeros(n_train, bool)},
        )
        idx = np.load(os.path.join(r["dir"], "val_idx.npy"))
        p = np.load(os.path.join(r["dir"], "val_probs.npy"))
        g["probs"][idx] = p
        g["mask"][idx] = True
    return groups


def oof_accuracy(groups, weights, y):
    keys = [k for k, w in weights.items() if w > 0]
    if not keys:
        return 0.0, None
    num = np.zeros((len(y), ds.N_CLS))
    den = np.zeros(len(y))
    for k in keys:
        w = weights[k]
        g = groups[k]
        num[g["mask"]] += w * g["probs"][g["mask"]]
        den[g["mask"]] += w
    ok = den > 0
    pred = np.full(len(y), -1)
    pred[ok] = num[ok].argmax(1)
    return float((pred[ok] == y[ok]).mean()), num / np.maximum(den, 1e-9)[:, None]


def greedy(groups, y, n_iter=24):
    keys = sorted(groups)
    counts = {k: 0 for k in keys}
    best_single, best_key = -1.0, None
    for k in keys:
        acc, _ = oof_accuracy(groups, {k: 1.0}, y)
        if acc > best_single:
            best_single, best_key = acc, k
    counts[best_key] = 1
    total = 1
    best_acc = best_single
    for _ in range(n_iter - 1):
        cand_key, cand_acc = None, best_acc
        for k in keys:
            trial = dict(counts)
            trial[k] += 1
            w = {kk: vv / (total + 1) for kk, vv in trial.items()}
            acc, _ = oof_accuracy(groups, w, y)
            if acc > cand_acc + 1e-9:
                cand_key, cand_acc = k, acc
        if cand_key is None:
            break
        counts[cand_key] += 1
        total += 1
        best_acc = cand_acc
    weights = {k: v / total for k, v in counts.items()}
    return {k: v for k, v in weights.items() if v > 0}, best_acc


def tune_prior(oof_probs, y, prior, mask, grid=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5)):
    """Choose the prior-correction exponent on OOF only, restricted to covered rows."""
    best_tau, best_acc = 0.0, -1.0
    logp = np.log(np.maximum(oof_probs, 1e-12))
    for tau in grid:
        adj = logp + tau * np.log(prior)[None, :]
        acc = float((adj[mask].argmax(1) == y[mask]).mean())
        if acc > best_acc:
            best_acc, best_tau = acc, tau
    return best_tau, best_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", default="", help="comma separated group names; default = all")
    p.add_argument("--name", default="blend")
    p.add_argument("--out", default="")
    p.add_argument("--allow-partial", action="store_true",
                   help="keep groups whose OOF coverage is < 100%% (their accuracy is "
                        "measured on a different subset, so the greedy comparison is biased)")
    a = p.parse_args()

    dev = ds.Device("cpu")
    y = dev.train_y.numpy().astype(np.int64)
    runs = scan_runs()
    if not runs:
        print("no runs with val_probs found")
        return
    groups = build_group_oof(runs, len(y))
    if a.groups:
        want = {g.strip() for g in a.groups.split(",") if g.strip()}
        groups = {k: v for k, v in groups.items() if k in want}
    partial = [k for k, v in groups.items() if v["mask"].mean() < 0.999]
    if partial and not a.allow_partial:
        print(f"[guard] 排除未跑满的组（如需保留加 --allow-partial）：{partial}")
        groups = {k: v for k, v in groups.items() if k not in partial}
    print("available groups:")
    for k in sorted(groups):
        cov = groups[k]["mask"].mean()
        single = oof_accuracy(groups, {k: 1.0}, y)[0]
        print(f"  {k:<24} coverage={cov:.3f}  single_oof={single:.5f}")

    weights, oof_acc = greedy(groups, y)
    print("\nweights:", {k: round(v, 3) for k, v in sorted(weights.items())})
    print(f"OOF accuracy (blend): {oof_acc:.5f}")
    _, oof_probs = oof_accuracy(groups, weights, y)

    prior = np.bincount(y, minlength=ds.N_CLS).astype(np.float64)
    prior /= prior.sum()
    covered = np.zeros(len(y), bool)
    for k, w in weights.items():
        if w > 0:
            covered |= groups[k]["mask"]
    tau, prior_acc = tune_prior(oof_probs, y, prior, covered)
    print(f"prior correction tau={tau}  OOF acc={prior_acc:.5f}")

    test = None
    used = []
    for k, w in sorted(weights.items()):
        members = [r for r in runs if r["group"] == k and r["has_test"]]
        if not members:
            print(f"  [warn] group {k} has no test_probs yet - skipped")
            continue
        sub = np.mean([np.load(os.path.join(r["dir"], "test_probs.npy")) for r in members], axis=0)
        test = sub * w if test is None else test + sub * w
        used.append((k, round(w, 3), len(members)))
    if test is None:
        print("no test probabilities available")
        return
    print("used:", used)
    if tau > 0:
        test = np.log(np.maximum(test, 1e-12)) + tau * np.log(prior)[None, :]
    pred = test.argmax(1)

    sample = pd.read_csv(ds.SAMPLE_SUB)
    lut = {i: int(c) for i, c in zip(dev.test_ids, pred)}
    out = pd.DataFrame({"ID": sample["ID"], "Emotion": [lut[i] for i in sample["ID"]]})
    path = a.out or os.path.join(ds.SUB_DIR, f"sub_{a.name}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\nwrote {path}  rows={len(out)}")
    print("predicted histogram:", out["Emotion"].value_counts().sort_index().to_dict())
    with open(os.path.join(ds.OUT_DIR, f"blend_{a.name}.json"), "w") as fh:
        json.dump(
            {
                "weights": weights,
                "oof_accuracy": oof_acc,
                "tau": tau,
                "oof_accuracy_prior": prior_acc,
                "submission": path,
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
