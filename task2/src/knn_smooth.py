"""Transductive kNN smoothing of the ensemble probabilities.

Images that look nearly identical should receive the same label, so predictions
are blended with a similarity-weighted average of their nearest neighbours.
No labels are used -- only image content and the model's own probabilities.

Validation protocol (deliberately conservative): neighbours are searched *within
one fold's validation split only*.  For a val image in fold f the prediction comes
from the model that never saw fold f, and every neighbour is likewise unseen by
that model, so no cross-fold information can leak into the accuracy estimate.
That mirrors the real setting, where all 7,065 test images are unseen.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds  # noqa: E402
import ensemble as E  # noqa: E402
import blend_search as B  # noqa: E402


def knn_matrix(px, k=10, chunk=1024):
    """Return (idx [n,k], sim [n,k]) for nearest neighbours of every row."""
    x = px.astype(np.float32)
    x /= 255.0
    sq = np.einsum("ij,ij->i", x, x)
    n = len(x)
    idx = np.empty((n, k), np.int64)
    sim = np.empty((n, k), np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = sq[s:e, None] + sq[None, :] - 2.0 * (x[s:e] @ x.T)
        np.maximum(d, 0, out=d)
        # exclude self
        d[np.arange(e - s), np.arange(s, e)] = np.inf
        part = np.argpartition(d, k - 1, axis=1)[:, :k]
        rows = np.arange(e - s)[:, None]
        vals = d[rows, part]
        o = np.argsort(vals, axis=1)
        idx[s:e] = part[rows, o]
        sim[s:e] = np.exp(-vals[rows, o] / 60000.0)  # scale from earlier distance stats
    return idx, sim


def smooth(p, idx, sim, alpha, power=1.0):
    w = sim ** power
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
    nb = np.einsum("nk,nkc->nc", w, p[idx])
    return (1 - alpha) * p + alpha * nb


def main():
    dev = ds.Device("cpu")
    y = dev.train_y.numpy().astype(np.int64)
    runs = [r for r in E.scan_runs() if r["has_test"]]
    groups = {k: g for k, g in E.build_group_oof(runs, len(y)).items() if g["mask"].all()}

    ref = json.load(open(os.path.join(ds.OUT_DIR, "blend_search.json"), encoding="utf-8"))
    name = max(ref, key=lambda k: ref[k]["oof_prior"])
    w = ref[name]["weights"]
    tau = ref[name]["tau"]
    prior = np.bincount(y, minlength=ds.N_CLS).astype(np.float64)
    prior /= prior.sum()
    lg = name.endswith("logit")
    base_groups = (
        {k: dict(v, probs=np.log(np.maximum(v["probs"], 1e-9))) for k, v in groups.items()}
        if lg else groups
    )
    _, P = B.combine(base_groups, w, len(y))
    # `combine` already returns log-space values when averaging log-probs
    logits = P if lg else np.log(np.maximum(P, 1e-9))
    if tau > 0:
        logits = logits + tau * np.log(prior)[None, :]
    P = np.exp(logits - logits.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    base_acc = float((P.argmax(1) == y).mean())
    print(f"reference blend '{name}': OOF={base_acc:.5f}")

    folds = ds.make_folds(5)
    px = dev.train_px.numpy()

    print("\nfold-wise kNN smoothing (neighbours restricted to the same val split):")
    print(f"{'k':>3}{'alpha':>7}{'acc':>9}{'delta':>9}")
    cache = {}
    best = (base_acc, 0, 0.0)
    for k in (1, 3, 5, 10):
        for va in folds:
            vi = np.asarray(va["val"])
            key = (k, vi[0])
            if key not in cache:
                cache[key] = knn_matrix(px[vi], k=k)
        for alpha in (0.1, 0.2, 0.3, 0.5, 0.7):
            correct = 0
            total = 0
            for va in folds:
                vi = np.asarray(va["val"])
                idx, sim = cache[(k, vi[0])]
                sp = smooth(P[vi], idx, sim, alpha)
                correct += int((sp.argmax(1) == y[vi]).sum())
                total += len(vi)
            acc = correct / total
            flag = ""
            if acc > best[0]:
                best = (acc, k, alpha)
                flag = "  <-"
            print(f"{k:>3}{alpha:>7.1f}{acc:>9.5f}{acc-base_acc:>+9.5f}{flag}")

    print(f"\nbest: k={best[1]} alpha={best[2]} acc={best[0]:.5f} (delta {best[0]-base_acc:+.5f})")
    if best[0] <= base_acc + 1e-4:
        print("kNN smoothing does not help -- no submission written")
        return

    # ---- apply to test ----
    k, alpha = best[1], best[2]
    test = np.zeros((dev.n_test, ds.N_CLS))
    tot = 0.0
    for kk, ww in w.items():
        members = [r for r in runs if r["group"] == kk]
        sub = np.mean([np.load(os.path.join(r["dir"], "test_probs.npy")) for r in members], axis=0)
        if lg:
            sub = np.log(np.maximum(sub, 1e-9))
        test += ww * sub
        tot += ww
    test /= tot
    if tau > 0:
        test = test + tau * np.log(prior)[None, :]
    test = np.exp(test - test.max(1, keepdims=True))
    test /= test.sum(1, keepdims=True)

    idx, sim = knn_matrix(dev.test_px.numpy(), k=k)
    sp = smooth(test, idx, sim, alpha)
    pred = sp.argmax(1)
    sample = pd.read_csv(ds.SAMPLE_SUB)
    lut = {i: int(c) for i, c in zip(dev.test_ids, pred)}
    out = pd.DataFrame({"ID": sample["ID"], "Emotion": [lut[i] for i in sample["ID"]]})
    path = os.path.join(ds.SUB_DIR, "sub_knn.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path}")
    with open(os.path.join(ds.OUT_DIR, "knn_smooth.json"), "w") as fh:
        json.dump({"base": base_acc, "best": best[0], "k": k, "alpha": alpha}, fh, indent=2)


if __name__ == "__main__":
    main()
