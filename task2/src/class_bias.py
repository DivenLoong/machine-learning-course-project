"""Tune a per-class multiplicative bias on the ensemble probabilities.

Six classes, five free parameters.  Because that is enough to overfit 28k rows,
the gain is measured with nested cross-validation on the OOF matrix and the
submission is only written when the *cross-validated* gain is positive.
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


def search(p, y, grid=np.arange(-0.3, 0.31, 0.05), rounds=3):
    b = np.zeros(ds.N_CLS)
    best = float((p.argmax(1) == y).mean())
    for _ in range(rounds):
        improved = False
        for c in range(ds.N_CLS):
            cur = b[c]
            for v in grid:
                b[c] = cur + v
                acc = float(((p + b[None, :]).argmax(1) == y).mean())
                if acc > best + 1e-9:
                    best, cur = acc, b[c]
                    improved = True
            b[c] = cur
        if not improved:
            break
    return b, best


def main():
    dev = ds.Device("cpu")
    y = dev.train_y.numpy().astype(np.int64)
    runs = [r for r in E.scan_runs() if r["has_test"]]
    groups = {k: g for k, g in E.build_group_oof(runs, len(y)).items() if g["mask"].all()}
    ref = json.load(open(os.path.join(ds.OUT_DIR, "blend_search.json"), encoding="utf-8"))
    name = max(ref, key=lambda k: ref[k]["oof_prior"])
    w, tau = ref[name]["weights"], ref[name]["tau"]
    lg = name.endswith("logit")
    g = {k: dict(v, probs=np.log(np.maximum(v["probs"], 1e-9))) for k, v in groups.items()} if lg else groups
    _, P = B.combine(g, w, len(y))
    logits = P if lg else np.log(np.maximum(P, 1e-9))
    prior = np.bincount(y, minlength=ds.N_CLS).astype(np.float64)
    prior /= prior.sum()
    if tau > 0:
        logits = logits + tau * np.log(prior)[None, :]
    base = float((logits.argmax(1) == y).mean())
    print(f"reference '{name}' OOF={base:.5f}")

    # nested CV: tune on 4/5, score on 1/5
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    correct = 0
    for tr, va in skf.split(logits, y):
        b, _ = search(logits[tr], y[tr])
        correct += int(((logits[va] + b[None, :]).argmax(1) == y[va]).sum())
    cv = correct / len(y)
    print(f"nested-CV acc with per-class bias = {cv:.5f}  delta={cv - base:+.5f}")

    if cv <= base + 2e-4:
        print("per-class bias does not generalise -- no submission written")
        return

    b, fit_acc = search(logits, y)
    print("biases:", np.round(b, 3).tolist(), f"in-sample={fit_acc:.5f}")

    test = np.zeros((dev.n_test, ds.N_CLS))
    tot = 0.0
    for k, ww in w.items():
        m = [r for r in runs if r["group"] == k]
        sub = np.mean([np.load(os.path.join(r["dir"], "test_probs.npy")) for r in m], axis=0)
        if lg:
            sub = np.log(np.maximum(sub, 1e-9))
        test += ww * sub
        tot += ww
    test /= tot
    if tau > 0:
        test = test + tau * np.log(prior)[None, :]
    pred = (test + b[None, :]).argmax(1)
    sample = pd.read_csv(ds.SAMPLE_SUB)
    lut = {i: int(c) for i, c in zip(dev.test_ids, pred)}
    out = pd.DataFrame({"ID": sample["ID"], "Emotion": [lut[i] for i in sample["ID"]]})
    path = os.path.join(ds.SUB_DIR, "sub_classbias.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
