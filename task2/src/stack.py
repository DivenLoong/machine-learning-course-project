"""Stack the per-group OOF probabilities with a meta-classifier.

Weighted averaging assumes every group contributes linearly in probability space.
A meta-learner can instead learn per-class, per-group corrections.  To get an
unbiased estimate the meta-model is itself cross-validated on the 28,275 OOF rows
(features are already out-of-fold w.r.t. the base models).

    python src/stack.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds  # noqa: E402
import ensemble as E  # noqa: E402


def build_matrices(runs, groups, n_train, n_test, logit: bool):
    keys = sorted(groups)
    Xtr = np.zeros((n_train, len(keys) * ds.N_CLS), np.float32)
    Xte = np.zeros((n_test, len(keys) * ds.N_CLS), np.float32)
    for j, k in enumerate(keys):
        g = groups[k]
        p = g["probs"]
        if logit:
            p = np.log(np.maximum(p, 1e-6))
            p = p - p.mean(1, keepdims=True)
        Xtr[:, j * ds.N_CLS : (j + 1) * ds.N_CLS] = p

        members = [r for r in runs if r["group"] == k]
        t = np.mean([np.load(os.path.join(r["dir"], "test_probs.npy")) for r in members], axis=0)
        if logit:
            t = np.log(np.maximum(t, 1e-6))
            t = t - t.mean(1, keepdims=True)
        Xte[:, j * ds.N_CLS : (j + 1) * ds.N_CLS] = t
    return keys, Xtr, Xte


def cv_score(X, y, C, n_splits=5, seed=0, logit=True):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), np.int64)
    for tr, va in skf.split(X, y):
        clf = LogisticRegression(C=C, max_iter=2000, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        pred[va] = clf.predict(X[va])
    return float((pred == y).mean())


def main():
    dev = ds.Device("cpu")
    y = dev.train_y.numpy().astype(np.int64)
    runs = [r for r in E.scan_runs() if r["has_test"]]
    groups = E.build_group_oof(runs, len(y))
    groups = {k: g for k, g in groups.items() if g["mask"].all()}
    print(f"groups used: {len(groups)}")

    best_avg = 0.0
    for split_frac in (0.0,):
        pass
    # reference: the best simple blend found earlier
    ref = json.load(open(os.path.join(ds.OUT_DIR, "blend_search.json"), encoding="utf-8"))
    best_avg = max(v["oof_prior"] for v in ref.values())
    print(f"best simple blend OOF (reference): {best_avg:.5f}\n")

    results = {}
    for logit in (True, False):
        keys, Xtr, Xte = build_matrices(runs, groups, dev.n_train, dev.n_test, logit)
        for C in (0.03, 0.1, 0.3, 1.0, 3.0):
            acc = cv_score(Xtr, y, C, logit=logit)
            tag = "logit" if logit else "prob"
            results[(tag, C)] = acc
            print(f"  stack[{tag:<5}] C={C:<5} cv_acc={acc:.5f}")

    (btag, bC), bacc = max(results.items(), key=lambda kv: kv[1])
    print(f"\nbest stack: {btag} C={bC}  cv_acc={bacc:.5f}  delta_vs_blend={bacc-best_avg:+.5f}")

    if bacc <= best_avg:
        print("stacking does not beat the simple blend -- no submission written")
        return

    logit = btag == "logit"
    keys, Xtr, Xte = build_matrices(runs, groups, dev.n_train, dev.n_test, logit)
    clf = LogisticRegression(C=bC, max_iter=4000, n_jobs=-1).fit(Xtr, y)
    proba = clf.predict_proba(Xte)
    pred = proba.argmax(1)

    sample = pd.read_csv(ds.SAMPLE_SUB)
    lut = {i: int(c) for i, c in zip(dev.test_ids, pred)}
    out = pd.DataFrame({"ID": sample["ID"], "Emotion": [lut[i] for i in sample["ID"]]})
    path = os.path.join(ds.SUB_DIR, "sub_stack.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path}")
    with open(os.path.join(ds.OUT_DIR, "stack.json"), "w") as fh:
        json.dump({"cv_acc": bacc, "kind": btag, "C": bC, "groups": keys,
                   "reference_blend": best_avg}, fh, indent=2)


if __name__ == "__main__":
    main()
