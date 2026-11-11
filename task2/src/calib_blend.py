"""Temperature-calibrate each group on OOF, then re-run the bagged blend.

Groups trained with different backbones/resolutions end up with different
confidence sharpness.  Averaging raw probabilities lets the sharpest group
dominate.  Fitting one temperature per group (minimising OOF negative log
likelihood) equalises them before blending.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds  # noqa: E402
import ensemble as E  # noqa: E402
import blend_search as B  # noqa: E402


def fit_temp(p, y):
    logp = np.log(np.maximum(p, 1e-9))

    def nll(logt):
        t = np.exp(logt)
        z = logp / t
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1, keepdims=True))
        return float(-(z[np.arange(len(y)), y] - lse[:, 0]).mean())

    r = minimize_scalar(nll, bounds=(-1.2, 1.2), method="bounded")
    return float(np.exp(r.x)), float(r.fun)


def calib_probs(p, t):
    logp = np.log(np.maximum(p, 1e-9)) / t
    logp -= logp.max(1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(1, keepdims=True)


def main():
    dev = ds.Device("cpu")
    y = dev.train_y.numpy().astype(np.int64)
    runs = [r for r in E.scan_runs() if r["has_test"]]
    groups = E.build_group_oof(runs, len(y))
    groups = {k: g for k, g in groups.items() if g["mask"].all()}

    cal = {}
    print(f"{'group':<26}{'T':>7}{'NLL_before':>12}{'NLL_after':>11}")
    for k, g in groups.items():
        p = g["probs"]
        t, after = fit_temp(p, y)
        logp = np.log(np.maximum(p, 1e-9))
        z = logp - logp.max(1, keepdims=True)
        before = float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1, keepdims=True))[:, 0]).mean())
        cal[k] = {"probs": calib_probs(p, t), "mask": g["mask"], "T": t}
        print(f"{k:<26}{t:>7.3f}{before:>12.4f}{after:>11.4f}")

    # reference: uncalibrated bagged blend
    ref = json.load(open(os.path.join(ds.OUT_DIR, "blend_search.json"), encoding="utf-8"))
    ref_best = max(v["oof_prior"] for v in ref.values())
    print(f"\nuncalibrated best blend OOF: {ref_best:.5f}")

    w = B.bagged_greedy(cal, y)
    ok, p = B.combine(cal, w, len(y))
    acc = float((p[ok].argmax(1) == y[ok]).mean())
    prior = np.bincount(y, minlength=ds.N_CLS).astype(np.float64)
    prior /= prior.sum()
    tau, acc_tau = 0.0, acc
    for t in (0.05, 0.1, 0.2, 0.3):
        adj = p + t * np.log(prior)[None, :]
        a = float((adj[ok].argmax(1) == y[ok]).mean())
        if a > acc_tau:
            acc_tau, tau = a, t
    print(f"calibrated bagged blend OOF: {acc:.5f}   +prior(tau={tau}): {acc_tau:.5f}")
    print(f"delta vs uncalibrated: {acc_tau - ref_best:+.5f}")
    print("weights:", {k: round(v, 3) for k, v in sorted(w.items())})

    if acc_tau <= ref_best:
        print("\nno improvement -- no submission written")
        return

    test = np.zeros((dev.n_test, ds.N_CLS))
    tot = 0.0
    for k, ww in w.items():
        members = [r for r in runs if r["group"] == k]
        sub = np.mean([np.load(os.path.join(r["dir"], "test_probs.npy")) for r in members], axis=0)
        sub = calib_probs(sub, cal[k]["T"])  # same temperature as fitted on OOF
        test += ww * sub
        tot += ww
    test /= tot
    if tau > 0:
        test = np.log(np.maximum(test, 1e-9)) + tau * np.log(prior)[None, :]
    pred = test.argmax(1)
    sample = pd.read_csv(ds.SAMPLE_SUB)
    lut = {i: int(c) for i, c in zip(dev.test_ids, pred)}
    out = pd.DataFrame({"ID": sample["ID"], "Emotion": [lut[i] for i in sample["ID"]]})
    path = os.path.join(ds.SUB_DIR, "sub_calib.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
