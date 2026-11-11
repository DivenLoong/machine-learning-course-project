"""Search several blending strategies on OOF and emit the best submissions.

Strategies compared (all evaluated on the same 28,275-row OOF matrix):
  * plain greedy forward selection with replacement
  * bagged greedy  -- greedy on many random 60% subsamples, weights averaged
                      (Caruana et al., "Ensemble selection from libraries")
  * uniform over the top-k groups by single-group OOF
  * bagged greedy in logit space
Every candidate also gets a prior-correction exponent tuned on OOF.
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


def combine(groups, weights, n):
    num = np.zeros((n, ds.N_CLS))
    den = np.zeros(n)
    for k, w in weights.items():
        if w <= 0:
            continue
        g = groups[k]
        num[g["mask"]] += w * g["probs"][g["mask"]]
        den[g["mask"]] += w
    ok = den > 0
    out = np.zeros((n, ds.N_CLS))
    out[ok] = num[ok] / den[ok, None]
    return ok, out


def score(groups, weights, y, subset=None):
    ok, p = combine(groups, weights, len(y))
    if subset is not None:
        ok = ok & subset
    return float((p[ok].argmax(1) == y[ok]).mean())


def greedy(groups, y, idx=None, n_iter=24):
    keys = sorted(groups)
    sub = np.zeros(len(y), bool)
    sub[idx if idx is not None else np.arange(len(y))] = True
    counts = {k: 0 for k in keys}
    best_k, best_a = None, -1.0
    for k in keys:
        a = score(groups, {k: 1.0}, y, sub)
        if a > best_a:
            best_k, best_a = k, a
    counts[best_k] = 1
    total = 1
    for _ in range(n_iter - 1):
        cand, cand_a = None, best_a
        for k in keys:
            t = dict(counts)
            t[k] += 1
            a = score(groups, {kk: vv / (total + 1) for kk, vv in t.items()}, y, sub)
            if a > cand_a + 1e-9:
                cand, cand_a = k, a
        if cand is None:
            break
        counts[cand] += 1
        total += 1
        best_a = cand_a
    return {k: v / total for k, v in counts.items() if v > 0}


def bagged_greedy(groups, y, n_bags=24, frac=0.6, seed=0):
    rng = np.random.default_rng(seed)
    acc = {k: 0.0 for k in groups}
    for b in range(n_bags):
        idx = rng.choice(len(y), size=int(len(y) * frac), replace=False)
        w = greedy(groups, y, idx)
        for k, v in w.items():
            acc[k] += v
    tot = sum(acc.values())
    return {k: v / tot for k, v in acc.items() if v > 0}


def to_logit(p):
    return np.log(np.maximum(p, 1e-9))


def main():
    dev = ds.Device("cpu")
    y = dev.train_y.numpy().astype(np.int64)
    runs = [r for r in E.scan_runs() if r["has_test"]]
    groups = E.build_group_oof(runs, len(y))
    # keep only groups that cover every training row
    full = {k: g for k, g in groups.items() if g["mask"].all()}
    dropped = sorted(set(groups) - set(full))
    print(f"usable groups: {sorted(full)}")
    if dropped:
        print(f"dropped (partial coverage): {dropped}")

    prior = np.bincount(y, minlength=ds.N_CLS).astype(np.float64)
    prior /= prior.sum()

    singles = {k: score(full, {k: 1.0}, y) for k in full}
    print("\nsingle-group OOF:")
    for k, v in sorted(singles.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<26} {v:.5f}")

    cands = {}
    cands["greedy"] = greedy(full, y)
    cands["bagged"] = bagged_greedy(full, y)
    order = sorted(singles, key=singles.get, reverse=True)
    for k in range(2, min(7, len(order)) + 1):
        w = {g: 1.0 / k for g in order[:k]}
        cands[f"uniform_top{k}"] = w

    # logit-space bagged greedy
    lg = {k: dict(v, probs=np.log(np.maximum(v["probs"], 1e-9))) for k, v in full.items()}
    cands["bagged_logit"] = bagged_greedy(lg, y, seed=7)

    results = {}
    for name, w in cands.items():
        g = lg if name.endswith("logit") else full
        base = score(g, w, y)
        ok, p = combine(g, w, len(y))
        tau, best = 0.0, base
        for t in (0.05, 0.1, 0.2, 0.3, 0.5):
            adj = p + t * np.log(prior)[None, :]
            a = float((adj[ok].argmax(1) == y[ok]).mean())
            if a > best:
                best, tau = a, t
        results[name] = {"oof": base, "oof_prior": best, "tau": tau, "weights": w}
        print(f"\n{name:<16} OOF={base:.5f}  +prior(tau={tau})={best:.5f}  "
              f"groups={len(w)}")

    best_name = max(results, key=lambda k: results[k]["oof_prior"])
    print(f"\n=== best by OOF: {best_name} ===")

    sample = pd.read_csv(ds.SAMPLE_SUB)
    emitted = []
    for name in sorted(results, key=lambda k: -results[k]["oof_prior"]):
        r = results[name]
        g = lg if name.endswith("logit") else full
        test = np.zeros((dev.n_test, ds.N_CLS))
        tot = 0.0
        for k, w in r["weights"].items():
            members = [x for x in runs if x["group"] == k]
            sub = np.mean([np.load(os.path.join(x["dir"], "test_probs.npy")) for x in members], axis=0)
            if name.endswith("logit"):
                sub = np.log(np.maximum(sub, 1e-9))
            test += w * sub
            tot += w
        test /= tot
        if r["tau"] > 0:
            test = test + r["tau"] * np.log(prior)[None, :]
        pred = test.argmax(1)
        lut = {i: int(c) for i, c in zip(dev.test_ids, pred)}
        out = pd.DataFrame({"ID": sample["ID"], "Emotion": [lut[i] for i in sample["ID"]]})
        path = os.path.join(ds.SUB_DIR, f"sub_{name}.csv")
        out.to_csv(path, index=False)
        emitted.append((name, r["oof_prior"], path))
        print(f"wrote {path}")

    with open(os.path.join(ds.OUT_DIR, "blend_search.json"), "w") as fh:
        json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}, fh, indent=2)
    print("\nranked:")
    for n, a, p in sorted(emitted, key=lambda t: -t[1]):
        print(f"  {n:<16} {a:.5f}  {p}")


if __name__ == "__main__":
    main()
