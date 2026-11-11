"""Combine runs, then decide whether a class-prior correction is justified.

The training set is class balanced (100 per class) but the test set is not: its class
proportions follow the original public database.  Under an accuracy metric the optimal
decision is ``argmax_c  pi_c * p(c|x)``, so ignoring the shift costs accuracy.

The prior is estimated by inverting the out-of-fold confusion matrix, and the whole
procedure is validated by resampling the out-of-fold set to a known shifted prior.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import nnls
from sklearn.metrics import accuracy_score, confusion_matrix

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402

# Class proportions of the five species in the original public database (train split).
POOL_PRIOR = np.array([263, 221, 654, 607, 385], dtype=np.float64)
POOL_PRIOR /= POOL_PRIOR.sum()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="b0_288,cnx320")
    ap.add_argument("--test-probs",
                    default=os.path.join(P.OUT_DIR, "test_probs.npz"))
    ap.add_argument("--out", default=os.path.join(P.OUT_DIR, "blend.json"))
    return ap.parse_args()


def load_oof(tags):
    probs, labels = None, None
    for tag in tags:
        data = np.load(os.path.join(P.OUT_DIR, tag, "oof.npz"))
        p, seen, lab = data["probs"], data["seen"], data["labels"]
        if not seen.all():
            raise SystemExit(f"{tag} has incomplete out-of-fold coverage")
        probs = p if probs is None else probs + p
        labels = lab if labels is None else labels
    return probs / len(tags), labels


def invert_prior(pred_dist: np.ndarray, confusion: np.ndarray) -> np.ndarray:
    """Solve  A^T pi = q  for a non-negative pi summing to one."""
    n = len(pred_dist)
    a = np.vstack([confusion.T, 20.0 * np.ones((1, n))])
    b = np.concatenate([pred_dist, [20.0]])
    pi, _ = nnls(a, b)
    total = pi.sum()
    return pi / total if total > 0 else np.full(n, 1.0 / n)


def main() -> None:
    args = parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    probs, labels = load_oof(tags)
    pred = probs.argmax(axis=1)
    print(f"tags: {tags}")
    print(f"out-of-fold accuracy: {accuracy_score(labels, pred):.4f}  "
          f"(n={len(labels)})")
    cm = confusion_matrix(labels, pred)
    conf = cm / cm.sum(axis=1, keepdims=True)
    print("per-class recall:",
          {P.CLASSES[i]: round(float(conf[i, i]), 3) for i in range(len(P.CLASSES))})

    # ---- validate the estimator on resampled, deliberately shifted out-of-fold data
    rng = np.random.default_rng(0)
    n_test = 378
    print("\nvalidating prior estimation on resampled out-of-fold data:")
    for name, target in (("pool prior", POOL_PRIOR), ("uniform", np.full(5, 0.2))):
        est, acc_raw, acc_corr = [], [], []
        for _ in range(40):
            idx = []
            for c in range(5):
                k = int(round(target[c] * n_test))
                pool = np.where(labels == c)[0]
                idx += list(rng.choice(pool, size=k, replace=True))
            idx = np.array(idx)
            q = np.bincount(pred[idx], minlength=5) / len(idx)
            est.append(invert_prior(q, conf))
            acc_raw.append(accuracy_score(labels[idx], pred[idx]))
            acc_corr.append(accuracy_score(labels[idx], (probs[idx] * target).argmax(1)))
        est = np.mean(est, axis=0)
        print(f"  true {name:10s} {np.round(target, 3).tolist()}  "
              f"-> estimated {np.round(est, 3).tolist()}")
        print(f"       accuracy with uniform rule {np.mean(acc_raw):.4f} vs "
              f"oracle-prior rule {np.mean(acc_corr):.4f}")

    # ---- estimate the prior of the real test set and save the decision rule
    data = np.load(args.test_probs, allow_pickle=True)
    test_probs = data["probs"]
    q = np.bincount(test_probs.argmax(axis=1), minlength=5) / len(test_probs)
    pi_hat = invert_prior(q, conf)
    print(f"\ntest predicted distribution : {np.round(q, 3).tolist()}")
    print(f"pool prior for reference    : {np.round(POOL_PRIOR, 3).tolist()}")
    print(f"estimated test prior (NNLS) : {np.round(pi_hat, 3).tolist()}")

    rule = "prior" if np.abs(pi_hat - 0.2).max() > 0.04 else "uniform"
    print(f"decision rule selected      : {rule}")
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"tags": tags, "estimated_prior": pi_hat.tolist(),
                   "rule": rule, "oof_accuracy": float(accuracy_score(labels, pred)),
                   "confusion": cm.tolist(), "classes": P.CLASSES}, fh, indent=2)
    print("blend config written to", os.path.relpath(args.out, P.ROOT))


if __name__ == "__main__":
    main()
