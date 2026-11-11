"""Diagnostics for the pure-modelling pipeline.

Answers four questions with the data already on disk:
  1. Can the five leaderboard scores actually be distinguished from each other?
  2. Which ensemble members contribute, and which are redundant?
  3. Where exactly are the remaining out-of-fold errors?
  4. Do the models agree on what they get wrong (i.e. is there real diversity)?
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402

# ---------------------------------------------------------------- submissions
# (name, models, out-of-fold accuracy, leaderboard score) as measured on Kaggle.
SUBMISSIONS = [
    ("v1  10 models, no prior", 10, 0.9140, 0.9285),
    ("intermediate 15 models", 15, 0.9280, 0.9470),
    ("v3  30 models (final)", 30, 0.9300, 0.9523),
    ("v5  30 models + expert", 30, 0.9360, 0.9470),
    ("v7  40 models + expert", 40, 0.9360, 0.9444),
]
TEST_N = 378


def binomial_se(p: float, n: int) -> float:
    return float(np.sqrt(p * (1 - p) / n))


def section_leaderboard() -> None:
    print("=" * 78)
    print("1. Can the leaderboard scores be distinguished?")
    print("=" * 78)
    se = binomial_se(0.95, TEST_N)
    print(f"One test image is worth 1/{TEST_N} = {1 / TEST_N:.4f}.")
    print(f"At p ~ 0.95 the standard error on {TEST_N} test images is "
          f"{se:.4f} ({se / (1 / TEST_N):.1f} images).")
    print(f"95% confidence interval width: +-{1.96 * se:.4f} "
          f"(+-{1.96 * se * TEST_N:.1f} images)\n")
    print(f"{'submission':<26s} {'OOF':>7s} {'LB':>7s} {'errors':>7s}")
    for name, _n, oof, lb in SUBMISSIONS:
        errors = round((1 - lb) * TEST_N)
        print(f"{name:<26s} {oof:>7.4f} {lb:>7.4f} {errors:>7d}")
    best = max(SUBMISSIONS, key=lambda s: s[3])
    print(f"\nBest is {best[0]} at {best[3]:.4f}.")
    for name, _n, _oof, lb in SUBMISSIONS:
        if name == best[0]:
            continue
        diff = best[3] - lb
        print(f"  vs {name:<26s} delta={diff:.4f} "
              f"({diff * TEST_N:.1f} images, {diff / se:.2f} SE)"
              f"{'  <- not significant' if diff < 1.96 * se else ''}")
    print("\nOOF vs LB rank correlation:")
    oofs = [s[2] for s in SUBMISSIONS]
    lbs = [s[3] for s in SUBMISSIONS]
    print(f"  Spearman rho = {spearman(oofs, lbs):+.2f}  "
          f"(5 points, no statistical power)")


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb)))


# ---------------------------------------------------------------- ablation
def load_oof(tag: str):
    data = np.load(os.path.join(P.OUT_DIR, tag, "oof.npz"))
    return data["probs"].astype(np.float64), data["labels"]


def section_ablation(tags: list[str]) -> dict[str, np.ndarray]:
    print("\n" + "=" * 78)
    print("2. Leave-one-out contribution of each ensemble member")
    print("=" * 78)
    probs, labels = {}, None
    for tag in tags:
        probs[tag], labels = load_oof(tag)
    full = np.mean([probs[t] for t in tags], axis=0)
    base = accuracy_score(labels, full.argmax(1))
    print(f"full ensemble ({len(tags)} members): OOF = {base:.4f}")
    print(f"\n{'removed member':<12s} {'OOF without it':>15s} {'delta':>8s}")
    for tag in tags:
        sub = np.mean([probs[t] for t in tags if t != tag], axis=0)
        acc = accuracy_score(labels, sub.argmax(1))
        print(f"{tag:<12s} {acc:>15.4f} {acc - base:>+8.4f}")
    print("\nsingle-member OOF for reference:")
    for tag in tags:
        print(f"  {tag:<12s} {accuracy_score(labels, probs[tag].argmax(1)):.4f}")
    return probs


def section_diversity(probs: dict[str, np.ndarray], labels: np.ndarray) -> None:
    print("\n" + "=" * 78)
    print("3. Do the members fail on the same images?")
    print("=" * 78)
    tags = list(probs)
    wrong = {t: set(np.where(probs[t].argmax(1) != labels)[0]) for t in tags}
    full = np.mean([probs[t] for t in tags], axis=0)
    ens_wrong = set(np.where(full.argmax(1) != labels)[0])
    print(f"ensemble errors: {len(ens_wrong)}")
    union = set().union(*wrong.values())
    print(f"images wrong in at least one member: {len(union)}")
    print(f"images wrong in every member: {len(set.intersection(*wrong.values()))}")
    print(f"\nerrors fixed by the ensemble (wrong in some member, right in ensemble): "
          f"{len(union - ens_wrong)}")
    print(f"errors introduced by the ensemble (right everywhere, wrong in ensemble): "
          f"{len(ens_wrong - union)}")
    print("\npairwise error-set overlap (Jaccard):")
    print("        " + "".join(f"{t[:9]:>11s}" for t in tags))
    for a in tags:
        row = f"{a:<8s}"
        for b in tags:
            wa, wb = wrong[a], wrong[b]
            j = len(wa & wb) / max(len(wa | wb), 1)
            row += f"{j:>11.2f}"
        print(row)


def section_errors(probs: dict[str, np.ndarray], labels: np.ndarray) -> None:
    print("\n" + "=" * 78)
    print("4. Where are the remaining errors?")
    print("=" * 78)
    full = np.mean([probs[t] for t in probs], axis=0)
    pred = full.argmax(1)
    cm = confusion_matrix(labels, pred)
    print("confusion counts (rows = truth):")
    print("                  " + "".join(f"{c[:9]:>11s}" for c in P.CLASSES))
    for i, cls in enumerate(P.CLASSES):
        print(f"{cls:<18s}" + "".join(f"{v:>11d}" for v in cm[i]))

    conf = full.max(axis=1)
    correct = pred == labels
    print(f"\nmean confidence: correct {conf[correct].mean():.3f}, "
          f"wrong {conf[~correct].mean():.3f}")
    for thr in (0.5, 0.7, 0.9, 0.99):
        keep = conf >= thr
        if keep.sum() == 0:
            continue
        print(f"  confidence >= {thr:.2f}: covers {keep.sum():3d}/{len(labels)} "
              f"({keep.mean():5.1%}) of images, accuracy on them "
              f"{correct[keep].mean():.4f}")
    print(f"\nimages with confidence < 0.5: {int((conf < 0.5).sum())}")


def main() -> None:
    section_leaderboard()
    tags = ["b0_288", "cnx320", "cnx320p", "cnx320m", "cnx384m", "cnxs320m"]
    probs = section_ablation(tags)
    _, labels = load_oof(tags[0])
    section_diversity(probs, labels)
    section_errors(probs, labels)


if __name__ == "__main__":
    main()
