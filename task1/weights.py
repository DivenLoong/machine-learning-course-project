"""Bagged ensemble selection (Caruana et al.) over the out-of-fold probabilities.

Equal weighting is only optimal when every run is equally good.  Runs differ a lot
(0.88 to 0.93 out-of-fold), and simply averaging in a weak run measurably hurt the
blend, so weights are chosen greedily *with replacement* and averaged over several
bootstrap subsets of the out-of-fold data to limit selection overfitting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--bags", type=int, default=12)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--bag-fraction", type=float, default=0.7)
    ap.add_argument("--out", default=os.path.join(P.OUT_DIR, "weights.json"))
    return ap.parse_args()


def load(tags):
    probs, labels = [], None
    for tag in tags:
        data = np.load(os.path.join(P.OUT_DIR, tag, "oof.npz"))
        p, seen, lab = data["probs"], data["seen"], data["labels"]
        if not seen.all():
            raise SystemExit(f"{tag}: incomplete out-of-fold coverage")
        if labels is not None and not np.array_equal(labels, lab):
            raise SystemExit(f"{tag}: label order differs from the other runs")
        probs.append(p.astype(np.float64))
        labels = lab
    return np.stack(probs), labels


def main() -> None:
    args = parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    stack, labels = load(tags)
    n_models, n_samples = stack.shape[0], stack.shape[1]
    print(f"runs: {len(tags)}, samples: {n_samples}")

    for i, tag in enumerate(tags):
        print(f"  {tag:<10s} accuracy={accuracy_score(labels, stack[i].argmax(1)):.4f}  "
              f"logloss={log_loss(labels, stack[i], labels=list(range(len(P.CLASSES)))):.4f}")

    rng = np.random.default_rng(0)
    counts = np.zeros(n_models)
    for bag in range(args.bags):
        size = int(args.bag_fraction * n_samples)
        idx = rng.choice(n_samples, size=size, replace=False)
        y = labels[idx]
        total = np.zeros((size, len(P.CLASSES)))
        chosen = 0
        for _ in range(args.steps):
            best_tag, best_loss = None, np.inf
            for m in range(n_models):
                cand = (total + stack[m][idx]) / (chosen + 1)
                loss = log_loss(y, cand, labels=list(range(len(P.CLASSES))))
                if loss < best_loss:
                    best_tag, best_loss = m, loss
            total += stack[best_tag][idx]
            chosen += 1
            counts[best_tag] += 1

    weights = counts / counts.sum()
    blend = np.tensordot(weights, stack, axes=1)
    acc = accuracy_score(labels, blend.argmax(1))
    print("\nselected weights:")
    for tag, w in sorted(zip(tags, weights), key=lambda kv: -kv[1]):
        print(f"  {tag:<10s} {w:.3f}")
    print(f"\nweighted blend out-of-fold accuracy: {acc:.4f} "
          f"(accuracy on the same data used for selection, so mildly optimistic)")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"tags": tags, "weights": weights.tolist(),
                   "oof_accuracy": float(acc)}, fh, indent=2)
    print("weights written to", os.path.relpath(args.out, P.ROOT))


if __name__ == "__main__":
    main()
