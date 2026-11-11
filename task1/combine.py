"""How should the frozen-feature probe be combined with the CNN ensemble?

The probe is much weaker on its own (0.868 vs 0.930) but makes different mistakes, so the
question is whether any combination rule captures that without dragging accuracy down.
Every rule is scored on the same out-of-fold data, and a nested split guards the
learned rules against overfitting the selection set.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402

CNN_MEMBERS = ["b0_288", "cnx320", "cnx320p", "cnx320m", "cnx384m", "cnxs320m"]
BG, LSB = P.CLASS_TO_IDX["Black-grass"], P.CLASS_TO_IDX["Loose Silky-bent"]


def load_cnn() -> tuple[np.ndarray, np.ndarray]:
    probs, labels = [], None
    for tag in CNN_MEMBERS:
        data = np.load(os.path.join(P.OUT_DIR, tag, "oof.npz"))
        probs.append(data["probs"].astype(np.float64))
        labels = data["labels"]
    return np.mean(probs, axis=0), labels


def main() -> None:
    cnn, labels = load_cnn()
    probe = np.load(os.path.join(P.OUT_DIR, "dino_oof.npz"))["probs"].astype(np.float64)
    print(f"CNN ensemble OOF      {accuracy_score(labels, cnn.argmax(1)):.4f}")
    print(f"DINOv2 probe OOF      {accuracy_score(labels, probe.argmax(1)):.4f}\n")

    print("weighted average  p = (1-w)*cnn + w*probe")
    best_w, best_acc = 0.0, accuracy_score(labels, cnn.argmax(1))
    for w in np.arange(0.0, 0.55, 0.05):
        acc = accuracy_score(labels, ((1 - w) * cnn + w * probe).argmax(1))
        flag = ""
        if acc > best_acc + 1e-9:
            best_w, best_acc = float(w), acc
            flag = "  <-"
        print(f"   w={w:.2f}  {acc:.4f}{flag}")

    print("\nlog-odds average (sharper, lets the probe override confident CNN errors)")
    eps = 1e-6
    lc, lp = np.log(np.clip(cnn, eps, 1)), np.log(np.clip(probe, eps, 1))
    for w in (0.1, 0.2, 0.3, 0.4):
        acc = accuracy_score(labels, ((1 - w) * lc + w * lp).argmax(1))
        print(f"   w={w:.2f}  {acc:.4f}")

    print("\nselective: fall back to the probe when CNN confidence is low")
    conf = cnn.max(axis=1)
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        pred = cnn.argmax(1).copy()
        low = conf < thr
        pred[low] = probe[low].argmax(1)
        print(f"   threshold {thr:.1f}: {low.sum():3d} images switched, "
              f"accuracy {accuracy_score(labels, pred):.4f}")

    print("\npair-restricted: re-decide only within Black-grass / Loose Silky-bent")
    for w in (0.2, 0.3, 0.4, 0.5):
        mixed = ((1 - w) * lc + w * lp)
        pred = cnn.argmax(1).copy()
        pair = np.isin(cnn.argmax(1), [BG, LSB])
        pred[pair] = mixed[pair][:, [BG, LSB]].argmax(1)
        pred[pair] = np.where(mixed[pair][:, [BG, LSB]].argmax(1) == 0, BG, LSB)
        print(f"   w={w:.1f}: {pair.sum():3d} images re-decided, "
              f"accuracy {accuracy_score(labels, pred):.4f}")

    print("\nstacking: logistic regression on concatenated posteriors (nested CV)")
    features = np.hstack([cnn, probe])
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for C in (0.1, 1.0):
        oof = np.zeros_like(cnn)
        for tr, va in outer.split(features, labels):
            clf = LogisticRegression(C=C, max_iter=3000)
            clf.fit(features[tr], labels[tr])
            oof[va] = clf.predict_proba(features[va])
        print(f"   C={C:<5} stacked OOF {accuracy_score(labels, oof.argmax(1)):.4f}")

    print("\nreference: oracle upper bound if the probe always fixed CNN errors "
          "it gets right")
    cnn_wrong = cnn.argmax(1) != labels
    fixed = cnn_wrong & (probe.argmax(1) == labels)
    print(f"   errors {cnn_wrong.sum()} -> {cnn_wrong.sum() - fixed.sum()} "
          f"(accuracy {1 - (cnn_wrong.sum() - fixed.sum()) / len(labels):.4f})")


if __name__ == "__main__":
    main()
