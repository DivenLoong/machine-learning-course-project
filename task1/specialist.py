"""Combine the 5-class ensemble with a Black-grass / Loose Silky-bent specialist.

The specialist only redistributes probability *within* the confusable pair, so images
the main model considers Scentless Mayweed or Sugar beet are left untouched.
Combination happens in log-odds space with a mixing weight tuned on out-of-fold data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-tags",
                    default="b0_288,cnx320,cnx320p,cnx320m,cnx384m,cnxs320m")
    ap.add_argument("--specialist", default="spec_bl",
                    help="comma-separated specialist run tags; their probabilities are "
                         "averaged before combining")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--test-main", default=os.path.join(P.OUT_DIR, "test_probs_main6.npz"))
    ap.add_argument("--test-specialist", default=os.path.join(P.OUT_DIR,
                                                              "test_probs_spec.npz"),
                    help="comma-separated npz files, one per specialist (same order)")
    ap.add_argument("--out", default="")
    return ap.parse_args()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def subset_rows(labels, subset_orig):
    return np.array([i for i, y in enumerate(labels) if y in subset_orig])


def main() -> None:
    args = parse_args()
    tags = [t.strip() for t in args.main_tags.split(",") if t.strip()]
    full = P.scan_train()
    labels = np.array([y for _, y in full])

    main_probs = np.mean([np.load(os.path.join(P.OUT_DIR, t, "oof.npz"))["probs"]
                          for t in tags], axis=0).astype(np.float64)
    spec_tags = [t.strip() for t in args.specialist.split(",") if t.strip()]
    spec_probs, spec_orig = None, None
    for tag in spec_tags:
        spec = np.load(os.path.join(P.OUT_DIR, tag, "oof.npz"))
        spec_probs = (spec["probs"].astype(np.float64) if spec_probs is None
                      else spec_probs + spec["probs"].astype(np.float64))
        spec_orig = spec["orig_labels"]
    spec_probs /= len(spec_tags)

    # The specialist is trained on a filtered subset, so re-derive which rows of the
    # full 500-sample ordering it corresponds to.
    subset_orig = sorted(set(int(v) for v in spec_orig))
    rows = subset_rows(labels, subset_orig)
    assert len(rows) == len(spec_probs), (len(rows), len(spec_probs))
    assert np.array_equal(labels[rows], spec_orig)
    bg, lsb = P.CLASS_TO_IDX["Black-grass"], P.CLASS_TO_IDX["Loose Silky-bent"]
    spec_bg = spec_probs[:, list(subset_orig).index(bg)]
    pair_ids = np.array([bg, lsb])
    spec_ids = np.array(subset_orig)

    pair_mass = main_probs[:, bg] + main_probs[:, lsb]
    q5 = main_probs[:, bg] / np.maximum(pair_mass, 1e-9)
    qs = np.full(len(labels), np.nan)
    qs[rows] = spec_bg

    print(f"main ensemble on the {len(rows)} grass images: "
          f"{accuracy_score(labels[rows], pair_ids[main_probs[rows][:, pair_ids].argmax(1)]):.4f}")
    print(f"specialist alone on the same rows      : "
          f"{accuracy_score(spec_orig, spec_ids[spec_probs.argmax(1)]):.4f}\n")

    best = None
    for alpha in [float(a) for a in args.alphas.split(",")]:
        combined = main_probs.copy()
        if alpha > 0:
            q = sigmoid((1 - alpha) * logit(q5[rows]) + alpha * logit(qs[rows]))
            combined[rows, bg] = pair_mass[rows] * q
            combined[rows, lsb] = pair_mass[rows] * (1 - q)
        acc_all = accuracy_score(labels, combined.argmax(1))
        acc_pair = accuracy_score(
            labels[rows], pair_ids[combined[rows][:, pair_ids].argmax(1)])
        print(f"  alpha={alpha:<5} pair accuracy={acc_pair:.4f}  overall OOF={acc_all:.4f}")
        if best is None or acc_all > best[1]:
            best = (alpha, acc_all, acc_pair)
    print(f"\nbest alpha={best[0]}  overall OOF={best[1]:.4f}  pair={best[2]:.4f}")

    if args.out and best[0] > 0:
        alpha = best[0]
        tmain = np.load(args.test_main)
        tspec_files = [f.strip() for f in args.test_specialist.split(",") if f.strip()]
        tspec_probs = None
        for path in tspec_files:
            arr = np.load(path)["probs"].astype(np.float64)
            tspec_probs = arr if tspec_probs is None else tspec_probs + arr
        tspec_probs /= len(tspec_files)
        ids = list(tmain["ids"])
        tprobs = tmain["probs"].astype(np.float64)
        assert tspec_probs.shape[0] == len(ids)
        t_mass = tprobs[:, bg] + tprobs[:, lsb]
        t_q5 = tprobs[:, bg] / np.maximum(t_mass, 1e-9)
        t_qs = tspec_probs[:, 0]
        t_q = sigmoid((1 - alpha) * logit(t_q5) + alpha * logit(t_qs))
        tprobs[:, bg] = t_mass * t_q
        tprobs[:, lsb] = t_mass * (1 - t_q)

        with open(os.path.join(P.OUT_DIR, "blend.json"), encoding="utf-8") as fh:
            prior = np.asarray(json.load(fh)["estimated_prior"])
        preds = (tprobs * prior).argmax(1)

        order = [line.split(",")[0] for line in
                 open(P.SAMPLE_SUBMISSION, encoding="utf-8").read().strip().splitlines()[1:]]
        row_of = {name: i for i, name in enumerate(ids)}
        out_path = os.path.join(P.ROOT, args.out)
        with open(out_path, "w", encoding="utf-8", newline="") as fh:
            fh.write("ID,Category\n")
            for name in order:
                fh.write(f"{name},{P.CLASSES[preds[row_of[name]]]}\n")
        np.savez(os.path.join(P.OUT_DIR, "test_probs_specialist_blend.npz"),
                 probs=tprobs, ids=np.array(ids))
        print(f"wrote {os.path.relpath(out_path, P.ROOT)} with alpha={alpha}")
        import collections
        print("distribution:", collections.Counter(P.CLASSES[p] for p in preds).most_common())


if __name__ == "__main__":
    main()
