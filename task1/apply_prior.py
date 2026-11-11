"""Write a submission from an already-computed probability file plus the class prior.

Separated from predict.py so that a corrected prior does not cost another full GPU pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default=os.path.join(P.OUT_DIR, "test_probs.npz"))
    ap.add_argument("--prior-json", default=os.path.join(P.OUT_DIR, "blend.json"))
    ap.add_argument("--prior-scale", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = np.load(args.probs, allow_pickle=True)
    probs, ids = data["probs"].astype(np.float64), list(data["ids"])
    with open(args.prior_json, encoding="utf-8") as fh:
        prior = np.asarray(json.load(fh)["estimated_prior"], dtype=np.float64)
    assert len(prior) == probs.shape[1], (len(prior), probs.shape)
    prior = prior ** args.prior_scale
    preds = (probs * prior).argmax(axis=1)

    order = [line.split(",")[0] for line in
             open(P.SAMPLE_SUBMISSION, encoding="utf-8").read().strip().splitlines()[1:]]
    row_of = {name: i for i, name in enumerate(ids)}
    out_path = os.path.join(P.ROOT, args.out)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("ID,Category\n")
        for name in order:
            fh.write(f"{name},{P.CLASSES[preds[row_of[name]]]}\n")
    print(f"wrote {os.path.relpath(out_path, P.ROOT)} ({len(order)} rows)")
    print("distribution:", Counter(P.CLASSES[p] for p in preds).most_common())


if __name__ == "__main__":
    main()
