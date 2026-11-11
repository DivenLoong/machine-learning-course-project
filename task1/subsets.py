"""Compare candidate ensembles on the out-of-fold predictions."""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402

CANDIDATES = [
    ("single: cnxs320m", ["cnxs320m"]),
    ("5 strong (pre-cnxs)", ["b0_288", "cnx320", "cnx320p", "cnx320m", "cnx384m"]),
    ("6 strong (+cnxs320m)", ["b0_288", "cnx320", "cnx320p", "cnx320m", "cnx384m",
                              "cnxs320m"]),
    ("6 strong + rn50b", ["b0_288", "cnx320", "cnx320p", "cnx320m", "cnx384m",
                          "cnxs320m", "rn50b"]),
    ("6 strong + b0m", ["b0_288", "cnx320", "cnx320p", "cnx320m", "cnx384m",
                        "cnxs320m", "b0m"]),
    ("8 strong-ish (all)", ["b0_288", "cnx320", "cnx320p", "cnx320m", "cnx384m",
                            "cnxs320m", "rn50b", "b0m"]),
    ("convnext family", ["cnx320", "cnx320p", "cnx320m", "cnx384m", "cnxs320m"]),
    ("cnx320p+cnx384m+cnxs320m", ["cnx320p", "cnx384m", "cnxs320m"]),
    ("6 strong + cnx320r3", ["b0_288", "cnx320", "cnx320p", "cnx320m", "cnx384m",
                             "cnxs320m", "cnx320r3"]),
    ("7 runs + cnxs320r3", ["b0_288", "cnx320", "cnx320p", "cnx320m", "cnx384m",
                            "cnxs320m", "cnx320r3", "cnxs320r3"]),
    ("5 best (drop b0_288,cnx320)", ["cnx320p", "cnx320m", "cnx384m", "cnxs320m",
                                     "cnx320r3"]),
    ("v3 set + cnx320p_s7", ["b0_288", "cnx320", "cnx320p", "cnx320m", "cnx384m",
                             "cnxs320m", "cnx320p_s7"]),
    ("v3 set + cnx320p_s7 + r3", ["b0_288", "cnx320", "cnx320p", "cnx320m", "cnx384m",
                                  "cnxs320m", "cnx320r3", "cnxs320r3", "cnx320p_s7"]),
    ("cnx320p_s7 alone", ["cnx320p_s7"]),
]


def main() -> None:
    tags = sorted({t for _, sub in CANDIDATES for t in sub})
    probs, labels = {}, None
    for tag in tags:
        data = np.load(os.path.join(P.OUT_DIR, tag, "oof.npz"))
        probs[tag] = data["probs"].astype(np.float64)
        labels = data["labels"]
    print(f"available runs: {tags}\n")
    for name, sub in CANDIDATES:
        blend = np.mean([probs[t] for t in sub], axis=0)
        acc = accuracy_score(labels, blend.argmax(1))
        print(f"{name:<28s} {acc:.4f}   ({len(sub)} runs)")


if __name__ == "__main__":
    main()
