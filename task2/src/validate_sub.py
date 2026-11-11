"""Validate a submission CSV against the competition's sample submission."""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds  # noqa: E402


def validate(path: str) -> bool:
    sample = pd.read_csv(ds.SAMPLE_SUB)
    sub = pd.read_csv(path)
    ok = True
    if list(sub.columns) != list(sample.columns):
        print(f"FAIL columns {list(sub.columns)} != {list(sample.columns)}")
        ok = False
    if len(sub) != len(sample):
        print(f"FAIL row count {len(sub)} != {len(sample)}")
        ok = False
    if len(sub) == len(sample) and not (sub["ID"].values == sample["ID"].values).all():
        print("FAIL ID order/set differs from sample submission")
        ok = False
    if sub["Emotion"].isna().any():
        print("FAIL NaN in Emotion")
        ok = False
    bad = sorted(set(sub["Emotion"].unique()) - set(range(ds.N_CLS)))
    if bad:
        print(f"FAIL labels outside 0..{ds.N_CLS - 1}: {bad}")
        ok = False
    dup = sub["ID"].duplicated().sum()
    if dup:
        print(f"FAIL {dup} duplicated IDs")
        ok = False
    hist = sub["Emotion"].value_counts().sort_index().to_dict()
    print(f"{'OK  ' if ok else 'BAD '} {os.path.basename(path)}  rows={len(sub)}  hist={hist}")
    return ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python src/validate_sub.py <csv> [more csvs...]")
        sys.exit(2)
    all_ok = all(validate(p) for p in sys.argv[1:])
    sys.exit(0 if all_ok else 1)
