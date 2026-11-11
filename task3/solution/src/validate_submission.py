"""Validate a submission file against the official sample format."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import SAMPLE_SUB, load_test_df


def main(path: str):
    sub = pd.read_csv(path)
    ref = pd.read_csv(SAMPLE_SUB)
    assert list(sub.columns) == ["ImageID", "value"], sub.columns.tolist()
    assert len(sub) == len(ref) == 40, (len(sub), len(ref))
    assert sub["ImageID"].tolist() == ref["ImageID"].tolist(), "ImageID order differs"
    assert sub["ImageID"].is_unique
    vals = pd.to_numeric(sub["value"], errors="coerce").values
    assert np.isfinite(vals).all(), "NaN/Inf found"
    assert (vals >= 0).all(), "negative coordinate found"

    sizes = load_test_df().set_index("id")
    for _, row in sub.iterrows():
        iid, axis = row["ImageID"].rsplit("_", 1)
        i = int(iid.split("_")[0])
        limit = float(sizes.loc[i, "width" if axis == "X" else "height"])
        v = float(row["value"])
        assert 0 <= v <= limit, (row["ImageID"], v, limit)
    print(f"OK: {path} rows={len(sub)} min={vals.min():.2f} max={vals.max():.2f}")
    print(sub.head(6).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
