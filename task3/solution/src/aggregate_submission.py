"""Robust aggregation of per-fold test predictions into a submission."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import SAMPLE_SUB


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", required=True)
    p.add_argument("--method", choices=["mean", "median", "trimmed", "hybrid"],
                   default="hybrid")
    p.add_argument("--trim", type=int, default=3)
    p.add_argument("--outlier-threshold", type=float, default=80.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.matrix)
    rows = []
    for i, g in df.groupby("id"):
        p = g[["pred_x", "pred_y"]].values.astype(np.float64)
        if args.method == "mean":
            xy = p.mean(0)
        elif args.method == "median":
            xy = np.median(p, 0)
        elif args.method in ("trimmed", "hybrid"):
            med = np.median(p, 0)
            dist = np.linalg.norm(p - med, axis=1)
            if args.method == "hybrid" and dist.max() <= args.outlier_threshold:
                xy = p.mean(0)
            else:
                keep = np.argsort(dist)[: max(1, len(p) - args.trim)]
                xy = p[keep].mean(0)
        rows.append({"id": int(i), "pred_x": xy[0], "pred_y": xy[1]})
    agg = pd.DataFrame(rows).sort_values("id")
    agg.to_csv(Path(args.out).with_name(Path(args.out).stem + "_raw.csv"), index=False)

    sub = pd.read_csv(SAMPLE_SUB)
    values = {}
    for r in agg.itertuples():
        values[f"{r.id}_Fovea_X"] = float(r.pred_x)
        values[f"{r.id}_Fovea_Y"] = float(r.pred_y)
    sub["value"] = sub["ImageID"].map(values)
    if sub["value"].isna().any():
        raise RuntimeError("Missing ImageID in aggregation")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(sub)} rows, method={args.method})")


if __name__ == "__main__":
    main()
