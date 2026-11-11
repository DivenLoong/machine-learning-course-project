"""Stage-1 ablation driver: run train.py configs one after another, then summarise.

    python src/ablate.py --stage res      # resolution sweep
    python src/ablate.py --stage aug      # augmentation ablation at the chosen resolution
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"


def stage_res(epochs: int):
    return [
        (f"abl_res{res}", ["--res", str(res), "--batch", str(bs)])
        for res, bs in [(224, 32), (112, 64), (64, 128)]
    ] + [("abl_res64_e60", ["--res", "64", "--batch", "128", "--epochs", "60"])]


def stage_aug(epochs: int, res: int, bs: int):
    common = ["--res", str(res), "--batch", str(bs)]
    geo = ["--rot", "15", "--scale", "0.15", "--erase-p", "0.4", "--mix-p", "0.7"]
    return [
        ("aug_base", common + []),
        ("aug_geo", common + geo),
        ("aug_geo_dp", common + geo + ["--drop-path", "0.25"]),
        ("aug_geo_lrh", common + geo + ["--lr-head", "3e-4"]),
        ("aug_geo_all", common + geo + ["--drop-path", "0.25", "--lr-head", "3e-4",
                                        "--label-smoothing", "0.2", "--brightness", "0.3",
                                        "--contrast", "0.3"]),
    ]


def run_one(tag: str, extra, epochs: int, fold: int, backbone: str):
    summary = os.path.join(ds.RUNS_DIR, tag, "summary.json")
    if os.path.exists(summary):
        print(f"[skip] {tag} already done", flush=True)
        return json.load(open(summary))
    args = [PY, os.path.join(HERE, "train.py"), "--backbone", backbone,
            "--epochs", str(epochs), "--fold", str(fold), "--tag", tag, "--seed", str(fold)] + extra
    print(f"[run] {tag}: {' '.join(args[2:])}", flush=True)
    rc = subprocess.run(args, cwd=ds.ROOT).returncode
    if rc != 0 or not os.path.exists(summary):
        print(f"[fail] {tag} rc={rc}", flush=True)
        return None
    return json.load(open(summary))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["res", "aug"], default="res")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--backbone", default=BACKBONE)
    p.add_argument("--res", type=int, default=64)
    p.add_argument("--batch", type=int, default=128)
    a = p.parse_args()

    cfgs = stage_res(a.epochs) if a.stage == "res" else stage_aug(a.epochs, a.res, a.batch)
    results = {}
    for tag, extra in cfgs:
        r = run_one(tag, extra, a.epochs, a.fold, a.backbone)
        if r:
            results[tag] = r
    print("\n=== ablation summary (fold %d, %d epochs) ===" % (a.fold, a.epochs))
    print(f"{'tag':<20}{'res':>5}{'val_acc':>9}{'val_ema':>9}{'final':>9}{'min':>8}")
    for tag, r in sorted(results.items(), key=lambda kv: -kv[1].get("final_val_acc", 0)):
        print(f"{tag:<20}{r['res']:>5}{r.get('acc', 0):>9.4f}{'':>9}"
              f"{r.get('final_val_acc', 0):>9.4f}{r.get('minutes', 0):>8.1f}")
    out = os.path.join(ds.OUT_DIR, f"ablation_{a.stage}.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
