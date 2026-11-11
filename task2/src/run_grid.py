"""Sequential job runner for stage 2/3: (backbone, resolution, batch, epochs) x folds.

Each job is skipped when outputs/runs/<tag>/summary.json already exists, so the
script is safely resumable after an interruption.  Test-time probabilities are
produced right after each fold finishes.

    python src/run_grid.py --jobs "convnext_tiny.fb_in22k_ft_in1k:224:32:60,resnet18.a1_in1k:64:128:60" \
        --folds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SHORT = {
    "convnext_tiny.fb_in22k_ft_in1k": "ctn",
    "convnext_small.fb_in22k_ft_in1k": "cns",
    "tf_efficientnetv2_s.in21k_ft_in1k": "efs",
    "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k": "swt",
    "resnet18.a1_in1k": "rn18",
    "resnet34.a1_in1k": "rn34",
    "resemotenet": "rem",
}


def tag_for(backbone, res, fold):
    key = SHORT.get(backbone, backbone.split(".")[0][:8])
    return f"{key}{res}_f{fold}"


def run(cmd, log):
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.run(cmd, cwd=ds.ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", required=True,
                   help="comma separated backbone:res:batch:epochs")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--max-hours", type=float, default=0.0)
    p.add_argument("--no-predict", action="store_true")
    a = p.parse_args()

    folds = [int(x) for x in a.folds.split(",") if x.strip() != ""]
    jobs = []
    for spec in a.jobs.split(","):
        parts = spec.strip().split(":")
        jobs.append({"backbone": parts[0], "res": int(parts[1]),
                     "batch": int(parts[2]), "epochs": int(parts[3]) if len(parts) > 3 else 60})

    os.makedirs(ds.OUT_DIR, exist_ok=True)
    log = os.path.join(ds.OUT_DIR, "run_grid.log")
    t0 = time.time()
    for job in jobs:
        for fold in folds:
            tag = tag_for(job["backbone"], job["res"], fold)
            summary = os.path.join(ds.RUNS_DIR, tag, "summary.json")
            if not os.path.exists(summary):
                cmd = [PY, os.path.join(HERE, "train.py"), "--backbone", job["backbone"],
                       "--res", str(job["res"]), "--batch", str(job["batch"]),
                       "--epochs", str(job["epochs"]), "--fold", str(fold),
                       "--tag", tag, "--seed", str(fold)]
                print(f"[train] {tag}", flush=True)
                rc = run(cmd, log)
                if rc != 0:
                    print(f"[fail] {tag} rc={rc} (continuing)", flush=True)
                    continue
            else:
                print(f"[skip] {tag} done", flush=True)
            if not a.no_predict:
                tp = os.path.join(ds.RUNS_DIR, tag, "test_probs.npy")
                if not os.path.exists(tp):
                    print(f"[predict] {tag}", flush=True)
                    run([PY, os.path.join(HERE, "predict.py"), "--tag", tag], log)
            if a.max_hours and (time.time() - t0) / 3600 > a.max_hours:
                print("[stop] time budget reached", flush=True)
                return
    print(f"[done] grid finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
