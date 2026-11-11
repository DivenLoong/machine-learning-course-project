"""Inventory of every training run and submission produced so far."""

from __future__ import annotations

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402


def main() -> None:
    print(f"{'tag':<12s} {'classes':>7s} {'OOF':>10s} {'folds':>6s} {'pseudo':>7s} "
          f"{'arch':<34s}")
    for name in sorted(os.listdir(P.OUT_DIR)):
        path = os.path.join(P.OUT_DIR, name)
        if not os.path.isdir(path) or name.startswith("imgcache"):
            continue
        folds = len([f for f in os.listdir(path) if f.endswith("_model.pt")])
        summary_path = os.path.join(path, "summary.json")
        if not os.path.exists(summary_path):
            print(f"{name:<12s} {'-':>7s} {'INCOMPLETE':>10s} {folds:>6d} {'-':>7s}")
            continue
        with open(summary_path, encoding="utf-8") as fh:
            s = json.load(fh)
        classes = s.get("classes", P.CLASSES)
        arch = s["args"].get("arch", "?")
        print(f"{name:<12s} {len(classes):>7d} {s['oof_accuracy']:>10.4f} {folds:>6d} "
              f"{s.get('n_pseudo', 0):>7d} {arch:<34s}")


if __name__ == "__main__":
    main()
