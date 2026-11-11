"""Compatibility entry point for the experiment workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from drive_vessel.engine import train_fold
from drive_vessel.utils import load_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--config", default="configs/resnet34.yaml")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.out_dir)
    config = load_yaml(args.config)
    result = train_fold(config, args.fold, output)
    write_json(output / "experiment_result.json", result)
    print(result)


if __name__ == "__main__":
    main()
