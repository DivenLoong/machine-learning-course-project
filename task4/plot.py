from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_runs(root: Path) -> list[dict]:
    runs = []
    for final_path in sorted(root.glob("*/final_info.json")):
        run_dir = final_path.parent
        with final_path.open("r", encoding="utf-8") as handle:
            final = json.load(handle)
        history_path = run_dir / "history.json"
        history = []
        if history_path.is_file():
            with history_path.open("r", encoding="utf-8") as handle:
                history = json.load(handle).get("history", [])
        runs.append({"name": run_dir.name, "final": final, "history": history})
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot screening and final results")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output", default="runs/figures")
    parser.add_argument("--include-prefix", nargs="*", default=["screen_", "ext_", "final_"])
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runs = [
        run
        for run in load_runs(Path(args.runs))
        if any(run["name"].startswith(prefix) for prefix in args.include_prefix)
    ]
    if not runs:
        raise ValueError("No matching completed runs found")

    fig, axis = plt.subplots(figsize=(10, 6))
    for run in runs:
        if not run["history"]:
            continue
        epochs = [row["epoch"] for row in run["history"]]
        dice = [row["val_dice"] for row in run["history"]]
        axis.plot(epochs, dice, linewidth=1.8, label=run["name"])
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation Dice")
    axis.set_title("DRIVE vessel segmentation screening")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "Figure_1.png", dpi=180)
    plt.close(fig)

    names = [run["name"] for run in runs]
    scores = [run["final"]["best_dice"] for run in runs]
    order = np.argsort(scores)
    fig, axis = plt.subplots(figsize=(11, max(5, len(runs) * 0.35)))
    axis.barh([names[index] for index in order], [scores[index] for index in order], color="#3572a5")
    axis.axvline(0.84, color="#c44e52", linestyle="--", linewidth=1.2, label="Promotion target")
    axis.set_xlabel("Best validation Dice")
    axis.set_title("Candidate comparison")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "Figure_2.png", dpi=180)
    plt.close(fig)
    print(f"Saved figures to {output}")


if __name__ == "__main__":
    main()
