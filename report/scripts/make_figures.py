"""Render report figures from the versioned training and submission records."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

REPORT = Path(__file__).resolve().parents[1]
DATA = REPORT / "data"
FIGURES = REPORT / "figures"
BLUE, TEAL, AMBER, INK, GRID = "#346A98", "#21867A", "#D58A39", "#19324D", "#DCE4EC"


def read_json(relative):
    return json.loads((DATA / relative).read_text(encoding="utf-8"))


def read_csv(relative):
    with (DATA / relative).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def finish(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def pair():
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 2.62), layout="constrained")
    for axis in axes:
        axis.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.85)
        axis.set_axisbelow(True)
    return fig, axes


def line(ax, x, y, label, color=BLUE, **kwargs):
    ax.plot(x, y, label=label, color=color, linewidth=1.65, **kwargs)


def class_bars(ax, counts, labels):
    values = [counts[label] for label in labels]
    ax.barh(labels, values, color=[TEAL if v == max(values) else BLUE for v in values], height=0.58)
    ax.set(xlim=(0, max(values) * 1.18), xlabel="Training images")
    ax.invert_yaxis()
    for i, value in enumerate(values):
        ax.text(value + max(values)*0.018, i, f"{value:,}", va="center", color=INK, fontsize=8)


def score_lookup(slug, name):
    return next(row for row in SCORES if row["competition"] == slug and row["filename"] == name)


def paired_scores(ax, slug, entries, ylabel, show_private=True):
    ix = np.arange(len(entries))
    results = [score_lookup(slug, entry[1]) for entry in entries]
    ax.set(xticks=ix, xticklabels=[entry[0] for entry in entries], ylabel=ylabel)
    ax.plot(ix, [float(row["public"]) for row in results], "o", color=BLUE, label="Public")
    if show_private:
        ax.plot(ix, [float(row["private"]) for row in results], "^", color=AMBER, label="Private")
        ax.legend(frameon=False, fontsize=8)


def plant():
    run = read_json("plant/fix_cnx320p/fold0_history.json")
    epochs = np.arange(1, len(run["train_loss"]) + 1)
    fig, (left, right) = pair()
    line(left, epochs, run["train_loss"], "Training loss")
    line(right, epochs, run["val_acc"], "Validation accuracy", TEAL)
    left.set(xlabel="Epoch", ylabel="Cross-entropy", title="Training loss")
    right.set(xlabel="Epoch", ylabel="Accuracy", title="Fold 0 validation")
    right.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    for axis in (left, right):
        axis.axvline(4, color=AMBER, linestyle="--", linewidth=1)
    finish(fig, "plant_training")

    fig, (left, right) = pair()
    counts = read_json("class_counts.json")["plant"]
    class_bars(left, counts, list(counts))
    tags = ["fix_b0_288", "fix_cnx320", "fix_cnx320p", "fix_cnxs320m", "fix_cnx384m"]
    labels = ["B0 / 288", "Tiny / 320", "Tiny / pseudo", "Small / mask", "Tiny / 384"]
    values = [read_json(f"plant/{tag}/summary.json")["oof_accuracy"] for tag in tags]
    right.hlines(range(len(values)), 0.9, values, color=GRID)
    right.scatter(values, range(len(values)), color=[TEAL if v == max(values) else BLUE for v in values], zorder=3)
    right.set(yticks=range(len(values)), yticklabels=labels, xlim=(0.9, 0.963), xlabel="OOF accuracy")
    right.invert_yaxis()
    for i, value in enumerate(values):
        right.text(value + .002, i, f"{value:.3f}", va="center", fontsize=8)
    finish(fig, "plant_data")


def fer():
    rows = [json.loads(line) for line in (DATA / "fer/cns224_f0/log.jsonl").read_text(encoding="utf-8").splitlines()]
    x = [row["epoch"] + 1 for row in rows]
    fig, (left, right) = pair()
    line(left, x, [row["loss"] for row in rows], "Training loss")
    line(right, x, [row["train_acc"] for row in rows], "Train", BLUE)
    for k, label, color in (("val_acc", "Validation", AMBER), ("val_acc_ema", "Validation EMA", TEAL)):
        ev = [(i, row[k]) for i, row in zip(x, rows) if k in row]
        line(right, [i for i, _ in ev], [v for _, v in ev], label, color)
    left.set(xlabel="Epoch", ylabel="Cross-entropy", title="Training loss")
    right.set(xlabel="Epoch", ylabel="Accuracy", title="Train and validation")
    right.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    right.legend(frameon=False, fontsize=7.6)
    finish(fig, "fer_training")

    fig, (left, right) = pair()
    counts = read_json("class_counts.json")["fer"]
    class_bars(left, counts, ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
    paired_scores(right, "2026task2", [("v1\n112px", "sub_v1_112.csv"), ("v2\n224px", "sub_v2_cns224.csv"),
                                       ("Bagged", "sub_bagged.csv"), ("Logit", "sub_bagged_logit.csv")],
                  "Public accuracy", show_private=False)
    right.set_ylim(.741, .749)
    finish(fig, "fer_data")


def fovea():
    run = "exp030_vessel_10fold_seed123"
    rows = read_csv(f"fovea/{run}/history_f0.csv")
    x = [int(row["epoch"]) for row in rows]
    fig, (left, right) = pair()
    line(left, x, [float(row["loss"]) for row in rows], "Training loss")
    y = [float(row["val_rmse_px"]) for row in rows]
    line(right, x, y, "Validation RMSE", TEAL)
    best = int(np.argmin(y))
    right.scatter(x[best], y[best], s=28, color=AMBER, zorder=3, label=f"Min: epoch {x[best]}")
    left.set(xlabel="Epoch", ylabel="Training loss", title="Heatmap and coordinate objective")
    right.set(xlabel="Epoch", ylabel="RMSE (px)", title="Fold 0 validation", yscale="log")
    right.legend(frameon=False, fontsize=8)
    finish(fig, "fovea_training")

    fig, (left, right) = pair()
    selected = [r for r in read_csv("fovea/fold_errors.csv") if r["run"] == run]
    values = [float(r["rmse_px"]) for r in selected]
    left.bar([r["fold"] for r in selected], values,
             color=[AMBER if value == max(values) else BLUE for value in values])
    left.set(xlabel="Fold", ylabel="OOF RMSE (px)", title="Seed 123, held-out predictions")
    paired_scores(right, "2026task3", [(v, f"submission_{v}.csv") for v in ("v2", "v7", "v11", "v12", "v15")],
                  "Leaderboard RMSE (px)")
    finish(fig, "fovea_results")


def drive():
    rows = read_csv("drive/native_model_log.csv")
    x = [int(row["epoch"]) for row in rows]
    y = [float(row["val_f1"]) for row in rows]
    fig, (left, right) = pair()
    line(left, x, [float(row["train_loss"]) for row in rows], "Training loss")
    line(right, x, y, "Native validation Dice", TEAL)
    best = int(np.argmax(y))
    right.scatter(x[best], y[best], s=28, color=AMBER, zorder=3, label=f"Max: epoch {x[best]}")
    left.set(xlabel="Epoch", ylabel="BCE-Dice", title="Native U-Net training")
    right.set(xlabel="Epoch", ylabel="Native validation Dice", title="Native U-Net validation")
    right.legend(frameon=False, fontsize=8)
    finish(fig, "drive_training")

    fig, (left, right) = pair()
    with (REPORT.parent / "task4" / "EXPERIMENT_LOG.csv").open(encoding="utf-8", newline="") as stream:
        entries = {row["experiment_id"]: row for row in csv.DictReader(stream)}
    selected = [("FR DTI .65/.40","E009"), ("FR + native","E032"), ("Native .35","E031"),
                ("FR .55, no DTI","E034"), ("Native .50","E030")]
    metric = [float(entries[key]["public_dice"]) for _, key in selected]
    for i, val in enumerate(metric):
        left.hlines(i, .74, val, color=GRID)
        left.scatter(val, i, color=TEAL if i == 0 else BLUE)
        left.text(val + .003, i, f"{val:.5f}", va="center", fontsize=7.5)
    left.set(yticks=range(len(selected)), yticklabels=[name for name, _ in selected],
             xlim=(.74,.85), xlabel="Public Dice")
    left.invert_yaxis()
    paired_scores(right, "2026task-4", [("FR\n.65/.40","submission_fr_official.csv"),
                                       ("FR\n.80/.40","submission_public_fr_notta_dti080.csv"),
                                       ("FR +\nnative","submission_candidate_fr075_native025_dti.csv")],
                  "Leaderboard Dice")
    finish(fig, "drive_results")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": INK, "text.color": INK,
                         "axes.edgecolor": "#A3B4C4", "axes.titlepad": 8, "pdf.fonttype": 42})
    FIGURES.mkdir(exist_ok=True)
    SCORES = read_csv("kaggle_scores.csv")
    plant()
    fer()
    fovea()
    drive()
