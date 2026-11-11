"""Step 4 - figures and tables for the write-up: curves, confusion matrix, per-class F1."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.stdout.reconfigure(encoding="utf-8")

import plantcls as P  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="b0_288")
    return ap.parse_args()


def draw_curves(histories: list, title: str, path: str) -> None:
    w, h, pad = 720, 320, 48
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    n = max(len(hh["val_acc"]) for hh in histories)
    d.line([(pad, h - pad), (w - 12, h - pad)], fill=(0, 0, 0))
    d.line([(pad, 12), (pad, h - pad)], fill=(0, 0, 0))

    def pt(i, v):
        return (pad + (w - 12 - pad) * i / max(n - 1, 1),
                h - pad - (h - pad - 12) * min(max(v, 0.0), 1.0))

    for k, hh in enumerate(histories):
        for key, colour in (("train_acc", (200, 200, 200)), ("val_acc", (200, 60, 40))):
            pts = [pt(i, v) for i, v in enumerate(hh[key])]
            if len(pts) > 1:
                d.line(pts, fill=colour, width=2)
    d.text((pad, 16), f"{title}  (grey=train acc, red=val acc)", fill=(0, 0, 0))
    for frac in (0.0, 0.5, 1.0):
        y = h - pad - (h - pad - 12) * frac
        d.text((8, y - 6), f"{frac:.1f}", fill=(90, 90, 90))
        d.line([(pad - 4, y), (pad, y)], fill=(0, 0, 0))
    img.save(path)


def draw_confusion(cm: np.ndarray, path: str) -> None:
    cell = 86
    left, top = 190, 90
    w = left + cell * len(P.CLASSES) + 20
    h = top + cell * len(P.CLASSES) + 20
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    for i in range(len(P.CLASSES)):
        d.text((8, top + i * cell + cell // 2 - 6), P.CLASSES[i][:22], fill=(0, 0, 0))
        d.text((left + i * cell + 6, top - 20), P.CLASSES[i][:9], fill=(0, 0, 0))
        for j in range(len(P.CLASSES)):
            v = cm_norm[i, j]
            colour = (int(255 - 150 * v), int(255 - 90 * v), int(255 - 150 * v))
            box = [left + j * cell, top + i * cell, left + (j + 1) * cell - 2,
                   top + (i + 1) * cell - 2]
            d.rectangle(box, fill=colour, outline=(120, 120, 120))
            d.text((box[0] + 8, box[1] + 30), f"{cm[i, j]}\n{v:.2f}", fill=(0, 0, 0))
    d.text((left, 18), "confusion matrix (rows = truth, cols = prediction)", fill=(0, 0, 0))
    img.save(path)


def main() -> None:
    args = parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    all_probs, all_y, histories, lines = [], [], [], []

    for tag in tags:
        run_dir = os.path.join(P.OUT_DIR, tag)
        with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as fh:
            summary = json.load(fh)
        data = np.load(os.path.join(run_dir, "oof.npz"))
        probs, seen, labels = data["probs"], data["seen"], data["labels"]
        if seen.all():
            all_probs.append(probs)
            all_y.append(labels)
        histories += [json.load(open(os.path.join(run_dir, f), encoding="utf-8"))
                      for f in sorted(os.listdir(run_dir)) if f.endswith("_history.json")]
        lines.append(f"## run `{tag}`\n")
        lines.append(f"- args: `{json.dumps(summary['args'], ensure_ascii=False)}`")
        lines.append(f"- OOF accuracy: **{summary['oof_accuracy']:.4f}**, "
                     f"macro F1: **{summary['oof_macro_f1']:.4f}**")
        lines.append(f"- folds: " + ", ".join(
            f"acc={m['accuracy']:.4f}/f1={m['macro_f1']:.4f}/{m['minutes']:.1f}min"
            for m in summary["fold_metrics"]))
        draw_curves(histories, f"accuracy curves - {tag}",
                    os.path.join(P.OUT_DIR, f"curves_{tag}.png"))

    if not all_probs:
        print("no complete OOF predictions yet")
        return
    probs = np.mean(all_probs, axis=0)
    y = all_y[0]
    pred = probs.argmax(axis=1)
    acc = float((pred == y).mean())
    f1 = f1_score(y, pred, average="macro")
    cm = confusion_matrix(y, pred)
    draw_confusion(cm, os.path.join(P.OUT_DIR, "confusion_matrix.png"))
    lines.append("\n## ensemble of out-of-fold predictions\n")
    lines.append(f"- accuracy: **{acc:.4f}**, macro F1: **{f1:.4f}**")
    lines.append(f"- expected errors on 378 test images: **{round((1 - acc) * 378, 1)}**")
    lines.append("\n```\n" + classification_report(
        y, pred, target_names=P.CLASSES, digits=4, zero_division=0) + "```")
    report = os.path.join(P.OUT_DIR, "report.md")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))
    print("report written to", os.path.relpath(report, P.ROOT))


if __name__ == "__main__":
    main()
