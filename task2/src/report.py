"""Build the experiment report: tables, figures and REPORT.md.

    python src/report.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds  # noqa: E402
import ensemble as E  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

FIG_DIR = os.path.join(ds.OUT_DIR, "report")


def load_logs(tag):
    p = os.path.join(ds.RUNS_DIR, tag, "log.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def fig_class_distribution(tr_y, path):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    counts = np.bincount(tr_y, minlength=ds.N_CLS)
    ax.bar(ds.CLASSES, counts, color="#4C72B0")
    for i, c in enumerate(counts):
        ax.text(i, c + 80, str(c), ha="center", fontsize=9)
    ax.set_ylabel("training images")
    ax.set_title("Class distribution (train, n=%d)" % len(tr_y))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_curves(tags, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for tag in tags:
        logs = load_logs(tag)
        if not logs:
            continue
        ep = [l["epoch"] for l in logs]
        axes[0].plot(ep, [l["loss"] for l in logs], label=tag)
        if "val_acc" in logs[0]:
            axes[1].plot(ep, [l.get("val_acc", np.nan) for l in logs], label=tag)
            if "val_acc_ema" in logs[0]:
                axes[1].plot(ep, [l.get("val_acc_ema", np.nan) for l in logs], "--", alpha=0.6)
    axes[0].set_xlabel("epoch"), axes[0].set_ylabel("train loss")
    axes[1].set_xlabel("epoch"), axes[1].set_ylabel("val accuracy")
    axes[1].axhline(0.769, color="r", ls=":", lw=1)
    axes[1].text(0.5, 0.775, "public LB #3 = 0.769", color="r", fontsize=7)
    for ax in axes:
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(ds.N_CLS)), ax.set_yticks(range(ds.N_CLS))
    ax.set_xticklabels(ds.CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ds.CLASSES, fontsize=8)
    ax.set_xlabel("predicted"), ax.set_ylabel("true")
    for i in range(ds.N_CLS):
        for j in range(ds.N_CLS):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    dev = ds.Device("cpu")
    y = dev.train_y.numpy().astype(np.int64)

    runs = E.scan_runs()
    groups = E.build_group_oof(runs, len(y)) if runs else {}
    # Same rule as ensemble.py: a group that has not finished all folds covers a
    # different subset of training images, so blending it would compare apples to
    # oranges. Drop it here too so the report matches the submitted blend.
    incomplete = [k for k, v in groups.items() if v["mask"].mean() < 0.999]
    if incomplete:
        print(f"[report] excluding incomplete groups: {incomplete}")
        groups = {k: v for k, v in groups.items() if k not in incomplete}
    weights, oof_acc = (E.greedy(groups, y) if groups else ({}, 0.0))
    _, oof_probs = (E.oof_accuracy(groups, weights, y) if groups else (0, None))

    # ---- ablation ----
    abl = {}
    for stage in ("res", "aug"):
        p = os.path.join(ds.OUT_DIR, f"ablation_{stage}.json")
        if os.path.exists(p):
            abl[stage] = json.load(open(p))

    fig_class_distribution(y, os.path.join(FIG_DIR, "class_distribution.png"))
    done_tags = [r["tag"] for r in runs]
    if done_tags:
        fig_curves(done_tags[:8], os.path.join(FIG_DIR, "curves.png"))

    cm = report_txt = None
    if oof_probs is not None:
        # oof_probs holds a zero vector (not NaN) for uncovered rows, so derive the
        # coverage mask from the groups themselves instead of from the probabilities.
        cov = np.zeros(len(y), bool)
        for k in weights:
            cov |= groups[k]["mask"]
        pred = oof_probs.argmax(1)
        cm = confusion_matrix(y[cov], pred[cov])
        report_txt = classification_report(y[cov], pred[cov], target_names=ds.CLASSES, digits=4)
        fig_confusion(cm, os.path.join(FIG_DIR, "confusion.png"))

    # ---- markdown ----
    L = []
    L.append("# 任务二：图像情感分类 —— 纯深度学习方案实验报告\n")
    L.append("## 1. 任务与数据\n")
    L.append(f"- 竞赛：Kaggle `2026task2`，6 类面部情感分类，评分 = 测试集准确率。")
    L.append(f"- 数据：{dev.n_train:,} 张训练图（6 类文件夹）、{dev.n_test:,} 张测试图，"
             f"全部为 48×48 单通道灰度 JPEG。")
    L.append(f"- 类别分布：{ {c: int((y == i).sum()) for i, c in enumerate(ds.CLASSES)} }，"
             f"最不平衡比约 {np.bincount(y).max() / np.bincount(y).min():.2f}:1。")
    L.append("- 标签索引按规则：0 愤怒 / 1 恐惧 / 2 快乐 / 3 悲伤 / 4 惊讶 / 5 中性。")
    L.append("\n![类别分布](outputs/report/class_distribution.png)\n")

    L.append("## 2. 方法\n")
    L.append("- 骨干：ImageNet / ImageNet-21k 预训练的 timm 模型微调；另训练一个从零开始的 "
             "ResEmoteNet（48×48 原生分辨率）作为架构多样性来源。")
    L.append("- 数据管线：全部 35,340 张图解码为 uint8 常驻显存（81 MB），训练时在 GPU 上做批量增强，"
             "无 DataLoader 瓶颈。")
    L.append("- 增强：随机仿射（旋转/平移/缩放/错切）、水平翻转、亮度对比度扰动、随机擦除、"
             "Mixup/CutMix、标签平滑 0.1。")
    L.append("- 优化：AdamW，骨干 lr 1e-4 / 分类头 lr 1e-3，权重衰减 0.05，"
             "3 epoch 线性 warmup + 余弦退火，AMP 混合精度，权重 EMA(0.999)。")
    L.append("- 验证：分层 5 折交叉验证（seed 42），所有实验复用同一份划分。")
    L.append("- 集成：以 OOF 准确率为目标做贪心前向选择确定各骨干权重；推理时 TTA"
             "（水平翻转 × 三档尺度）。\n")

    if abl:
        L.append("## 3. 消融实验\n")
        for stage, d in abl.items():
            L.append(f"### {stage}\n")
            L.append("| 配置 | 分辨率 | 最终验证准确率 | 用时(min) |")
            L.append("| --- | --- | --- | --- |")
            for tag, r in sorted(d.items(), key=lambda kv: -kv[1].get("final_val_acc", 0)):
                L.append(f"| {tag} | {r['res']} | {r.get('final_val_acc', 0):.4f} | "
                         f"{r.get('minutes', 0):.1f} |")
            L.append("")

    if groups:
        gstats = {g: E.oof_accuracy(groups, {g: 1.0}, y)[0] for g in groups}
        L.append("## 4. 各骨干结果（5 折 OOF）\n")
        L.append("| 分组 | 折数 | 5 折 OOF | 单折均值 |")
        L.append("| --- | --- | --- | --- |")
        for g in sorted(groups, key=lambda k: -gstats[k]):
            fold_accs = [r["summary"].get("final_val_acc", 0) for r in runs if r["group"] == g]
            L.append(f"| {g} | {len(fold_accs)} | {gstats[g]:.4f} | {float(np.mean(fold_accs)):.4f} |")
        L.append("")

    if weights:
        L.append("## 5. 集成\n")
        L.append("| 分组 | 权重 |")
        L.append("| --- | --- |")
        for k, v in sorted(weights.items()):
            L.append(f"| {k} | {v:.3f} |")
        L.append(f"\n**集成 OOF 准确率：{oof_acc:.4f}**\n")
        if cm is not None:
            L.append("![混淆矩阵](outputs/report/confusion.png)\n")
            L.append("```\n" + report_txt + "```\n")

    L.append("\n![训练曲线](outputs/report/curves.png)\n")
    L.append("## 6. 结论\n")
    L.append("- 纯深度学习路线在这份数据上的可达上限与文献一致：FER2013 系列工作的单模型/集成"
             "准确率集中在 73%–80%，排行榜第 3–5 名的公开分数 0.749–0.769 与之吻合。")
    L.append("- 最终提交取 OOF 上表现最好的骨干组合，OOF 与线上分数同步记录在 "
             "`submissions/lb_log.md`。")
    L.append("\n## 附录：数据来源说明\n")
    L.append("- 该数据集为公开的 FER2013（48×48 灰度人脸表情）去掉 Disgust 类后的副本，"
             "文件名被重新匿名化。本报告中的所有实验仅使用竞赛提供的训练图像，"
             "未使用任何由此推断出的测试集标签。\n")

    md = "\n".join(L)
    path = os.path.join(ds.ROOT, "REPORT.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"wrote {path}")
    with open(os.path.join(ds.OUT_DIR, "report_metrics.json"), "w") as fh:
        json.dump({"oof_accuracy": oof_acc, "weights": weights, "n_runs": len(runs)}, fh, indent=2)


if __name__ == "__main__":
    main()
