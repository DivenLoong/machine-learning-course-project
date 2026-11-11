# 实验事实与证据审计

核查日期为 2026 年 11 月 11 日。路径以综合仓库为根。原任务目录中的历史报告用于恢复早期对照，训练配置、代码和逐轮日志用于判断实现，Kaggle 提交历史用于确认线上分数。表中“历史汇总”表示已找到原记录，且当前材料不足以完整重算对应的融合或重现训练。

## 共同证据和口径

- [Kaggle 提交记录](data/kaggle_scores.csv)从已登录账户逐赛题查询，仅保留完成状态的提交编号、文件名、UTC 时间、Public/Private 和可核对的本地文件哈希。
- [数据计数](data/class_counts.json)由原目录按类别统计。源图像、逐样本真实标签、预测文件和模型权重在原工作区保留。源码仓库发布汇总证据。
- 数值来源分为直接观测、从已有文件计算和历史汇总。线上与本地口径、样本内与嵌套验证的结果在报告中分别标明。

## 任务一 · 植物幼苗细粒度分类

| 核查项 | 事实 | 证据 | 状态 |
| --- | --- | --- | --- |
| 数据与指标 | 500 张训练图、5 类均衡，378 行测试提交。OOF 使用准确率和宏 F1。赛题标称 F1，得分粒度与每张等权相符 | [plantcls.py](../task1/plantcls.py)、[计数](data/class_counts.json)、原样例提交及 Kaggle 记录 | 样本和代码已核，平台聚合细节待确认 |
| 预处理 | 透明通道合成、HSV 框、先裁剪后缩放，缓存需重建。旧路径曾在缩放图上使用原坐标 | [plantcls.py](../task1/plantcls.py)、[diag_black.py](../task1/diag_black.py)、原论文生成脚本及缺陷图 | 已核 |
| 骨干与训练 | ConvNeXt Tiny/Small 和 EfficientNet-B0，分层五折；冻结头阶段后解冻微调。五组实际训练分别为 50 或 44 轮。各折中间峰值单独记录，末轮权重生成 OOF | [train.py](../task1/train.py)、[修复后运行汇总](data/plant/) | 已核 |
| 对照 | 修复后 B0 0.922、Tiny 0.932、Tiny 伪标签 0.942、Small 0.950、Tiny 384 0.928；修复前后集成 0.9300 到 0.9600 | [修复后运行汇总](data/plant/)、原 README、原论文中的对照表 | 单组从运行汇总核实；0.9600 为历史集成记录 |
| 最终提交 | v16（56983147）Public/Private 0.9735；v13 与 v16 本地文件字节一致 | [Kaggle 记录](data/kaggle_scores.csv)、原 CSV | 已核；各阶段历史权重配比及先验文件精确映射待确认 |
| 外部同源数据 | v10（56867229）Public/Private 0.9947。原进度记录写明 41/378 逐像素匹配的风险 | [Kaggle 记录](data/kaggle_scores.csv)、原 `PROGRESS.md` 的公开同源数据段落 | 分数已核；重复计数标作历史审计记录 |

原实验主要证据包括任务一 `outputs/fix_*/summary.json`、逐折 `fold*_history.json`、`outputs/test_probs_*.npz`、`build_paper.py` 的对照表与本地 Kaggle CSV。图中逐类 F1 使用早期 OOF 汇总，版本在正文已注明。 在干净导出目录运行 `apply_prior.py`，输入原 `outputs/test_probs.npz`、`outputs/blend.json` 和课程样例 CSV，重现 378 行 v16 的完整字节。不同的概率、先验组合也能得到相同标签，故精确历史成员与调用仍待确认。

## 任务二 · 面部表情分类

| 核查项 | 事实 | 证据 | 状态 |
| --- | --- | --- | --- |
| 数据 | 28,275 训练 / 7,065 测试，六类 48×48 灰度。分层划分默认 seed 42 | [ds.py](../task2/src/ds.py)、[计数](data/class_counts.json)、课程目录 | 已核 |
| 训练 | ConvNeXt-Small 224 第 0 折计划与实际均 48 轮，批 32，运行 seed 0。EMA 最优记录在索引 11，0.7301503 | [运行配置、日志、摘要](data/fer/cns224_f0/)、[train.py](../task2/src/train.py) | 已核 |
| 分辨率消融 | Tiny 第 0 折 112 和 224 的保存权重准确率约 0.7144 与 0.7204，耗时 13.89 与 49.64 分钟 | [ablation_res.json](data/fer/ablation_res.json) | 已核，单折结果 |
| 融合 | v2 的 OOF 0.7371883，概率 bagging 的历史 OOF 0.7393；logit 0.7397 对应线上 0.7460 | [v2 权重](data/fer/blend_v2_cns224.json)、[搜索代码](../task2/src/blend_search.py)、原 `submissions/lb_log.md` | v2 有权重记录；后两项 OOF 来自历史汇总 |
| 线上 | v2（57308701）与 bagged（57332301）均为 0.7470 | [Kaggle 记录](data/kaggle_scores.csv)、原 CSV | 已核 |

训练损失与验证曲线来自同一折。表情图片案例由 `val_idx.npy` 和 `val_probs.npy` 对应到课程训练图。宏 F1 0.7227 以及逐类召回为原 `REPORT.md` 的旧贪心汇总。

## 任务三 · 眼底中心凹定位

| 核查项 | 事实 | 证据 | 状态 |
| --- | --- | --- | --- |
| 数据与坐标 | 80 训练、20 测试；编号限制 1 至 80；X/Y 元素合并后的原图像素 RMSE | [common.py](../task3/solution/src/common.py)、[坐标校验](../task3/solution/src/validate_submission.py)、原样例 | 已核 |
| 热图 | B1 四通道输入、U 形解码、输出步幅 4、软圆盘。圆盘分支实际使用 BCE+Dice+质心 L1 | [model.py](../task3/solution/src/model.py)、[data.py](../task3/solution/src/data.py)、[train.py](../task3/solution/src/train.py) | 已核 |
| 训练与选择 | seeds 777、123、2026 各 10 折。每折实际 60 轮，按验证 RMSE 保存最优模型。seed 123 第 0 折最佳第 22 轮 | [配置和逐轮记录](data/fovea/)、[分折统计](data/fovea/fold_errors.csv) | 已核，分折统计由保存坐标重算 |
| 本地融合 | v11 13.357 px，v12 13.197 px；v15 样本内 12.994 px、嵌套 13.190 px | 原 `solution/reports/UPDATE_2026-10-27_18.md`、[VascX 推理](../task3/solution/src/vascx_fovea.py) | 历史汇总；v15 的分档阈值与原始生成命令待确认 |
| 最终提交 | v15（57701069）Public 9.90927 px，Private 19.54239 px | [Kaggle 记录](data/kaggle_scores.csv)、原 v15 文件 | 已核 |

分折样本数为 7 至 9，因为重复内容按组划分。用每折原坐标残差计算 `sqrt(mean(dx²+dy²)/2)`。同一折的最大样本平方误差占比分母为该折平方误差总和。VascX 权重是外部来源，最终几何融合的参数由历史运行恢复至能证实的粒度。

## 任务四 · DRIVE 血管分割

| 核查项 | 事实 | 证据 | 状态 |
| --- | --- | --- | --- |
| 课程数据与评分 | 20 张课程训练图与 20 张测试图；本地 Dice、阈值、FOV、RLE 均有实现 | [data.py](../task4/drive_vessel/data.py)、[metrics.py](../task4/drive_vessel/metrics.py)、[rle.py](../task4/drive_vessel/rle.py) | 已核 |
| 主线结构 | FR-UNet 四级分辨率，五个全分辨率头 logits 平均；上游 DRIVE 第 40 轮权重 | [models.py](../task4/drive_vessel/models.py)、[BEST_CONFIG](../task4/BEST_CONFIG)、[上游许可](../task4/licenses/FR-UNet.txt) | 已核；上游逐轮训练记录待确认 |
| 推理 | 512 缩放，无 TTA，DTI 0.65/0.40，FOV 掩膜、最小连通域面积 16 | [inference.py](../task4/inference.py)、[后处理实现](../task4/drive_vessel/inference.py) | 已核 |
| 冻结提交 | 编号 57946948。Public 0.81766 / Private 0.82534，冻结快照与原根目录 submission.csv 字节一致 | [Kaggle 记录](data/kaggle_scores.csv)、原冻结快照 | 已核 |
| 原生 U-Net | 40 轮，最佳第 15 轮 0.80780。训练在 FOV 内采样；现存验证函数累计补边整图的 TP/FP/FN，调用未传 FOV 掩膜 | [独立原生日志](data/drive/native_model_log.csv)、[训练器](../task4/vendor/reti_ves_seg/train.py)、[指标](../task4/vendor/reti_ves_seg/losses.py) | 已核；历史摘要使用“FOV Dice”一词，正文按代码解释 |
| 另一 U-Net | `e030_small_unet_frangi_f0` 内部训练日志有 11 个 epoch，最佳第 8 轮；与原生模型独立 | [内部日志](data/drive/internal_small_unet_history.json)、[配置](../task4/configs/small_unet_frangi.yaml) | 已核 |
| 负面结果 | 原生阈值 0.35 为 0.78848，75/25 融合为 0.80897，关闭 DTI 的候选为 0.77485 | [实验日志](../task4/EXPERIMENT_LOG.csv)、[Kaggle 记录](data/kaggle_scores.csv) | 已核；阈值 0.50 的 0.75079 保留历史记录 |

官方 FR-UNet checkpoint 源文件 SHA-256 为 `5fbfbc42c4801161b1a03eb73596934c3375e1a8a6fe70a85225ef028595015c`。本地封装文件为 `492d6393c39c51caf142a521ef2d4ff602bb3455fc1bd1fdb78dbf37b82f4fc5`。冻结提交的 SHA-256 为 `1b4415c5293e17c219b9719fa274714ee233e634dc17f511b1178e685f917fb6`。
