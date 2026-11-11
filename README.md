# 机器学习课程设计

四项图像任务的训练、推理代码与综合实验报告。下表采用报告所列提交的 Kaggle 成绩。定位用原图坐标 RMSE，越低越好。

| 任务 | 主要方法与实验 | 公开榜 | 私有榜 |
| --- | --- | ---: | ---: |
| [植物幼苗分类](task1/README.md) | 修复裁剪与缩放顺序，比较 ConvNeXt、EfficientNet 和五折集成 | 0.9735 | 0.9735 |
| [面部表情分类](task2/README.md) | 比较输入分辨率，用折外预测选择 ConvNeXt 等骨干的融合权重 | 0.7470 | 0.7470 |
| [眼底中心凹定位](task3/README.md) | 用 EfficientNet-B1 预测热图，比较血管输入与 VascX 坐标融合 | 9.90927 px | 19.54239 px |
| [DRIVE 血管分割](task4/README.md) | 用 FR-UNet 官方权重推理，比较双阈值、视野过滤和原生分辨率 U-Net | 0.81766 Dice | 0.82534 Dice |

提交编号、文件名和哈希见 [Kaggle 记录](report/data/kaggle_scores.csv)。各任务的本地验证口径和历史记录缺项见[证据审计](report/evidence_audit.md)。

## 目录

```text
task1/            幼苗分类
task2/src/        表情分类
task3/solution/   中心凹定位
task4/            血管分割、配置与测试
report/           LaTeX、实验记录和图表
requirements.txt  四项任务的依赖
```

## 环境与运行

在 Windows、Python 3.13 和 PyTorch 2.14.0+cu126 环境核验过。按本机 CUDA 版本安装 [PyTorch 和 torchvision](https://pytorch.org/get-started/locally/)，再从仓库根目录安装其余依赖：

```shell
python -m pip install -r requirements.txt
```

按各任务说明放置从赛题或课程渠道取得的数据。表情分类的训练与推理需要 CUDA，中心凹任务的血管图和 VascX 分支还需要对应的上游依赖与权重。

| 工作目录 | 入口 |
| --- | --- |
| `task1/` | `train.py` 训练，`predict.py` 生成类别提交 |
| `task2/` | `src/train.py` 按折训练，`src/predict.py` 生成测试概率，`src/ensemble.py` 融合 |
| `task3/solution/` | `src/train.py` 训练热图模型，`src/predict.py` 输出原图坐标 |
| `task4/` | `inference.py` 运行 FR-UNet，`vendor/reti_ves_seg/train.py` 训练原生 U-Net 对照 |

