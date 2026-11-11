# 任务二 · 面部表情分类

课程数据有 28,275 张训练图和 7,065 张测试图，均为 48×48 灰度人脸图像。实验比较六类表情分类的输入分辨率、骨干和融合方法。将[课程数据](https://www.kaggle.com/competitions/2026task2)放在 `task2/2026task2/` 下，图像位于 `fer_data/fer_data/{train,test}/`，样例提交为 `submission.csv`。训练图按类别分目录。也可把 `FER_PROJECT_ROOT` 指向含 `2026task2/` 的目录。

## 单折训练与推理

在 `task2/` 下运行，需要 CUDA。这里用 ConvNeXt-Small、224 像素输入和第 0 折。

```shell
python src/train.py --backbone convnext_small.fb_in22k_ft_in1k --res 224 --epochs 48 --fold 0 --tag demo_cns224_f0
python src/predict.py --tag demo_cns224_f0
```

每折的验证预测和测试概率保存在 `outputs/runs/<tag>/`。完成第 1 至 4 折时，分别修改 `--fold` 和 `--tag`。融合脚本按骨干与分辨率分组，需要每个入选组覆盖全部验证样本。

```shell
python src/ensemble.py --name blend
python src/validate_sub.py submissions/sub_blend.csv
```

`src/blend_search.py` 另有概率空间和 logit 空间的融合对照。报告中的 v2 和概率 bagging 均为 0.7470，见[v2 融合权重](../report/data/fer/blend_v2_cns224.json)和[提交记录](../report/data/kaggle_scores.csv)。[原训练日志](../report/data/fer/cns224_f0)记录了 224 像素第 0 折保存的 EMA 权重。ResEmoteNet 分支按[原论文](https://arxiv.org/abs/2409.10545)适配六类输出。
