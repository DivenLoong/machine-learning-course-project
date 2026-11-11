# 任务一 · 植物幼苗分类

[课程数据](https://www.kaggle.com/competitions/2026task1)包含五类共 500 张训练图和 378 张测试图。在 `task1/` 下放置 `dataset-for-task2/train/<类别>/`、`dataset-for-task2/test/` 和任务根目录的样例 `submission-for-task2.csv`。

## 训练与推理

下面以修复裁剪后的 ConvNeXt-Tiny 配置为例。训练按五折保存权重，推理对各折和增强视图的概率取平均。

```shell
python eda.py
python build_cache.py
python train.py --tag demo_cnx320 --arch convnext_tiny.fb_in22k_ft_in1k --size 320 --head-epochs 4 --ft-epochs 40 --batch-size 32 --raug --mixup 0.2 --cutmix 1.0 --lr 0.0002
python predict.py --tags demo_cnx320 --tta-flip --out submission_demo.csv
```

多配置融合时，把完整的运行标记用逗号连接传给 `--tags`。推理会写入 `outputs/test_probs.npz`。若用于伪标签训练，先留存这份概率，再以新标记训练并传入 `--pseudo` 路径和 `--pseudo-thresh 0.9`。

## 核对报告中的 v16

若保留了原课程运行的 `outputs/test_probs.npz` 和 `outputs/blend.json`，可用先验校正脚本和样例 CSV 重算提交：

```shell
python apply_prior.py --probs outputs/test_probs.npz --prior-json outputs/blend.json --out submission_v16_check.csv
```

已用 SHA-256 `8d1d42ed430588da8838f93fefdec165e8e813ad5838db6981c8cabed201dcb2` 验证重算的 378 行。v13 与 v16 的本地文件字节相同，v16 的历史成员权重与原始命令仍待确认。同源外部数据版本的 0.9947 涉及测试图重叠风险。参见[五折记录](../report/data/plant)、[证据审计](../report/evidence_audit.md)和[原任务仓库](https://github.com/DivenLoong/plant-seedling-classification)。
