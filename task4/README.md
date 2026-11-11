# 任务四 · DRIVE 视网膜血管分割

将[课程数据](https://www.kaggle.com/competitions/2026task-4)放在 `task4/2026task-4/segmentation/`，保留 `train/image/`、`train/label/` 和 `test/image/`。训练与阈值选择只使用课程训练标注，测试图用于推理。

## FR-UNet 推理

在 `task4/` 下运行。将[上游 DRIVE 第 40 轮权重](https://github.com/lseventeen/FR-UNet/tree/1856bc206055fdfccde184d0ebf9ac8c3207387d/pretrained_weights/DRIVE)下载到 `weights/fr_unet_drive_epoch40.pth`，然后封装权重并生成 RLE 提交：

```shell
python -m drive_vessel pack-initial --config configs/fr_unet_official_patch_drive.yaml --output weights/fr_unet_drive_epoch40_packaged.pt
python inference.py --output submission_reproduced.csv
```

`inference.py` 优先读取本地存档概率。干净检出在缺少存档时从封装权重推理。最终配方使用 512×512 输入、双阈值 0.65/0.40、视野掩膜和 16 像素连通域面积下限。冻结提交的公开榜为 0.81766 Dice，私有榜为 0.82534 Dice，配置与哈希见 [BEST_CONFIG](BEST_CONFIG) 和[提交记录](../report/data/kaggle_scores.csv)。跨环境从权重重新推理时，阈值附近的结果可能略有不同。

## 原生分辨率 U-Net 对照

用绿色通道的 CLAHE 结果和 Frangi 响应作双通道输入，在视野内采样 patch。验证汇总补边整图的像素计数。

```shell
python scripts/prepare_native_drive.py
python vendor/reti_ves_seg/train.py --drive-root external_data/drive_native --arch unet --epochs 40 --batch-size 16 --patch-size 64 --patches-per-epoch 8000 --out runs/native/model.pth
python scripts/infer_native_reti.py --checkpoint runs/native/model.pth --split test --threshold 0.35 --output runs/native_test
python -m pytest -q
```

[原生日志](../report/data/drive/native_model_log.csv)第 15 轮的补边整图验证 Dice 为 0.80780。其他对照见[结果汇总](RESULTS.md)和[实验日志](EXPERIMENT_LOG.csv)，上游代码和权重归属见[第三方说明](THIRD_PARTY_NOTICES.md)。
