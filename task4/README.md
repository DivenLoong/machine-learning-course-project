# 任务四 · DRIVE 视网膜血管分割

从[课程赛题](https://www.kaggle.com/competitions/2026task-4)取得数据，放到 `task4/2026task-4/segmentation`。项目期望其中包含 `train/image`、`train/label`、`test/image` 和课程样例提交。正式路线使用课程训练标签和测试图像，课程测试图只用于推理。

## 冻结 FR-UNet 推理

在 task4 目录执行。将 [FR-UNet 上游的 DRIVE epoch 40 文件](https://github.com/lseventeen/FR-UNet/tree/1856bc206055fdfccde184d0ebf9ac8c3207387d/pretrained_weights/DRIVE)放至 `weights/fr_unet_drive_epoch40.pth`，然后封装并推理：

```shell
python -m drive_vessel pack-initial --config configs/fr_unet_official_patch_drive.yaml --output weights/fr_unet_drive_epoch40_packaged.pt
python inference.py --output submission_reproduced.csv
```

最终设置见 [BEST_CONFIG](BEST_CONFIG)，冻结提交 SHA-256 为 `1b4415c5293e17c219b9719fa274714ee233e634dc17f511b1178e685f917fb6`。跨 PyTorch、CUDA、图像处理库版本的重算可能出现阈值附近的数值差异。Kaggle 原提交的公开榜 Dice 0.81766、私有榜 0.82534。

## 代表性自训对照

以下命令对应独立的原生 CLAHE+Frangi U-Net。使用 FOV 内 patch 采样，验证按代码累加完整补边图像的像素计数。

```shell
python scripts/prepare_native_drive.py
python vendor/reti_ves_seg/train.py --drive-root external_data/drive_native --arch unet --epochs 40 --batch-size 16 --patch-size 64 --patches-per-epoch 8000 --out runs/native/model.pth
python scripts/infer_native_reti.py --checkpoint runs/native/model.pth --split test --threshold 0.35 --output runs/native_test
python -m pytest -q
```

[原生训练日志](../report/data/drive/native_model_log.csv)第 15 轮最佳 Dice 0.80780。冻结模型和各候选见 [RESULTS.md](RESULTS.md) 和 [EXPERIMENT_LOG.csv](EXPERIMENT_LOG.csv)。上游许可与模型归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 [licenses](licenses/)。
