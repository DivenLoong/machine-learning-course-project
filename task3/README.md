# 任务三 · 眼底中心凹定位

模型预测中心凹热图，再将预测点换算到原图坐标。将[课程数据](https://www.kaggle.com/competitions/2026task3)放在 `task3/2026task3/detection/` 下，保留 `train/`、`test/`、`fovea_localization_train_GT.csv` 和 `sample_submission.csv`。可用 `FOVEA_DATA_DIR` 指向其他数据目录。训练只使用编号 1 至 80 的标注。

## RGB 基线

在 `task3/solution/` 下运行。下面训练第 0 折并检查生成的坐标文件：

```shell
python src/train.py --backbone efficientnet_b1 --fold 0 --target circle --size 512 --epochs 60 --out runs/rgb_demo
python src/predict.py --run runs/rgb_demo --out submissions/rgb_demo.csv
python src/validate_submission.py submissions/rgb_demo.csv
```

## 血管输入与 VascX

按 [Fundus Image Toolbox](https://github.com/berenslab/fundus_image_toolbox) 的说明安装工具及分割权重，再执行 `python src/make_vessel.py` 生成 `cache/vessel/`。以下命令运行随机种子 123 的十折实验：

```shell
python src/train.py --backbone efficientnet_b1 --fold -1 --n-splits 10 --folds-file folds10_seed123.json --seed 123 --target circle --use-vessel --aug strong --lr 0.0001 --epochs 60 --out runs/exp030_vessel_10fold_seed123
python src/predict.py --run runs/exp030_vessel_10fold_seed123 --use-vessel --out submissions/vessel_seed123.csv
```

另一组对应 `folds10_seed777.json` 和种子 777，使用独立输出目录。VascX 依赖[预处理包](https://github.com/Eyened/retinalysis-fundusprep)和[上游中心凹权重](https://github.com/Eyened/retinalysis-vascx-models)。下列命令中的权重路径应换成本地路径，Windows 下为 TorchScript 权重选用 ASCII 路径。

```shell
python src/vascx_fovea.py --test --size 1024 --sliding --checkpoint path/to/fovea_may26.pt --out cache/vascx_test.csv
```

报告中的 v15 使用血管模型和 VascX 坐标的距离分档融合，公开榜 9.90927 px、私有榜 19.54239 px。现有文件可核对[各折配置、日志与误差](../report/data/fovea)以及 v15 的提交哈希，分档阈值和原始融合命令仍待确认。
