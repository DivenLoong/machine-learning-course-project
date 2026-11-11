# 任务三 · 眼底中心凹定位

从[课程赛题](https://www.kaggle.com/competitions/2026task3)获取数据。任务目录中保留 `2026task3/detection/{train,test}`、`fovea_localization_train_GT.csv` 和 `sample_submission.csv`。后两项位于 detection 下。可用 `FOVEA_DATA_DIR` 覆盖 detection 路径。训练代码只读取编号 1 至 80 的标注。

在 `task3/solution` 中执行：

```shell
python src/train.py --backbone efficientnet_b1 --fold 0 --target circle --size 512 --epochs 60 --out runs/rgb_demo
python src/predict.py --run runs/rgb_demo --out submissions/rgb_demo.csv
python src/validate_submission.py submissions/rgb_demo.csv
```

血管输入需要 [Fundus Image Toolbox](https://github.com/berenslab/fundus_image_toolbox)及其上游分割权重。安装该包并在同目录运行 `python src/make_vessel.py`，生成 `cache/vessel`。主线两组分别使用 `folds10_seed123.json` 和 `folds10_seed777.json`。其中一组的示例：

```shell
python src/train.py --backbone efficientnet_b1 --fold -1 --n-splits 10 --folds-file folds10_seed123.json --seed 123 --target circle --use-vessel --aug strong --lr 0.0001 --epochs 60 --out runs/exp030_vessel_10fold_seed123
python src/predict.py --run runs/exp030_vessel_10fold_seed123 --use-vessel --out submissions/vessel_seed123.csv
```

VascX 分支需要上游预处理包 `rtnls_fundusprep` 及其公开中心凹 TorchScript 权重，从 [VascX 官方权重库](https://github.com/Eyened/retinalysis-vascx-models)取得模型，从[官方预处理包](https://github.com/Eyened/retinalysis-fundusprep)安装预处理依赖或放置 `external/vascx_src/rtnls_fundusprep`，使用以下入口。Windows TorchScript 使用 ASCII 路径传递权重。

```shell
python src/vascx_fovea.py --test --size 1024 --sliding --checkpoint path/to/fovea_may26.pt --out vascx_test.csv
```

v15 的文件哈希、公开榜 9.90927 px 和私有榜 19.54239 px 已核对。历史记录可确认距离分档融合，完整分档参数与原始生成命令待确认。已保留全部三组十折的训练曲线和相应配置。
