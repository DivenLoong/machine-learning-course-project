# 任务一 · 植物幼苗分类

从[课程赛题](https://www.kaggle.com/competitions/2026task1)取得数据。进入 task1 目录，放置 `dataset-for-task2/train/<类别>/*`、`dataset-for-task2/test/*` 与根目录的 `submission-for-task2.csv`。五类训练图各 100 张，提交为 378 行的 ID 与类别。

```shell
python eda.py
python build_cache.py
python train.py --tag fix_cnx320 --arch convnext_tiny.fb_in22k_ft_in1k --size 320 --head-epochs 4 --ft-epochs 40 --batch-size 32 --raug --mixup 0.2 --cutmix 1.0 --lr 0.0002
python predict.py --tags fix_cnx320 --tta-flip --out submission_demo.csv
```

`predict.py` 写出 `outputs/test_probs.npz`。需要验证伪标签阶段时，将该文件路径传入 `train.py --pseudo outputs/test_probs.npz --pseudo-thresh 0.9`，为不同阶段使用独立的运行标记。融合时将多个完整运行标记以逗号分隔传入 `predict.py --tags`。先验可用 `blend.py` 估计，再通过 `apply_prior.py --probs … --prior-json … --out …` 生成提交。

五折结果与代表性历史配置见 [report/data/plant](../report/data/plant)。最终引用的 v16 在两侧榜单均为 0.9735。v13 与 v16 的本地 CSV 哈希一致。旧报告的四配置记录与文件名可查，精确推理命令和先验文件对应待确认。同源库版本的 0.9947 具有测试图重叠风险，详见证据审计。原任务的独立代码记录见[植物分类仓库](https://github.com/DivenLoong/plant-seedling-classification)。
