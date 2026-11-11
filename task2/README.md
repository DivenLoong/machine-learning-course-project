# 任务二 · 面部表情分类

从[课程赛题](https://www.kaggle.com/competitions/2026task2)获取 6 类表情数据。进入 task2，将训练目录放在 `2026task2/fer_data/fer_data/train/<类别>/*.jpg`、测试图放在对应的 `test/*.jpg`，官方样例为 `2026task2/submission.csv`。可用环境变量 `FER_PROJECT_ROOT` 指向含这份目录结构的其他项目目录。

```shell
python src/train.py --backbone convnext_small.fb_in22k_ft_in1k --res 224 --epochs 48 --fold 0 --tag cns224_f0
python src/predict.py --tag cns224_f0
```

按同样命令依次跑完 fold 1 至 4，并为每折改变 `--tag`。如准备多个骨干，按其真实配置训练并生成全部折的验证和测试概率。先运行 `python src/ensemble.py --name blend` 做组级贪心融合，也可运行 `python src/blend_search.py` 产生概率与 logit 空间候选。已有提交可以用 `python src/validate_sub.py submissions/sub_bagged.csv` 检查。

Small 224 第 0 折的原始训练轨迹、48 轮配置和保存的 EMA 峰值见 [cns224_f0](../report/data/fer/cns224_f0)。v2 和概率 bagging 并列 0.7470。ResEmoteNet 的本地实现依据[原论文](https://arxiv.org/abs/2409.10545)适配六类输出。
