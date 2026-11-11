# 机器学习课程设计综合实验

四项视觉任务分别包含数据处理、训练、验证与提交入口。各任务使用对应的课程数据、评价指标和工作目录。

| 任务 | 主线 | Kaggle Public | Kaggle Private |
| --- | --- | ---: | ---: |
| [植物幼苗分类](task1/README.md) | 前景裁剪修复，ConvNeXt / EfficientNet 五折集成 | 0.9735 | 0.9735 |
| [面部表情分类](task2/README.md) | ConvNeXt 等骨干与折外概率融合 | 0.7470 | 0.7470 |
| [眼底中心凹定位](task3/README.md) | EfficientNet-B1 热图、血管图与 VascX 坐标融合 | 9.90927 px | 19.54239 px |
| [DRIVE 血管分割](task4/README.md) | FR-UNet 官方权重、双阈值迭代及 FOV 过滤 | 0.81766 Dice | 0.82534 Dice |

提交编号、文件名与哈希见[平台记录](report/data/kaggle_scores.csv)。各任务的折外验证、阈值对照和历史记录限制见[证据审计](report/evidence_audit.md)。

## 环境和数据

验证环境使用 Windows 上的 Python 3.13、PyTorch 2.14.0+cu126 与 CUDA GPU。建议新建虚拟环境，按 [PyTorch 安装说明](https://pytorch.org/get-started/locally/) 选择与本机 CUDA 匹配的 torch 和 torchvision，再运行：

```shell
python -m pip install -r requirements.txt
```

课程数据从各任务链接或教师发放渠道获得。训练图像、测试图像、模型权重、课程提交文件和含学生信息的 PDF 为本地材料。各任务 README 给出相对路径、取得上游权重的方法和训练及推理命令。训练命令按任务指定的目录执行。

## 报告与核查

- [LaTeX 主文件](report/main.tex)与[复现附录](report/chapters/reproduction.tex)。公开源码可编译去个人资料版，正式 PDF 随课程材料交付。
- [图表脚本](report/scripts/make_figures.py)从仓库内日志和 CSV 生成图形。
- [提交核查脚本](report/scripts/check_submissions.py)接受本地原始课程目录，检查真实提交的顺序、坐标、RLE 和哈希。
- [逐页质量检查](report/visual_quality_check.md)记录正式版的图表来源与编译检查。

```shell
python report/scripts/make_figures.py
cd report
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

报告依赖 XeLaTeX、BibTeX、Fandol、ctex、gbt7714 与 TikZ。Windows 验证使用 Ubuntu-22.04 WSL 的 TeX Live。课程代码及模型归属详见各任务说明和 [第三方许可](task4/licenses/)。
