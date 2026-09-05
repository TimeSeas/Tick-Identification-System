# CLAUDE.md

本项目用于指导 Claude Code (claude.ai/code) 在本仓库中的工作方式。

## ⚠️ 毕业设计自主完成原则（最高优先级）

本项目是用户的**本科毕业设计**（2026-09 起）。用户要求**自主完成代码**：

- **Claude 不代写功能代码**。涉及功能开发时，只提供：方案设计、步骤指导、原理解释、代码评审、答疑；代码由用户亲手写
- 例外：项目文档与上下文维护（CLAUDE.md、docs、memory）、以及用户明确点名让 Claude 修的具体 bug/脚本，可以动手
- 毕业设计整体规划见 `毕业设计/毕业设计系统规划.md`（8 个工作包 + 时间线 + 论文结构）

## 项目概述

蜱虫与叮咬伤口智能识别系统（本科毕业设计）- 上传照片，系统回答：有没有蜱虫（或是什么昆虫）、什么蜱种、叮咬伤口在哪里、风险建议是什么。三模型架构：M1 多类别昆虫检测（在单类 mAP@0.50=0.842 基础上扩展，降误检）、M2 蜱种精细分类（TickIDNet 3 物种）、M3 叮咬伤口分割（tick-bite 数据集）。成果形态：Web 应用为主 + 桌面 GUI / 边缘部署可选。

路线图（毕设 WP，详见 `毕业设计/毕业设计系统规划.md`）：
- WP1-2：蜱种数据收集（TickIDNet 75 样例先导 + iNaturalist 扩展）与 3 类分类模型
- WP3：叮咬伤口分割（tick-bite 数据集直接训练）
- WP4：多类别昆虫检测（SAM 试点扩量，误检率对比实验）
- WP5-8：三模型级联集成、Web 应用、桌面 GUI（可选）、边缘部署（可选）

## 环境配置

```bash
# 使用 conda 创建环境（主力路径，cu118 + detectron2 预编译 wheel）
conda env create -f environment.yml
conda activate tick_detection
```

- 本机已安装好 `tick_detection` 环境，日常直接 activate 即可
- `setup_env_anaconda.ps1` 创建同名环境，但 torch CUDA 12.1 + detectron2 git 源码，与 environment.yml 不一致，仅作备选
- `setup_env.bat` / `setup_env.ps1` 创建 venv `tick_env`（非 conda），历史脚本
- ⚠️ `requirements.txt` 是历史 pip freeze 快照，不含 torch/torchvision/detectron2 等核心依赖，不能单独用于建环境

## 常用命令

### 训练

```bash
# 单类别蜱虫检测（默认 2 类 = 背景 + 蜱虫）
python src/train/train_torchvision.py
```

多类别检测尚未实现（见「多类别支持」），`--multi-class` / `--num-classes` 参数目前仅占位。

### 推理

```bash
# 单张图片
python detect.py <图片路径> -o output.jpg -c 0.5

# 文件夹批量处理（输入目录硬编码在脚本顶部，默认 archive3/farm_insects/Aphids，需编辑脚本修改）
python detect_folder.py

# 测试集批量推理
python src/inference/inference_demo.py
```

### 评估与测试

```bash
# 模型评估（torchvision + COCOeval，mAP 来源）
python src/eval/evaluate_model.py

# 误检率测试（archive3 蚜虫/叶螨/蓟马/菜青虫）
python test_false_positives.py

# 数据集格式校验
python scripts/verify_coco_dataset.py

# 数据诊断：打印训练集首条标注的 segmentation 字段（排查多边形问题）
python src/train/检查数据集有无多边形.py
```

### SAM 数据增强

```bash
# 主数据集掩码补全（bbox 提示 → 多边形写回，原文件备份为 _annotations.coco.bbox_only.json）
python scripts/sam_auto_label.py --mode fill --coco "ticks image detection.v5i.coco/train/_annotations.coco.json" "ticks image detection.v5i.coco/valid/_annotations.coco.json" "ticks image detection.v5i.coco/test/_annotations.coco.json"

# 新数据自动标注（AMG，文件夹名即类别；产出 COCO + X-AnyLabeling 复核副本 + 预览）
python scripts/sam_auto_label.py --mode auto --input-dirs archive2/dataset/train archive3/farm_insects --classes tick mite aphid colorado_potato_beetle thrips corn_earworm fruit_fly --per-class 30 --out-dir output/sam_pilot
```

两模式均支持断点续跑（被杀/中断后重跑自动跳过已完成部分）。

## 关键路径

| 组件 | 路径 |
|------|------|
| 蜱虫数据集（COCO格式） | `ticks image detection.v5i.coco/` |
| 训练脚本（主力，torchvision） | `src/train/train_torchvision.py` |
| 训练脚本（备选，MMDetection） | `src/train/train_mask_rcnn.py` + `configs/mask_rcnn_ticks.py` |
| detectron2 配置与注册（历史备选） | `configs/train_mask_rcnn.yaml`、`src/data/register_datasets.py`、`src/detectron2/`（vendored） |
| 模型评估 | `src/eval/evaluate_model.py`（torchvision）、`src/eval/evaluate.py`（detectron2） |
| ONNX 导出 | `src/deploy/export_onnx.py` |
| 单张图片推理 | `detect.py` |
| 文件夹批量推理 | `detect_folder.py`、`src/inference/inference_demo.py` |
| 模型权重 | `output/mask_rcnn_ticks.pth`（训练好的蜱虫模型） |
| COCO 预训练权重 | `checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth`（不是训练好的蜱虫模型） |
| TickIDNet（蜱种数据） | `TickIDNet/Sample Images/`（3 物种 75 张样例）、`TickIDNet/Data/All_data.csv`（12177 行元数据）、`TickIDNet/Confidence Thresholds/`（分物种置信阈值）、`models/tickidnet_model/`（官方 TF SavedModel） |
| 蜱种图片数据集（2026-09-03 下载） | `data/tick_species/`（**173 物种 7623 张**，文件夹即类别：GBIF 3 物种包 + GBIF API 12 物种 + HF fine-grained-challenges Ixodidae 4499 张；学名↔中文名对照表 `物种中文名对照表.csv`；全局清单 `fgc_ixodidae_manifest.csv`；原始包 `data/tick_species_raw/`） |
| 叮咬伤口数据集（COCO） | `tick-bite-improved.v7-random_split_resize_640.coco-segmentation/`（1032 图 / 1806 多边形标注） |
| 标注指南 | `docs/annotation_guide.md` |
| 论文支撑材料 | `论文研究材料/SAM自动标注原理与效果分析.md`（原理、实验结果、原因分析） |
| 调参教程 | `调参教程/SAM标注代码修改与调参教程.md`（参数速查、改码练习） |
| 毕业设计规划 | `毕业设计/毕业设计系统规划.md`（选题、架构、8 个 WP、论文结构、进度对照） |
| 自动标注AI模型 | `models/yolov8m.pt`、`models/yolov8m.onnx`、`models/classes.txt`（28类）、`models/yolov8_custom.yaml` |
| SAM 权重与脚本 | `models/sam_vit_b_01ec64.pth`（vit_b，375MB）、`scripts/sam_auto_label.py`（fill/auto 两模式） |
| 数据工具脚本 | `scripts/verify_coco_dataset.py`、`scripts/convert_folder_to_coco.py`、`scripts/download_gbif_images.py`（GBIF 归档包图片下载）、`scripts/download_gbif_species.py`（GBIF API 按物种名下载，任意蜱种）、`scripts/extract_fgc_ixodidae.py`（HF lance 解包）——均断点续跑 |
| 商业计划书生成 | `scripts/generate_business_plan.py` → `output/智蜱卫士_商业计划书.docx` |

## 架构

### 训练流程
- `src/train/train_torchvision.py` - 主力训练脚本，torchvision Mask R-CNN + COCO 预训练权重，默认输出 `output/mask_rcnn_ticks.pth`
- `src/train/train_mask_rcnn.py` - MMDetection 训练入口（mmengine Runner + `configs/mask_rcnn_ticks.py`），历史备选
- detectron2 备选：`src/data/register_datasets.py`（注册 tick_train/val/test）+ `configs/train_mask_rcnn.yaml`，框架源码 vendored 于 `src/detectron2/`

### 推理流程
- `detect.py` - 单张图片推理并可视化（2 类模型）
- `detect_folder.py` - 批量文件夹推理（输入目录硬编码在脚本顶部）
- `src/inference/inference_demo.py` - 测试集批量推理
- `test_false_positives.py` - 在其他昆虫类别上测试误检率

### 评估与部署
- `src/eval/evaluate_model.py` - torchvision 模型 + pycocotools COCOeval，输出 mAP
- `src/deploy/export_onnx.py` - ONNX 导出（detectron2 TracingAdapter，历史）

### 检测模型
- 主力：torchvision Mask R-CNN + ResNet-50-FPN 主干网络
- 单类别模式：背景 + 蜱虫（2类）
- 多类别模式：规划中，代码未实现（见「多类别支持」）

## 数据结构

### 蜱虫数据集（COCO格式）
```
ticks image detection.v5i.coco/
├── train/_annotations.coco.json + images（1136 张）
├── valid/_annotations.coco.json + images（332 张）
└── test/_annotations.coco.json + images（134 张）
```

✅ 2026-09-01 已用 SAM 补全掩码：1649 条标注 segmentation 原为全空，现以 bbox 为提示由 SAM 生成真实多边形并写回；categories 清理为单条 `{"id":1,"name":"ticks"}`（原 id=0 重复项未被任何标注引用，已随清理移除）。原文件备份为各 split 目录下的 `_annotations.coco.bbox_only.json`。如需重新生成可运行 `scripts/sam_auto_label.py --mode fill`（断点续跑，已生成部分自动跳过）。

### 其他昆虫数据集

- `archive/` - 11类昆虫（8605张图片，仅文件夹分类标签，无边界框）
- `archive2/dataset/` - tick/mite 各 176 张（train 121 + test 55），已有部分标注：train/tick 95 张、test/tick 30 张的 X-AnyLabeling JSON + COCO 导出 `archive2/dataset/train/annotations/instances_default.json`（28 类，2026-04-27）
- `archive3/farm_insects/` - 15类农田昆虫（约1591张）

### 蜱虫叮咬伤口数据集（COCO格式）
```
tick-bite-improved.v7-random_split_resize_640.coco-segmentation/
├── train/_annotations.coco.json + images（828 张，1451 标注）
├── valid/_annotations.coco.json + images（102 张，164 标注）
└── test/_annotations.coco.json + images（102 张，191 标注）
```
共 1032 图 / 1806 条**多边形标注**（类别 `tick_bite_region` id=1；categories 另有一条未被引用的 id=0 冗余项，训练前需清理）。640×640，2026-08-31 Roboflow 导出。⚠️ 部分标注面积接近全图，训练前需统计面积分布人工复核。

### 蜱虫元数据（TickIDNet）
- `TickIDNet/Sample Images/`：3 物种 × 25 张 = 75 张样例图，文件名编码物种/性别/虫态/饱血（`Genus_species_sex_lifestage_source_alive_feedstage_#.jpg`）
- `TickIDNet/Data/All_data.csv`（12177 行）：Feed-Stage fed/unfed/unk、Sex f/m/unk、Lifestage a/n/l/unk、Species 5 种（Dermacentor variabilis、Ixodes scapularis、Amblyomma americanum 为主，另有 Amblyomma maculatum 3 行、Dermacentor andersoni 1 行）
- `TickIDNet/Confidence Thresholds/`：分物种置信阈值脚本与 3 个物种 CSV（TickIDNet 论文方法，毕设 WP2 复现引用）
- 全量图片已就位（2026-09-03）：`data/tick_species/`（173 物种 7623 张，文件夹即类别）。三个来源：README 的 3 个 GBIF DOI 归档（`data/tick_species_raw/`）→ `download_gbif_images.py`；GBIF API 任意物种（`species_list.txt`）→ `download_gbif_species.py`；HF fine-grained-challenges 的 Ixodidae 组（4499 张，CC0）→ `extract_fgc_ixodidae.py`。注意 static.inaturalist.org 在本机网络 403，图片走 inaturalist-open-data S3 桶；HF lance 分片需校验完整尺寸后再解包
- TickIDNet 官方模型（TF SavedModel）：`models/tickidnet_model/model/`（tick_detection 环境无 TensorFlow，需另建 TF 环境或转 ONNX 才能运行）

## 多类别检测标注

archive2 已有部分标注（见「数据结构」），其余昆虫数据需继续手动标注。

另：SAM 自动标注试点已产出 `output/sam_pilot/`（2026-09-01，7 类 × 30 张 = 210 图 / 652 掩码）：`annotations.json`（COCO）+ `review/<类>/`（图片副本 + X-AnyLabeling 多边形 JSON，**待人工复核**）+ `previews/`（掩码预览）。复核通过后可扩量：`--per-class 0` 表示全量，`--classes` 可填 28 类中的任意 slug。

使用 X-AnyLabeling + YOLO 模型进行自动标注加速：
1. 模型位置：`models/yolov8m.pt`
2. 配置：View → Auto Labeling → Add Model
3. 使用：按 L 键对当前图片自动标注（Ctrl+Shift+L 批量）
4. 导出：File → Export → COCO 格式

详细步骤见 `docs/annotation_guide.md`。

## 多类别支持

⚠️ 规划中，代码未实现：`train_torchvision.py` 的 `setup_multi_class_config()` 是空操作，无类别名映射，`--multi-class` / `--num-classes` 目前仅占位；当前只有单类别（2 类）可用。

| ID | 名称 | 说明 |
|----|------|------|
| 1 | tick | 主要检测目标 |
| 2 | mite | 与蜱虫形态相似 |
| 3 | ant | 负样本 |
| 4 | bee | 负样本 |
| 5 | beetle | 负样本 |
| 6 | spider | 负样本 |

（规划映射；28 类完整列表见 `models/classes.txt`）
