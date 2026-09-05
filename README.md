# 蜱虫识别与细粒度属性检测系统

使用 Mask R-CNN 检测蜱虫并分类细粒度属性（饱血/饥饿状态）。阶段一已完成蜱虫检测，测试集 mAP@0.50 = **0.842**。

- 主力模型：torchvision Mask R-CNN + ResNet-50-FPN（COCO 预训练）
- 后续路线：阶段二 fed/unfed 属性分类（利用 TickIDNet 元数据）→ 阶段三 两阶段昆虫检测 + 蜱虫属性识别
- AI 开发约定见 [CLAUDE.md](CLAUDE.md)，标注流程见 [docs/annotation_guide.md](docs/annotation_guide.md)

## 环境安装

```bash
conda env create -f environment.yml
conda activate tick_detection
```

- 本机已安装好 `tick_detection` 环境，日常直接 `conda activate tick_detection` 即可
- `setup_env_anaconda.ps1` 创建同名环境，但 torch CUDA 版本（12.1）与 detectron2 安装方式（git 源码）与 `environment.yml`（cu118 + 预编译 wheel）不一致，仅作备选
- `setup_env.bat` / `setup_env.ps1` 创建的是 venv `tick_env`（非 conda），为历史脚本
- ⚠️ `requirements.txt` 是历史 pip freeze 快照，**不含 torch/torchvision/detectron2 等核心依赖，不能单独用于建环境**

## 常用命令

```bash
# 训练（单类别：背景 + 蜱虫）
python src/train/train_torchvision.py

# 单张图片推理
python detect.py <图片路径> -o output.jpg -c 0.5

# 文件夹批量推理（输入目录硬编码在脚本顶部，需编辑脚本修改）
python detect_folder.py

# 测试集批量推理
python src/inference/inference_demo.py

# 模型评估（mAP）
python src/eval/evaluate_model.py

# 其他昆虫类别误检率测试
python test_false_positives.py

# 数据集格式校验
python scripts/verify_coco_dataset.py

# 数据诊断：打印训练集首条标注的 segmentation 字段
python src/train/检查数据集有无多边形.py

# SAM 掩码补全 / 自动标注（支持断点续跑，详见 --help）
python scripts/sam_auto_label.py --mode fill --coco "ticks image detection.v5i.coco/train/_annotations.coco.json" ...
python scripts/sam_auto_label.py --mode auto --input-dirs archive2/dataset/train archive3/farm_insects --classes tick mite ... --per-class 30
```

## 目录结构

| 路径 | 说明 |
|------|------|
| `ticks image detection.v5i.coco/` | 主数据集（COCO 格式，train/valid/test） |
| `src/train/` | 训练脚本（torchvision 主力 + MMDetection 备选） |
| `src/eval/` | 模型评估（mAP） |
| `src/inference/`、`src/deploy/`、`src/demo/` | 批量推理、ONNX 导出、detectron2 演示 |
| `src/detectron2/` | vendored detectron2 源码（历史备选框架） |
| `detect.py`、`detect_folder.py` | 单张/批量推理入口 |
| `test_false_positives.py`、`test_model.py` | 误检率测试、模型冒烟测试 |
| `scripts/` | 数据校验/转换、SAM 标注与掩码补全（`sam_auto_label.py`）、商业计划书生成 |
| `models/` | 标注辅助模型（yolov8m.pt / onnx、sam_vit_b_01ec64.pth）及配置 |
| `output/` | 模型权重 `mask_rcnn_ticks.pth`、推理结果、商业计划书 |
| `TickIDNet/` | 上游论文仓库（元数据 `Data/All_data.csv`） |
| `archive/` `archive2/` `archive3/` | 其他昆虫数据集（详见下节） |

## 数据集

### 主数据集（有标注）

`ticks image detection.v5i.coco/`：train 1136 / valid 332 / test 134 张，COCO 标注。

### 其他昆虫数据集

| 数据集 | 内容 | 标注状态 |
|--------|------|----------|
| `archive/` | 11 类昆虫，8605 张 | 仅文件夹分类标签，无边界框 |
| `archive2/` | tick/mite 各 176 张（train 121 + test 55） | ⚠️ 已有部分标注：train/tick 95 张、test/tick 30 张的 X-AnyLabeling JSON + COCO 导出 `archive2/dataset/train/annotations/instances_default.json`（28 类） |
| `archive3/farm_insects/` | 15 类农田昆虫，约 1591 张 | 无边界框 |

### 蜱虫元数据（TickIDNet）

`TickIDNet/Data/All_data.csv`（12177 行）：Feed-Stage fed/unfed/unk、Sex f/m/unk、Lifestage a/n/l/unk、Species 5 种（3 主要 + 2 零星）。

## 多类别标注

多类别检测**尚未在训练脚本中实现**（`--multi-class`/`--num-classes` 参数目前仅占位）。标注工作进展见上表，标注流程见 [docs/annotation_guide.md](docs/annotation_guide.md)。

SAM 自动标注试点产物在 `output/sam_pilot/`（7 类 × 30 张，2026-09-01）：`annotations.json`（COCO）+ `review/<类>/`（X-AnyLabeling 多边形 JSON，**待人工复核**）+ `previews/`（掩码预览）。

## 已知问题

- ✅ 已解决（2026-09-01）：主数据集 1649 条标注原无多边形 segmentation，已用 SAM（bbox 提示）补全真实掩码并写回；原 `categories` 双 "ticks"（id 0/1）已清理为单条 id=1。原始文件备份在同目录 `_annotations.coco.bbox_only.json`。
