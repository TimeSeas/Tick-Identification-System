# SAM 标注代码修改与调参教程

> 面向本项目使用者 · 2026-09-01 · 目标：学会自己改参数、改代码，改善 SAM 自动标注效果
> 脚本：`scripts/sam_auto_label.py`（行号以当前版本为准，本教程行号与该文件一一对应）

## 1. 先认识脚本结构（10 秒导览）

| 行号 | 内容 | 作用 |
|------|------|------|
| 36–60 | `CLASS_MAPPING` | 文件夹名 → 类别 slug 映射（28 类） |
| 62 | `IMG_EXTS` | 接受的图片扩展名 |
| **64–71** | **`AMG_KWARGS`** | **auto 模式的核心调参区（最常改）** |
| 73 | `POLYGON_EPSILON` | 多边形简化精度 |
| 82–94 | `mask_to_polygons()` | 掩码→多边形，丢弃 <1% 面积的小轮廓 |
| 115–166 | `fill_masks()` | fill 模式（主数据集 bbox→掩码） |
| 169–282 | `run_auto()` | auto 模式（AMG 自动标注），**过滤规则在 212–220 行** |
| 285–324 | `preview_coco()` | preview 模式（抽查可视化） |
| 327–360 | `main()` | 命令行参数定义 |

改完任何代码后先运行 `python -m py_compile scripts/sam_auto_label.py` 检查语法，再重跑。**脚本支持断点续跑**：中途中断重跑会自动跳过已完成的图片/标注，改参数后放心重跑（已完成的不会重复计算，想全部重算见 5.3）。

## 2. 参数速查表（含义 + 往哪调）

### 2.1 AMG 核心超参（第 64–71 行 `AMG_KWARGS`）

| 参数 | 默认值 | 含义 | 调大 | 调小 |
|------|--------|------|------|------|
| `points_per_side` | 32 | 网格提示点密度（32×32=1024 点） | 漏检少、更密，但耗时≈平方增长（64→4 倍时间） | 更快、噪声候选少 |
| `pred_iou_thresh` | 0.88 | 模型自评掩码质量的置信阈值 | 更严，噪声少，但漏检多 | 召回高，噪声多 |
| `stability_score_thresh` | 0.90 | 掩码在阈值扰动下的稳定度 | 同上 | 同上 |
| `min_mask_region_area` | 64 | 掩码最小像素面积 | 只留大目标 | 保留小目标 |
| `crop_n_layers` | 1 | 多尺度裁剪层数（切块放大找小目标） | 小目标召回↑，耗时↑（每层约 +1.5×） | 更快 |
| `crop_n_points_downscale_factor` | 2 | 裁剪块内的网格缩放 | 与 crop 层数配合 | — |

### 2.2 auto 模式过滤规则（第 212–220 行，改逻辑在这里）

| 行 | 规则 | 默认 | 说明 |
|----|------|------|------|
| 213 | 面积过滤 `m["area"] < 0.001 * h * w` | 丢弃 <0.1% 图像的掩码 | 数值越小越宽松（如 0.0005） |
| 216 | 贴边过滤（掩码触到图像四边） | 丢弃 | 贴边多为背景；如果昆虫被裁在边缘可注释掉这行 |
| 219–220 | 每图保留面积 top-3 | `cands[:3]` | 改成 `[:5]` 保留更多候选；按"面积"排序可换成 `m["predicted_iou"]` 按置信度排序 |

### 2.3 其他可调项

| 行 | 参数 | 默认 | 说明 |
|----|------|------|------|
| 73 | `POLYGON_EPSILON` | 0.5 | 多边形简化容差（像素）；掩码边缘太碎就调到 1.0–2.0，追求精细则 0.3 |
| 89 | 小轮廓丢弃阈值 | 掩码面积 1% | 提高可减少碎片多边形 |
| 146 | fill 模式最小框 | 2px | 过小 bbox 的兜底，一般不用动 |
| 152 | fill 模式候选选择 | 3 个取最高分 | `scores.argmax()`；可改取中间分（`multimask_output=True` 的第 2 个通常更"折中"） |
| 336 | `--per-class` 默认 | 30 | 命令行可传 `--per-class 0` 跑全量 |

## 3. 常见问题 → 改哪里（速查）

| 症状 | 改法（行号） |
|------|--------------|
| **漏检多**（虫在但没掩码） | ① `pred_iou_thresh` 0.88→0.80、`stability_score_thresh` 0.90→0.85；② `points_per_side` 32→64；③ `crop_n_layers` 1→2；④ `min_mask_region_area` 64→32 |
| **噪声多**（背景被分割） | ① 两阈值各 +0.03；② 213 行面积阈值 0.001→0.002；③ top-3 改 top-1；④ 增加额外过滤（见 4.2 练习） |
| **小昆虫几乎全漏** | `crop_n_layers=2` + `points_per_side=64` + `min_mask_region_area=32` 组合 |
| **掩码边缘锯齿/太碎** | `POLYGON_EPSILON` 0.5→1.0 |
| **要新增一个类别** | 在 `CLASS_MAPPING`（36–60 行）加一行：`"新文件夹名": "新slug"`，slug 需与 `models/classes.txt` 一致 |
| **多虫图想每虫一个掩码** | AMG 无实例语义保证；正确做法是 5.1 的两阶段路线（检测框→SAM） |
| **跑太久** | `points_per_side` 32→24 或 16；先用 `--per-class 5` 试参数，满意再扩量 |

## 4. 动手练习（照做一遍就学会了）

### 4.1 改超参对比实验

1. 复制一份脚本备份：`cp scripts/sam_auto_label.py scripts/sam_auto_label.bak.py`（永远先备份）；
2. 把 65 行改成 `points_per_side=64`，66 行改成 `pred_iou_thresh=0.80`；
3. 用新 `--out-dir output/sam_pilot_test` 跑**同一个类 5 张**：
   ```bash
   python scripts/sam_auto_label.py --mode auto --input-dirs archive3/farm_insects \
       --classes Thrips --per-class 5 --seed 42 --out-dir output/sam_pilot_test
   ```
4. 打开 `output/sam_pilot_test/previews/` 对比默认参数下的 `output/sam_pilot/previews/thrips_*.png`，肉眼判断哪个更好；
5. 满意就保留参数，不满意继续调；对比完成删掉 `output/sam_pilot_test/` 即可。

### 4.2 加一条过滤规则（练改代码）

在 216 行贴边过滤之后、`cands.append(m)` 之前，加一条"置信度低于 0.9 的不要"：

```python
if m["predicted_iou"] < 0.9:
    continue
```

体会：所有 `m` 里可用的字段有 `segmentation`、`area`、`bbox`、`predicted_iou`、`stability_score`、`point_coords`，过滤规则就是用这些字段写 if。

### 4.3 换一个类别重跑

`--classes` 直接填其他 slug 或文件夹名（脚本两种都认），例如：

```bash
python scripts/sam_auto_label.py --mode auto --input-dirs archive3/farm_insects \
    --classes "Fall Armyworms" "Tomato Hornworms" --per-class 20
```

## 5. 重要注意事项

### 5.1 治本路线（质量问题的正解）

调 AMG 参数只能缓解问题——**"检测框 → SAM box 提示"才是正解**（主数据集 1649 条标注实证 0 失败）。工程顺序：先用 YOLOv8m（`models/yolov8m.pt`）或训练好的 Mask R-CNN 在目标图像上产出候选框，再仿照 `fill_masks()`（115–166 行）的 box 提示写法生成掩码。这部分如果要动手做，可以让 Claude 帮你在 sam_auto_label.py 里加一个 `--mode detect-then-segment`。

### 5.2 主数据集相关操作的红线

- 主数据集的掩码版 JSON 在同目录，原始 bbox-only 版本备份为 `_annotations.coco.bbox_only.json`，**不要删备份**；
- 若想对主数据集重新生成（如换参数）：先把 `_annotations.coco.bbox_only.json` 复制回 `_annotations.coco.json`，再跑 `--mode fill`（否则断点续跑会跳过已完成标注）；
- 想看任意 COCO JSON 的掩码效果：`--mode preview`（画红=掩码、绿=新框、黄=原框对比）。

### 5.3 想完全重算（清掉断点）

- auto：删除 `output/sam_pilot/annotations.json`（断点记录在里面）；
- fill：按 5.2 恢复备份 JSON。

### 5.4 环境与运行习惯

- 跑长任务前：**插电源或把睡眠设为"从不"**——本机电池供电 20 分钟无操作会休眠，休眠会杀掉任务并导致 GPU 封锁（需重启恢复）；
- 固定 `--seed` 保证结果可复现（同 seed 同抽样）；
- 每次只改 1–2 个参数再对比，别一次改一堆。
