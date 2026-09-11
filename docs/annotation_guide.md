# 昆虫边界框标注指南

## 概述

为扩展蜱虫检测模型为多类别检测（蜱虫/螨虫/其他昆虫），需要对以下数据进行边界框标注：

| 数据集 | 类别 | 数量 | 用途 |
|--------|------|------|------|
| archive2/dataset/train/tick | 蜱虫 | 121张（测试集另有55张） | 正样本 |
| archive2/dataset/train/mite | 螨虫 | 121张（测试集另有55张） | 易混淆负样本 |
| archive/Ant | 蚂蚁 | ~30张 | 负样本 |
| archive/Bee | 蜜蜂 | ~30张 | 负样本 |
| archive/Beetle | 甲虫 | ~30张 | 负样本 |

> 注：archive2 已有部分标注产物——train/tick 95 张、test/tick 30 张的 X-AnyLabeling JSON，以及 COCO 导出 `archive2/dataset/train/annotations/instances_default.json`（28 类，2026-04-27 导出），可在此基础上继续标注。

## 推荐工具

### 1. Labelme（推荐，简单易用）

**安装：**
```bash
pip install labelme
```

**使用方法：**
```bash
# 启动标注工具
labelme

# 命令行标注（适合批量处理）
labelme archive2/dataset/train/tick --output output --exist-ok
```

**标注步骤：**
1. 打开 Labelme 界面
2. 点击 "Open Dir" 选择图片文件夹
3. 点击 "Create RectBox" 绘制边界框
4. 输入类别标签（如 "tick", "mite", "ant"）
5. 按 "Next Image" 保存并跳到下一张
6. 标注完成后，点击 "Save" 保存 JSON 文件

**输出格式：** 每张图片生成 `.json` 文件，包含多边形点

### 2. CVAT（适合团队协作）

**在线使用：** https://cvat.org

**步骤：**
1. 注册并创建新项目
2. 添加标签：tick, mite, ant, bee, beetle
3. 上传图片或选择数据源
4. 使用矩形框工具标注
5. 导出为 COCO 格式

### 3. Label Studio（适合复杂项目）

```bash
pip install label-studio
label-studio start
```

### 4. X-AnyLabeling（推荐 Windows 用户，功能强大）

**官网：** https://github.com/CVHub520/X-AnyLabeling

---

#### 第一步：安装

**方法1：下载 exe 便携版（推荐，下载即用）**
1. 打开 https://github.com/CVHub520/X-AnyLabeling/releases
2. 下载最新版本的 `.exe` 文件（如 `anylabeling-v2.x.x.exe`）
3. 双击运行，无需安装

**方法2：pip 安装**
```bash
pip install anylabeling
# 运行
anylabeling
```

**方法3：本机已有环境启动（Git Bash）**

本机已安装独立 conda 环境 `x-anylabeling`（位于用户目录，不在本项目内），启动命令：

```bash
用anaconda环境
conda activate x-anylabeling
cd C:\Users\a\X-AnyLabeling
python anylabeling/app.py
# source ~/.bash_profile && conda activate x-anylabeling && cd /c/Users/a/X-AnyLabeling && python anylabeling/app.py
```

---

#### 第二步：认识界面

启动后界面分为三个区域：

```
┌─────────────────────────────────────────────────────────┐
│  菜单栏 (File, Edit, View, Help)                        │
├──────┬──────────────────────────────────┬──────────────┤
│      │                                  │              │
│ 工具栏 │         图片预览区               │   标签面板    │
│      │                                  │              │
│ [R]  │    （当前打开的图片）              │  [+] 添加    │
│ [M]  │                                  │  tick       │
│ [P]  │                                  │  mite       │
│      │                                  │  ant        │
│      │                                  │  bee        │
├──────┴──────────────────────────────────┴──────────────┤
│  状态栏：当前图片 1/180  |  已标注: 5 个对象             │
└─────────────────────────────────────────────────────────┘
```

**工具栏说明：**
| 图标 | 功能 | 快捷键 |
|------|------|---------|
| → | 矩形框（最常用） | R |
| ◇ | 多边形 | M |
| ● | 点 | P |
| ✋ | 拖动/平移 | - |
| 🔍 | 缩放 | - |
| 🗑️ | 删除选中 | Q |

---

#### 第三步：创建标签

1. 在右侧 **标签面板** 点击 **"Edit Labels"**（或右键空白区域）

2. 点击 **"+"** 添加新标签

3. 输入标签信息：
   ```
   名称(Name): tick
   显示名称(Display Name): tick - 蜱虫
   颜色(Color): #FF0000 (红色，可选)
   ```

4. 重复添加所有需要的标签（共28个类别）：

   **archive 原始类别（11类）+ archive2 的 tick/mite（2类）= 13类：**
   ```
   tick        - 蜱虫
   mite        - 螨虫
   ant         - 蚂蚁
   bee         - 蜜蜂
   beetle      - 甲虫
   butterfly   - 蝴蝶
   dragonfly   - 蜻蜓
   fly         - 苍蝇
   grasshopper - 蚱蜢
   ladybug     - 瓢虫
   mosquito    - 蚊子
   spider      - 蜘蛛
   wasp        - 黄蜂
   ```

   **archive3 农田昆虫类别（15类）：**
   ```
   africanized_honey_bee    - 非洲化蜜蜂（杀人蜂）
   aphid                    - 蚜虫
   armyworm                 - 粘虫
   brown_marmorated_stink_bug - 棕纹蝽
   cabbage_looper            - 菜青虫
   citrus_canker            - 柑橘溃疡病（病害，非昆虫）
   colorado_potato_beetle   - 科罗拉多甲虫
   corn_borer               - 玉米螟
   corn_earworm             - 棉铃虫
   fall_armyworm            - 秋粘虫
   fruit_fly                - 果蝇
   spider_mite              - 蜘蛛螨
   thrips                  - 蓟马
   tomato_hornworm          - 番茄天蛾
   western_corn_rootworm    - 西方玉米根虫
   ```

5. 点击 **"Save"** 保存

---

#### 第四步：打开图片文件夹

1. 点击菜单 **File** → **Open Directory**
2. 选择要标注的文件夹，例如：
   - `C:\MyProject\project8_bug\archive2\dataset\train\tick`
3. 左侧缩略图列表会显示所有图片

---

#### 第五步：手动标注（基础方法）

1. **选择工具**：点击左侧 **矩形框工具** 或按 `R`

2. **绘制边界框**：
   - 在图片上按住鼠标左键拖动
   - 松开鼠标，形成一个矩形框

3. **选择标签**：
   - 绘制完成后，右侧标签面板会高亮
   - 点击对应的标签（如 "tick"）

4. **调整边界框**（可选）：
   - 拖动框的边角可以调整大小
   - 拖动框内部可以移动位置

5. **标注下一张**：
   - 按 `D` 跳到下一张图片
   - 按 `A` 回到上一张

6. **删除标注**：
   - 选中标注框，按 `Q` 或点击工具栏的删除按钮

7. **保存**：
   - 按 `Ctrl+S` 保存当前图片
   - 或继续标注，系统会自动保存

---

#### 第六步：自动标注（AI 辅助）⭐

X-AnyLabeling 2.3.6 支持 **YOLO** 等模型自动检测物体，只需人工修正，大大加速标注！

**模型文件位置：** `C:\MyProject\project8_bug\models\yolov8m.pt`

---

**配置自动标注模型（2.3.6版本）：**

1. **打开自动标注设置**
   - 按 `L` 键 打开自动标注面板
   - 或点击菜单 **View** → **Auto Labeling**

2. **添加模型**
   - 在右侧面板点击 **"Add Model"** 或 **"+"** 按钮
   - 选择模型文件：`C:\MyProject\project8_bug\models\yolov8m.pt`
   - 如果遇到格式问题，尝试使用 ONNX 格式模型

3. **等待模型加载**
   - 首次加载需要下载模型权重（约50MB）
   - 状态栏显示 "Model loaded successfully" 表示加载成功

---

**使用自动标注：**

1. **打开一张图片**

2. **执行自动标注**
   - 按 `L` 键 对当前图片进行自动标注
   - 或点击工具栏的 **"Auto"** 按钮

3. **查看结果**
   - AI 自动绘制的边界框显示在图片上
   - 右侧标签面板显示检测到的类别和置信度

---

**修正自动结果（必须）：**

⚠️ **YOLOv8m 是通用模型，会把昆虫误识别为其他类别！必须修正！**

1. **检查每个边界框**
   - 点击边界框查看当前标签
   - 如果标签错误，点击右侧正确的标签（如 "tick"）进行替换

2. **删除错误框**
   - 选中错误框，按 `Q` 删除

3. **添加漏标框**
   - 用矩形框工具手动添加

4. **修正示例**
   ```
   AI 可能把蜱虫识别为: person, bird, cat, dog 等（错误）
   正确做法: 点击边界框 → 点击右侧 "tick" 标签
   ```

---

**批量自动标注（可选）：**

1. 按 `Ctrl+Shift+L` 对整个文件夹所有图片批量自动标注

2. 标注完成后，**逐张检查并修正**每一张图片的标签

---

**模型说明：**

| 模型 | 大小 | 说明 |
|------|------|------|
| yolov8m.pt | ~50MB | 通用目标检测，可能误识别昆虫 |
| yolo11m.pt | ~50MB | 较新版本，精度相当 |

如果自动标注效果不佳，建议直接手动标注 50-100 张，然后用这些数据训练专门的昆虫检测模型。

**推荐工作流程：**
```
1. 先手动标注 10-20 张图片（建立基础）
2. 使用这些图片训练一个初步模型
3. 用初步模型对剩余图片进行自动标注
4. 人工修正自动标注结果
5. 用修正后的数据重新训练模型
```

---

#### 第七步：导出 COCO 格式

1. 点击菜单 **File** → **Export**

2. 选择导出格式：**COCO**

3. 选择输出文件夹

4. 确认导出

5. 导出完成后，检查输出文件夹：
   ```
   输出文件夹/
   ├── annotations.json      ← 标注数据（COCO格式）
   ├── image1.jpg            ← 图片文件（如果选择复制）
   ├── image2.jpg
   └── ...
   ```

6. `annotations.json` 文件结构：
   ```json
   {
     "images": [...],
     "annotations": [...],
     "categories": [
       {"id": 1, "name": "tick"},
       {"id": 2, "name": "mite"},
       {"id": 3, "name": "ant"}
     ]
   }
   ```

---

#### 完整操作流程示例

以标注 archive2 的 tick 图片为例：

```
1. 启动 X-AnyLabeling

2. 创建标签：tick, mite, ant, bee, beetle, spider

3. File → Open Directory → 选择 archive2/dataset/train/tick

4. 选中第一张图片

5. 按 R 选择矩形框工具

6. 在蜱虫位置拖动绘制边界框

7. 在右侧点击 "tick" 标签

8. 按 D 跳到下一张

9. 重复 6-8，标注所有 tick 图片

10. 如果有 AI 模型，按 L 进行自动标注加速

11. 标注完成后，File → Export → COCO → 选择输出文件夹

12. 得到 annotations.json 用于训练
```

---

#### 常见问题

**Q: 标签面板没有显示？**
A: 点击菜单 View → Show Label List

**Q: 快捷键不生效？**
A: 确保输入法在英文状态

**Q: 导出的 JSON 为空？**
A: 确保每张图片至少有一个标注，并保存后再导出

**Q: 如何同时标注多个类别？**
A: 在右侧标签列表切换当前标签，再绘制新的边界框

## COCO 格式转换

标注完成后，使用以下脚本将 JSON 标注转换为 COCO 格式：

```python
# 参考 scripts/convert_folder_to_coco.py
```

## 标注规范

### 边界框绘制规则
- 紧贴目标边界，不留多余空白
- 覆盖完整目标，不能截断
- 多个目标需分别标注

### 标签命名
| 类别 | 标签 | 说明 |
|------|------|------|
| 蜱虫 | tick | 目标检测对象 |
| 螨虫 | mite | 与蜱虫形态相似 |
| 蚂蚁 | ant | 负样本 |
| 蜜蜂 | bee | 负样本 |
| 甲虫 | beetle | 负样本 |

### 质量要求
- 边界框位置误差 < 5像素
- 每张图至少有一个标注
- 标注覆盖率 > 95%

## 批量处理

对于 archive2 的 352 张 tick/mite 图片（train 121+121、test 55+55），建议：
1. 先标注 100 张 tick + 100 张 mite（训练集）
2. 剩余图片作为验证集

## 输出要求

最终输出的 COCO 格式 JSON 文件结构：
```json
{
  "images": [...],
  "annotations": [...],
  "categories": [
    {"id": 1, "name": "tick"},
    {"id": 2, "name": "mite"},
    {"id": 3, "name": "ant"}
  ]
}
```

## 下一步

完成标注后：

1. **X-AnyLabeling 用户**：直接通过 File → Export 导出 COCO 格式，无需额外转换

2. **其他工具用户**：使用转换脚本：
```bash
python scripts/convert_folder_to_coco.py --input 标注输出目录 --output annotations.json
```

3. **开始训练**：注意多类别检测尚未在训练脚本中实现（`--multi-class`/`--num-classes` 目前仅占位），标注完成后先按单类别流程接入训练：

```bash
python src/train/train_torchvision.py \
    --train-ann annotations_train.json \
    --train-img ./images \
    --val-ann annotations_val.json \
    --val-img ./images
```