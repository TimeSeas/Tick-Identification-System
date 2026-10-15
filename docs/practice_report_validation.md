# 基于Mask R-CNN的蜱虫识别系统的设计与实现

## 实践报告与模型核验说明

2026-10-15，题名由用户指定。当前交付聚焦蜱虫候选实例数量、逐只定位、28类蜱种识别、昆虫粗分类兜底与 Web 历史档案。伤口分割、部件分割、小程序、桌面与边缘端列为展望。旧风险规则仍在兼容接口中，前端不再将其作为主要功能。

用户本轮明确授权 Codex 完成前端设计和截图，并修复分类输入尺度、Docker模型配置、检测评估尺度三处具体问题；其他功能开发仍遵守 AGENTS.md 中自主完成代码的原则。

## 使用界面

1. 在识别工作台选择或拖入 JPG、PNG、BMP、WebP 图片，最大10 MB。
2. 点击开始识别，处理期间禁止重复提交；结果显示候选实例数、确认与待确认数量。
3. 切换原图和标注图，查看逐只结果卡片，分别读取检测置信度、分类置信度及坐标。
4. 在识别档案中分页查询、打开详情和下载当时的标注图。历史结果不会因更换模型而自动重算。

所有展示数值来自现有接口；前端未添加虚构统计、示例预测或模拟接口。中文名沿用项目对照表，不代表已完成外部分类学专家审定。

## 现役配置

| 环境变量 | 本地默认 | Docker容器内值 |
|---|---|---|
| DETECTOR_PATH | ../../output/det_v3/mask_rcnn_ticks_v3.pth | /app/models/det_v3/mask_rcnn_ticks_v3.pth |
| SPECIES_MODEL_PATH | ../../output/tick_species_clean_448/model/tick_species_best.pth | /app/models/tick_species_clean_448/model/tick_species_best.pth |
| SPECIES_SPLIT_JSON | ../../data/tick_species/split_clean.json | /app/data/split_clean.json |
| THRESHOLDS_PATH | ../../output/tick_species_clean_448/thresholds.csv | /app/models/tick_species_clean_448/thresholds.csv |
| SPECIES_INPUT_SIZE | 448 | 448 |
| INSECT_MODEL_PATH | ../../output/insect_classifier/model/insect_best.pth | /app/models/insect_classifier/model/insect_best.pth |
| INSECT_SPLIT_JSON | ../../data/insects_split.json | /app/insects_split.json |

`main.py`向蜱种分类器传入输入尺寸。检测评估脚本默认使用v3权重，并恢复 `min_size=(1024,)`、`max_size=1200`。粗分类仍为224输入。Compose新增昆虫划分JSON单文件挂载。切换模型时必须一并核对类别顺序、阈值、输入尺寸，不能只替换权重路径。

## 2026-10-15 完整测试集重测

使用现有权重与现有划分，没有重训；设备为 RTX 4060 Laptop GPU，PyTorch 2.5.0+cu118、torchvision 0.20.0+cu118。结果适用于本次离线组件评估。

| 模块 | 测试规模 | 本轮结果 |
|---|---|---|
| Mask R-CNN v3 | 134图、229框 | bbox AP@[.50:.95] 0.6164、AP50 0.8659、AP75 0.7085 |
| 蜱种分类224 | 589图、28类 | accuracy 67.23%、macro-F1 0.6589 |
| 蜱种分类448 | 589图、28类 | accuracy 75.55%、macro-F1 0.7419 |
| 昆虫粗分类224 | 1043图、28类 | accuracy 94.25%、macro-F1 0.8373 |

蜱种224与448使用同一测试划分，可比较分辨率配置；本轮macro-F1差为0.0830。现有预测CSV复核另为验证集数据，不能替换测试集结果。

完整本地材料位于 `毕业设计/实践报告工作区/experiments/`：`raw_evaluation.log`、`results.json`、三个分类模型的逐图预测CSV与混淆矩阵JSON、检测逐框预测JSON。汇总含模型和数据划分SHA256，便于确认实验对象。该目录不包含在代码推送中。

## 验证边界与后续实验

- AP50不是逐图计数准确率；分类准确率也不是完整级联系统准确率。尚需固定测试集上的逐图计数误差、召回与端到端蜱种结果统计。
- split_clean三组文件路径无交集，但本轮未完成感知去重、同一观察事件或同一实体分组划分，不能据此宣称所有数据泄漏风险已消除。
- 分物种阈值的MC<2%是验证集搜索目标，不是线上错误率保证；应报告测试集覆盖率及保留结果错误率。现有脚本按真类分组搜索、推理按预测类查询，解释时需注明。
- 历史数据库包含误检样例，不能用小样本回归结果推断总体零误检。固定负样本集和误检防御消融仍需补充。
- 本轮本机真实模型启动、网页交互及离线评估可复核；未执行Docker镜像构建、容器端到端测试、并发压测与安全审计。
- 前端构建仍有Element Plus相关大包提示，不影响构建成功；按需导入属于后续加载优化。

## 本地复现

在项目根目录、已激活 tick_detection 的终端执行：

```powershell
python src/eval/evaluate_model.py
python "毕业设计/实践报告工作区/rerun_models.py"
```

第二条为本地报告工作区核验工具，保存完整原始证据，不覆盖模型权重。若该本地目录未随交接提供，可按 `scripts/eval_species_model.py` 的 `--model`、`--classes-split`、`--test-split`、`--input-size` 参数分别复核分类模型；切勿混用旧31类模型与新划分。
