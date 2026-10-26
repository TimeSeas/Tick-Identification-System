# 共享存在性与蜱种分类：run01

登记日期：2026-10-26。状态：**训练停止，候选评估中**。完整完成13轮，第14轮1400batch后因非有限梯度停止，failure.json及日志保留；不是15轮成功。尚未通过系统验收，不修改默认模型。基础代码提交d9470672，随后预处理性能修正以config.json内实际文件SHA256为准。

## 输入与训练

- 输出根：`output/optimization_20261015/rebuild_20261024/`；正式检查点`recognition_run01/`，日志`logs/recognition_run01.log`。
- 输入`regions/recognition_split.json`，SHA256：`0a733b4d543e4e033c507e4ab56366ef780a0b187ffb8b945b0a99d84a52d255`。
- ImageNet通用初始化：`resnet50-0676ba61.pth`，SHA256：`0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`。不从旧蜱种模型初始化。
- 全部物种/非tick/COCO输入沿用跨任务group与SHA边界；COCO原坐标、其他图EXIF规范化。存在性和物种分头；未知物种正例不计算物种损失。
- 训练13213个视图：species整图2889、区域2570、非tick整图6001、困难代理区域490、insect tick整图73、COCO tick区域1190。
- 验证2343视图，校准2320视图。视图并非独立照片，不能直接拿其数量作为独立样本数。
- 448 letterbox、温和翻转/旋转/颜色增强；先缩到模型尺寸再增强，避免在多百万像素原图上旋转。ResNet50前两残差阶段冻结，BN保持预训练统计。
- AdamW，head lr=0.0003，encoder lr=0.00003，weight decay=0.0001；batch=8、accumulate=2、workers=2、AMP、seed42；最多15轮、patience5。
- 采样桶质量species=.45、未知species正例=.15、非tick=.40；各桶内逆平方根类频率。验证选分为binary balanced accuracy与species macro-F1均值，不用calibration选checkpoint。

```powershell
python src/train/train_recognition_rebuild.py --split output/optimization_20261015/rebuild_20261024/regions/recognition_split.json --out output/optimization_20261015/rebuild_20261024/recognition_run01 --epochs 15 --batch-size 8 --accumulate 2 --workers 2 --amp
```

恢复仅允许同配置、同总epochs、同文件SHA，从last_checkpoint.pth恢复；max-steps的partial检查点不允许正式部署。新实验使用新目录，原权重不覆盖。

2026-10-26第7轮中断，主/worker均退出，无failure.json或异常栈；前6轮有效。已使用同命令追加`--resume output/optimization_20261015/rebuild_20261024/recognition_run01/last_checkpoint.pth`恢复，日志单独保存到`logs/recognition_run01_resume01.log`。中断前后第7轮前350batch损失相同，正式epochs记录没有重复第1—6轮。该恢复不代表已完成15轮。

## 已发生的流程检查

检测pilot01因原图mask回贴OOM失败，pilot02因代理EXIF尺寸失败，证据保留。pilot03完成4训练步及全部500图验证。分类pilot完成4步及2343验证视图，用于排查梯度/内存/数据读取，不能计效果。

正式第1轮：存在性召回0.9738、代理FPR0.01753、species top1=0.3508、macro-F1=0.3294；第2轮分别0.9843、0.01336、0.6013、0.4104。这些是训练过程中的组件验证值，**不是最终模型、不是网页完整级联、不是独立测试结果**。完整逐类/视图结果见epochs.jsonl与val_epoch文件。

## 冻结候选

综合分最佳为第12轮（epoch字段11），存在性召回0.99214、代理FPR0.01753、species整图+ROI top1=0.70697、macro-F1=0.59515。它只作为待验收候选，不能直接视为优于旧模型。

- 训练best_checkpoint SHA256：`5adebb9315c0dea8c6b8a6274a10e46089e82e234df541c1bb4612b3a1437449`。
- 推理导出`recognition_run01/frozen_best.pth`，SHA256：`0c01ff4ad8a0eb6a2efa5f22e5c34acbf6575424e6d4d65c51a01c7d926bb065`，仅保存模型/类别/预处理/来源，不保存优化器，不覆盖训练证据。
- `comparison_val_run01/`及`comparison_calibration_run01/`比较完成。val新物种top1=.70697、macro-F1=.59515，旧头top1=.96078（旧模型见过部分来源，不是公平未见比较）；新物种头不部署。存在性recall旧.56245→新.99214、代理FPR旧.00417→新.01753，仅组件而非级联。
- 模型训练结束后修正训练器：允许AMP偶发梯度溢出时跳过该次更新并降低动态scale，记录跳过数，5次连续溢出失败；FP32非有限梯度仍失败。CPU真实GradScaler复现通过，不据此断言第14轮异常仅由AMP引起。后续实验须新目录，新源码哈希不得静默恢复本run。
- resume目录限制也已修复，仅允许检查点原实验目录。44项回归通过。训练采用的旧源SHA仍以原config和提交878f7830为准，不能把后续修正说成本次已训练内容。

## 完整级联比较与后续

以下都是开发/同源诊断；不把代理标签当人工真负。旧模型看过部分新分组来源。

| 实验目录 | 新定位召回 | 新计数MAE | 代理误报图（旧→新） | 结论 |
|---|---|---|---|---|
| cascade_val_run01（det=.3/presence=.5） | .91163 | .38830 | 79→73 /307 | 计数和误报失败 |
| cascade_val_run01_det05（det=.5/presence=.95） | .87442 | .19681 | 79→38 /307 | 开发val门槛通过 |
| cascade_calibration_run01（冻结同策略） | .95897 | .18085 | 77→34 /296 | 当时校准确认通过 |
| cascade_val_run01_guard（分来源门槛+区域复核） | .88372 | .20213 | 79→77 /307 | 误报门槛失败 |

`same_source_frozen_run01/`随后揭示precision候选对archive2近景蜱虫召回53/55→49/55，mite误报32/54→11/54。因此未切默认。109图已查看，后续不能再作为未见选参/验收。校准组件按每个正例来源98%经验召回得到presence=.007866711355745792，由12张insect_tick_full来源限定；对应区域复核仍未兼顾召回/误报。详见[后续计划](../plans/recognition_policy_followup_20261026.md)。

冻结63例已生成6页组件对照；39例定位已生成10页，包含失败。学习曲线与逐类错误CSV保存在对应产物目录。当前开始检测正式重训，原双头run01和失败信息完整保留。

## 证据边界

1. 完成训练，冻结best检查点与SHA；对比旧模型时明确旧权重见过部分新val来源，所以新分组val不能充当旧模型的独立泛化评估。
2. `evaluate_recognition_rebuild.py`做同输入组件诊断；`evaluate_recognition_cascade.py`评估数量/定位及代理误报，不能给无物种真值的框伪造物种准确率。
3. `regions/frozen_gallery.json`已在正式训练前固定63例，原图/区域质量预览`regions/quality_preview.png`已目视检查；最终展示成功与失败。
4. 只有通过级联及物种比较的组件才考虑替换。新存在性有效而新物种头退步时，可保留旧物种模型，不能把联合训练等同于必须全部部署。
5. 旧来源有遗漏、代理标签未逐图核验，同源109图不称为开放环境独立测试。服务请求耗时须另测，批量离线耗时不能代替网页时延。
