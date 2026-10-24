# 重建训练运行说明

适用2026-10-24重建；当前状态先看CONTEXT。所有命令在仓库根、`tick_detection`环境下执行，一次只占用一项GPU任务。

## 冻结数据

`output/optimization_20261015/rebuild_20261024/data/`包含species、insect、detector的train/val/calibration清单及summary。每行带group/SHA/来源，detector另带原COCO标注；非tick文件夹代理明确带negative_label。旧空标注不作为人工审核真负。不要手工把验收或校准项拷入train。

## 检测训练

先运行有partial标记的4步试跑，完整验证新val：

```powershell
python src/train/train_detector_rebuild.py --split output/optimization_20261015/rebuild_20261024/data/detector_split.json --out output/optimization_20261015/rebuild_20261024/detector_pilot --epochs 1 --batch-size 1 --accumulate 2 --amp --max-steps 4
```

输出目录必须为空或不存在。不要重复运行到同目录覆盖证据；首次日志保存在`logs/detector_pilot.log`。partial权重明确命名，不能静默恢复成完整训练。

检查训练链路后，正式实验使用新的输出目录，去掉max-steps。epoch数依试跑耗时与验证趋势确定，不预先承诺训练效果：

```powershell
python src/train/train_detector_rebuild.py --split output/optimization_20261015/rebuild_20261024/data/detector_split.json --out output/optimization_20261015/rebuild_20261024/detector_v4_run01 --epochs 10 --batch-size 1 --accumulate 2 --amp
```

正式完整轮检查点可加`--resume <last_checkpoint.pth>`并增大epochs续训。训练从本地COCO通用预训练加载，不从现役v3初始化；正例分割标注缺失会快失败，不伪造bbox矩形mask。逐轮保存配置、损失、完整val结果、代理误报、best/last和失败信息。检查点选择只用val；calibration不参与训练或checkpoint选择。

## 测试与模型切换

```powershell
python -m unittest discover -s tests -v
```

训练启动不等于模型可用。先检查日志与完整性，再按计划比较旧模型、真实计数、召回及代理误报；未达到门槛不修改默认配置。模型切换时同步类别、尺度、预处理、阈值和权重哈希，并保存可回滚配置。原权重和历史实验不覆盖。
