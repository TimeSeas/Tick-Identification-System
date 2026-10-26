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

## 双头识别训练

先运行区域准备：

```powershell
python scripts/prepare_recognition_regions.py --data output/optimization_20261015/rebuild_20261024/data --out output/optimization_20261015/rebuild_20261024/regions
```

候选区域是现役v3的辅助定位，继承来源物种标签而非生成标签。COCO区域使用原始坐标与已有人工作物种无关的tick标签；无标注空图不作为非tick。代理负例保留来源类别，val/calibration的固定区域仅用来匹配部署输入，不参与训练。缓存每20图持久化，同配置可续跑。

区域清单完整且目视检查后，先运行新目录中的partial试跑：

```powershell
python src/train/train_recognition_rebuild.py --split output/optimization_20261015/rebuild_20261024/regions/recognition_split.json --out output/optimization_20261015/rebuild_20261024/recognition_pilot --epochs 1 --batch-size 8 --accumulate 2 --amp --max-steps 4
```

正式训练去掉max-steps并使用新目录、epochs=15。骨干从本地ImageNet初始化，前两残差阶段冻结，小批BN保持预训练统计；存在性/物种分别监督，COCO未知物种不计算species损失。验证逐sample_type报告，best按二分类balanced accuracy与species macro-F1均值选择，calibration内容不读取。正式检查点只允许相同配置和相同总epochs恢复；partial不能续成正式训练。代码/清单哈希变化须新建实验。

run01实际命令使用workers=2，详见[实验登记](../experiments/recognition_run01.md)。完成轮次可用`plot_recognition_training.py --run <目录>`画学习曲线，用`analyze_recognition_errors.py --split <清单> --predictions <val_epoch_XXX.json> --out <新分析目录>`统计弱类和错误；这些不读测试集。

训练结束后，先用`evaluate_recognition_rebuild.py --checkpoint <best> --split <区域清单> --subset val --out <新目录>`比较组件，再用同命令的calibration输出和`calibrate_recognition_rebuild.py`生成阈值。阈值要同时报告接受比例、正确接受数和样本不确定性；拒绝更多不等于top1提高。级联数量/定位使用`evaluate_recognition_cascade.py --checkpoint <best> --split <detector_split.json> --subset val --out <新目录>`；这项模型策略比较不含上传像素上限和网页延迟测试。

## 回归与模型切换

```powershell
python -m unittest discover -s tests -v
```

训练启动不等于模型可用。先检查日志与完整性，再按计划比较旧模型、真实计数、召回及代理误报；未达到门槛不修改默认配置。模型切换时同步类别、尺度、预处理、阈值和权重哈希，并保存可回滚配置。原权重和历史实验不覆盖。

## 2026-10-26已完成的分支版本

当前默认是微调第2轮加.3检测门槛，全部训练/校准/来源代价见[微调](../experiments/detector_v3_finetune_run01.md)和[校准核验](../experiments/detector_calibration_run01.md)，运行与回滚见deployment.md。上文双头训练为实验方法，不代表已采用该模型。新一轮实验不得覆盖既有目录，不能根据文件名best或旧mAP直接替换部署。
