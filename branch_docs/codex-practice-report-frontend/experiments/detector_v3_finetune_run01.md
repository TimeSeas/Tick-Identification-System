# 现役v3困难负例微调 run01

登记2026-10-26，完整3轮已完成，原GPU会话86243已退出。动机：通用初始化3轮虽减少误报但损失定位召回，尝试保留现役能力再学习困难代理负例。不是全新未见来源训练，val只作开发比较。后续.3校准、分支默认与HTTP验证见[核验登记](detector_calibration_run01.md)。

- 冻结清单同detector_run02：`data/detector_supervision_split_v2.json`，SHA256 `aee2baf8b3c0b7e9b1fd110501e346f62b638cd7f7d5f30e5185690250e13013`，2937/495/484。
- 显式从`output/det_v3/mask_rcnn_ticks_v3.pth`初始化，SHA256 `4e14eeb2fae71667d5cd40e901f0a99d1ad68a6a0d212b31f91bc7cdd06929ae`。旧模型见过部分新val来源，不能描述为独立测试或重新洗白旧来源。
- 通用COCO缓存仍用于构建一致架构，随后严格加载全部v3参数。1024/1200、冻结低两阶段、SGD lr=.0003、momentum=.9、wd=.0005，StepLR5/.1、batch1/accumulate2、workers2、AMP、seed42，先完整3轮。
- 选模仍为val F1-.25×代理误报率；另外必须检查召回不降超过2pp、计数和完整级联。代理负例并非人工真负。保留每轮预测、best/last、失败及源码/模型哈希，不覆盖原文件。
- 输出 `output/optimization_20261015/rebuild_20261024/detector_v3_finetune_run01/`；日志 `logs/detector_v3_finetune_run01.log`。原通用初始化训练源码快照保存在detector_run02/source_snapshot/，不得以新源码静默恢复该run。

```powershell
python src/train/train_detector_rebuild.py --split output/optimization_20261015/rebuild_20261024/data/detector_supervision_split_v2.json --out output/optimization_20261015/rebuild_20261024/detector_v3_finetune_run01 --epochs 3 --batch-size 1 --accumulate 2 --workers 2 --amp --lr .0003 --warm-start output/det_v3/mask_rcnn_ticks_v3.pth --expected-warm-start-sha256 4e14eeb2fae71667d5cd40e901f0a99d1ad68a6a0d212b31f91bc7cdd06929ae
```

冻结训练器SHA256：a5f44e56d1dbafca2086b0628a0cab82b80477d0411f68e7a2b8cd5070c61a43。其余源码及输入身份由本run config.json记录；源码更改后不能静默恢复该run，须新实验。

## 逐轮记录

| 完整轮次 | 定位recall | precision | 计数MAE | 代理误报图/307 |
|---|---|---|---|---|
| 1 | .85116 | .93367 | .14362 | 57 |
| 2 | .86977 | .93035 | .15957 | 50 |

旧v3全图为.88372/.84444/.20213/252；第2轮初步符合召回降幅≤2pp、计数不退步及代理误报相对降30%的检测门槛，但不是完整级联通过。最终检查全部轮次，先排除不满足硬门槛的检查点，再按已登记val选分选择，不能盲用文件名best。export_detector_checkpoint.py冻结完整检查点并保存来源/配置/权重SHA，输出后再运行实际服务口径级联。

非蜱虫代理另统计正确兜底候选和正确确认覆盖，区分待确认与确认，不把“框少了”自动当昆虫分类变好。图片只有文件夹来源标签，不是独立昆虫学鉴定。


## 三轮完整结束与冻结

第3轮recall=.82791、precision=.92228、MAE=.17021、代理误报24/307，召回退步，不能为了更低误报选择它。最终选第2轮，既满足初步硬门槛，也是原选分最高。完整3轮均无AMP跳过。checkpoint_epoch_002导出frozen_detector.pth，SHA256 9e7a3a9b9cb32df87a37be71d85068f98a28d17bb61f828354534d4e95ed8b7c；导出frozen_detector.json保存原检查点SHA、配置及指标。原v3不覆盖。

实际服务口径完整495 val级联已完成。原规则drop_in通过开发门槛；regional_only误报56、detector_only误报74，均未达相对降30%要求，不选。输出cascade_detector_v3_finetune_run01/。冻结configs/inference_detector_finetune.json，保留原分类模型及校准，仅新检测器，尚未设为默认。

| 实际级联开发指标 | 原系统 | 新drop_in |
|---|---|---|
| 定位recall | .87907 | .86512 |
| precision | .84375 | .92079 |
| 计数MAE（188有标注图） | .20745 | .15426 |
| 非蜱虫代理误报（307图） | 79 | 22 |
| 正确昆虫兜底确认（307图） | 228 | 282 |

全39冻结案例已输出10页drop_in_gallery并全部查看：蝴蝶、蜻蜓、蚊、蜘蛛和叶螨多处误框消除；仍有胡蜂/果蝇误框和多蜱漏框，不是全部成功。原全图检测误报252不可与新完整级联22混合比较。

后续109回顾：正例检出53/55→51/55、螨误报32/54→9/54，输出detector_binary_regression_run01/。486物种图片开发：检出483→470，正确候选441→446、正确确认322→334、错误确认3→0，输出detector_species_val_run01/。只有图片标签，不能声称逐框蜱种准确率；旧模型看过部分来源，不能称独立测试。虽然误报和有效物种确认改善，来源检出退步，暂不替换默认。继续plans/proposal_verification_20261026.md，用原框与微调ROI头分离检验proposal损失。80补充物种尚未预测，仍不用于选参。
