# 现役v3困难负例微调 run01

登记2026-10-26，已启动，GPU会话86243。动机：通用初始化3轮虽减少误报但损失定位召回，尝试保留现役能力再学习困难代理负例。不是全新未见来源训练，val只作开发比较。

- 冻结清单同detector_run02：`data/detector_supervision_split_v2.json`，SHA256 `aee2baf8b3c0b7e9b1fd110501e346f62b638cd7f7d5f30e5185690250e13013`，2937/495/484。
- 显式从`output/det_v3/mask_rcnn_ticks_v3.pth`初始化，SHA256 `4e14eeb2fae71667d5cd40e901f0a99d1ad68a6a0d212b31f91bc7cdd06929ae`。旧模型见过部分新val来源，不能描述为独立测试或重新洗白旧来源。
- 通用COCO缓存仍用于构建一致架构，随后严格加载全部v3参数。1024/1200、冻结低两阶段、SGD lr=.0003、momentum=.9、wd=.0005，StepLR5/.1、batch1/accumulate2、workers2、AMP、seed42，先完整3轮。
- 选模仍为val F1-.25×代理误报率；另外必须检查召回不降超过2pp、计数和完整级联。代理负例并非人工真负。保留每轮预测、best/last、失败及源码/模型哈希，不覆盖原文件。
- 输出 `output/optimization_20261015/rebuild_20261024/detector_v3_finetune_run01/`；日志 `logs/detector_v3_finetune_run01.log`。原通用初始化训练源码快照保存在detector_run02/source_snapshot/，不得以新源码静默恢复该run。

```powershell
python src/train/train_detector_rebuild.py --split output/optimization_20261015/rebuild_20261024/data/detector_supervision_split_v2.json --out output/optimization_20261015/rebuild_20261024/detector_v3_finetune_run01 --epochs 3 --batch-size 1 --accumulate 2 --workers 2 --amp --lr .0003 --warm-start output/det_v3/mask_rcnn_ticks_v3.pth --expected-warm-start-sha256 4e14eeb2fae71667d5cd40e901f0a99d1ad68a6a0d212b31f91bc7cdd06929ae
```

冻结训练器SHA256：a5f44e56d1dbafca2086b0628a0cab82b80477d0411f68e7a2b8cd5070c61a43。其余源码及输入身份由本run config.json记录；当前不能更改训练相关源码。
