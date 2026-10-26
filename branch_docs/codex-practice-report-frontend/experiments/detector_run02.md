# 检测正式重训 run02

登记2026-10-26，完整数据预检通过后已启动，GPU会话46407。run01因一条微小三角形栅格mask为空失败，整张训练图片隔离；不造mask、不把该目标默认为负例。原数据与失败保留。

- 清单 `output/optimization_20261015/rebuild_20261024/data/detector_supervision_split_v2.json`，SHA256 `aee2baf8b3c0b7e9b1fd110501e346f62b638cd7f7d5f30e5185690250e13013`。
- train2937（1123有框正例+1814来源代理负例）、val495、cal484；43条隔离记录含42张未核验空图和1张mask异常训练图片。验证集合与run01不变。
- 参数、初始化、源码与run01训练器相同；只改输入清单与输出目录。先完整3轮，按val F1-.25×代理误报率选模。原v3不覆盖，默认系统保留。
- 启动前check_detector_supervision.py用真实Dataset逐张解码train/val，验证坐标、尺寸和栅格mask；校准图片不读，不运行模型。

```powershell
python src/train/train_detector_rebuild.py --split output/optimization_20261015/rebuild_20261024/data/detector_supervision_split_v2.json --out output/optimization_20261015/rebuild_20261024/detector_run02 --epochs 3 --batch-size 1 --accumulate 2 --workers 2 --amp
```

日志 `logs/detector_run02.log`，产物 `detector_run02/`，根仍为ignored目录rebuild_20261024。指标与证据边界沿用[run01登记](detector_run01.md)，不把开发验证当独立验收。新类别补充正例名单在模型预测之前冻结，只评蜱虫存在性，不对不支持物种伪造28类正确答案。

真实Dataset预检记录data/detector_preflight_v2.json，覆盖train2937/val495；源码SHA为cc3c38c51724e6135e3a7d9aca9b1af6cf0e60a82bee4c6a69385cccc072b94d。其余源码哈希与run01相同，本次config.json单独保留。

第1轮完整完成：train_loss=.19878，无AMP跳过；val495图，定位recall=.72558、precision=.73934、计数MAE=.39362、代理误报66/307。旧v3全图对应recall=.88372、MAE=.20213、代理误报252/307。只是训练中结果，召回/计数未过门槛。CPU比较与完整39例对照在detector_comparison_epoch01/；已目视第一页，可见两张近景蜱虫漏检，不能只展示负例改善。训练继续第2轮。


## 补充来源正例名单

模型预测前冻结80个现有本地、当前28类之外的物种各1图，data/unsupported_tick_challenge_v2.json，SHA256 de4e168d08c6bc72b8838e013211c82580160ffdfeb4d3d245bf594931b18080。筛查全部已知历史split*.json、当前昆虫/COCO各划分和已查看109图，使用SHA、已知photo/observation组与dHash≤4；不是仅排除当前28类训练清单。190个历史引用缺失/不可解码，不能证明完全独立；名单保存具体路径。文件夹物种标签未经逐图独立裁定。此正例集仅查“是否识出蜱虫”，不能算数量/定位/蜱种准确率或误报率；模型/策略冻结后再跑，不据其结果选参。

v1名单只筛当前clean分类来源，遗漏更早历史split，未进行模型预测，已停用并保留，不能与v2混用；首次参考筛查遇截断图片失败日志也保留，修订后记录不可解码而非强制读截断数据。最终使用v2，原日志位于logs/freeze_unsupported_tick_challenge_v3.log。
