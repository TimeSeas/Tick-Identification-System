# 实验登记与证据规则

每个训练/消融在开始前建立唯一记录，包含状态、源提交、参数、数据及权重SHA256、固定验证/验收定义、实际命令、日志/检查点路径。完成后记录完整结果及失败原因。仅启动后台任务时状态必须是running，不能写为完成。

| 实验ID | 状态 | 证据/说明 |
|---|---|---|
| legacy-20261015 | 历史完成 | dbf87bc1；详见history/2026-10-15，2026-10-24审计已指出数据重用局限 |
| rebuild-20261024-audit | 完成 | ignored目录data_audit.json和training_audit.md |
| rebuild-20261024-data | 完成 | ignored目录data/中的冻结清单与摘要，分组断言通过 |
| rebuild-20261024-recognition | 正式训练中 | 见[run01登记](recognition_run01.md)；区域清单完成，4步pilot通过，run01最多15轮，未替换默认模型 |
| rebuild-20261024-detector | 试跑完成，未正式训练 | pilot01 OOM、pilot02 EXIF失败均保留；pilot03完成4步及完整500图验证，仅partial；17项检测回归通过 |

本轮大文件根路径：`output/optimization_20261015/rebuild_20261024/`。新实验使用独立子目录，不覆盖已有best/last。参数选择只用验证；校准集只做阈值，验收集在冻结后评估。不存在充分独立验收时必须标明“同源对照/历史回顾”，不能用措辞替换证据。
