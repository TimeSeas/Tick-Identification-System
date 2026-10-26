# 实验登记与证据规则

每个训练/消融在开始前建立唯一记录，包含状态、源提交、参数、数据及权重SHA256、固定验证/验收定义、实际命令、日志/检查点路径。完成后记录完整结果及失败原因。仅启动后台任务时状态必须是running，不能写为完成。

| 实验ID | 状态 | 证据/说明 |
|---|---|---|
| legacy-20261015 | 历史完成 | dbf87bc1；详见history/2026-10-15，2026-10-24审计已指出数据重用局限 |
| rebuild-20261024-audit | 完成 | ignored目录data_audit.json和training_audit.md |
| rebuild-20261024-data | 完成 | ignored目录data/中的冻结清单与摘要，分组断言通过 |
| rebuild-20261024-recognition | 候选未通过整体验收 | 见[run01登记](recognition_run01.md)；13轮完成，第14轮失败；新物种头拒用，presence候选同源漏识别增加，不切默认 |
| rebuild-20261024-detector | 通用run02完整3轮未达门槛；v3微调完成 | [run01](detector_run01.md)空mask失败；[run02](detector_run02.md)召回失败；[v3微调](detector_v3_finetune_run01.md)完整3轮，冻结第2轮 |
| detector-calibration-20261026 | 完成并切换分支默认 | [门槛与架构核验](detector_calibration_run01.md)：有效run02、来源核验与CUDA HTTP run04完成，保留失败；默认balanced，未知物种和少量来源漏检仍有限制 |
| browser-species-20261026 | 完成，服务保持运行 | [浏览器物种核对](browser_species_20261026.md)：随机10种20图，15图正确确认、5图仅待确认，额外不一致候选保留；独立测试档案 |

本轮大文件根路径：`output/optimization_20261015/rebuild_20261024/`。新实验使用独立子目录，不覆盖已有best/last。参数选择只用验证；校准集只做阈值，验收集在冻结后评估。不存在充分独立验收时必须标明“同源对照/历史回顾”，不能用措辞替换证据。

新候选：[现役v3微调登记](detector_v3_finetune_run01.md)，小学习率困难代理负例训练，val明确为已见部分来源的开发比较。
