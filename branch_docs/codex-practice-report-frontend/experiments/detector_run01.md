# 检测正式重训 run01

登记：2026-10-26，已启动，GPU会话91302，先完整3轮。原v3不覆盖、不切默认。决策见[当前计划](../plans/recognition_policy_followup_20261026.md)。

输出根 `output/optimization_20261015/rebuild_20261024/`；检查点 `detector_run01/`，日志 `logs/detector_run01.log`。

- 输入 `data/detector_supervision_split.json`，SHA256 `6625ae25affc8aeeebc64ed0d569012ea44227d0fdcd205518e13422ea9ef896`。2938训练（1124有框正例+1814代理负例）、495验证、484校准；隔离42张未核验空图。原清单保留，不删除图片。
- 通用COCO初始化 `maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth`，SHA256 `bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0`，不从旧任务模型初始化。
- Mask R-CNN ResNet50-FPN，两类别，1024/1200，冻结低两阶段，SGD lr=.001、momentum=.9、wd=.0005，StepLR5 gamma=.1，seed42。
- batch1/accumulate2、workers2、AMP；同步图/框/mask增强。文件夹代理规范EXIF，COCO原坐标。AMP非有限梯度跳过降scale，连续5次失败；FP32异常失败。
- 完整val选分 F1-.25×代理图误报率，IoU=.5匹配、服务IoU=.3去重。临时关闭mask回贴避免显存耗尽，训练mask损失保留。
- config.json记录训练器、安全辅助、几何增强和服务检测器源码SHA；保存每轮预测/检查点、best/last及失败。训练期间源码不变。只允许原目录同配置恢复；partial不能作为正式模型。

```powershell
python src/train/train_detector_rebuild.py --split output/optimization_20261015/rebuild_20261024/data/detector_supervision_split.json --out output/optimization_20261015/rebuild_20261024/detector_run01 --epochs 3 --batch-size 1 --accumulate 2 --workers 2 --amp
```

val用于开发选模，不是最终独立测试；来源代理不是人工逐图核验真负。旧v3见过部分来源，旧新比较须注明边界。3轮是首阶段预算，检查趋势后再决定是否延长，完整保留失败。

训练后使用compare_detector_rebuild.py读取每轮val预测和已缓存v3全图检测，比较相同495张像素上的原始检测能力；完整渲染训练前固定39例，保留失败。此比较只含全图检测，不混入切块或旧级联指标。之后另测完整级联，不能把检测器指标当网页最终结果。

## 本次冻结源码

- service_detector_sha256: `f7ddd2d5206b65b0d120c66dfa258dd63746a6bcb88da6ffb21381dd1e30d11a`
- trainer_sha256: `cc3c38c51724e6135e3a7d9aca9b1af6cf0e60a82bee4c6a69385cccc072b94d`
- training_safety_sha256: `d23e5393ab6ace38d5f34a2079c7997cf0a3a3c6619ddd3bfa310e8b74965b4d`
- joint_transform_sha256: `441605def34777fa24b462b587a98ce9e8b2357ed6b6d5ac94ce6734fac11a46`
