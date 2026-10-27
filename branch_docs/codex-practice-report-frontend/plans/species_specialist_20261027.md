# 蜱种专项训练与未知物种拒识 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans，按任务顺序直接实施；每项保留检查证据。

**Goal:** 提高28类蜱种的正确候选和正确确认覆盖，减少未覆盖物种被确定命名，保留当前检测和计数能力。

**Architecture:** 冻结现役Mask R-CNN及昆虫分类器，单独小学习率微调原448 ResNet-50蜱种模型，原图/带上下文区域混合训练。首候选仅训练layer3/layer4/fc，BatchNorm运行统计冻结；已知类交叉熵加原模型蒸馏，未覆盖蜱种使用物种分布均匀化损失，不把它们当作非蜱虫。基于独立校准子集选择确认/拒识规则，完整验证后再决定接入。

**Tech Stack:** 现有PyTorch/torchvision、PIL、numpy、unittest、FastAPI，不新增依赖。

**Spec:** 用户2026-10-27批准“错误样本整理→单独蜱种训练→未知蜱种拒识”的建议；既有整体需求见recognition_rebuild_20261024.md。

## Global Constraints

- 只在codex/practice-report-frontend开发，不修改/合并主分支；只用项目已有数据。
- 独立文档只维护本目录，产物根为output/optimization_20261015/species_20261027/。
- 原数据、权重、划分、数据库和上一轮实验不覆盖。默认balanced配置在候选通过前冻结。
- 旧20/109/486/80均已查看，只作开发/回顾。新未知物种按物种分开train/val/calibration/holdout，并排除旧80所代表的全部物种。
- 新增训练数据筛查SHA、photo/observation来源及dHash近重复；冲突隔离，不按模型预测改标签。原模型看过部分已知val来源，微调对照不称独立泛化测试。
- 同GPU只运行一个任务，活跃任务源文件/权重/清单不修改。训练前核验网页后端PID和命令，仅停止本任务的CUDA后端，结束恢复。

## Review Focus

1. 未覆盖蜱种不能被标为非蜱虫；拒识只影响种名确认，不隐去检测框。
2. 同照片不同尺寸/裁剪、同观测与全图/区域视图不得跨集合；未知物种本身也不得跨集合。
3. 增强/推理保持EXIF、上下文和center/full视图一致；原始COCO坐标不静默旋转。
4. 没有足够已知类/未知类校准样本时不得自动发布或降低标准；不能用全部拒绝提高表面正确率。
5. 重启/恢复必须绑定数据、源代码、原权重、类别顺序和预处理身份；partial、失败或改变身份的产物不作为正式模型。

## Task 1: 整理错误与冻结专项数据

**Files:** 新建scripts/prepare_species_specialist.py、tests/test_species_specialist.py；复用已有group/fingerprint/prepare_image。

**Interfaces:** build_unknown_assignment(labels,seed)->dict[str,str]；validate_specialist_manifest(manifest)->list[str]；输出known全图/区域与unknown全图统一行（path/sha256/group_id/species_label/bbox/subset）。

- [x] 先写失败测试：未知物种按物种隔离、已知/未知标签契约、SHA/group跨集合拒绝、训练未知与旧80物种分离。
- [x] 实现数据构建，读取旧regions/recognition_split.json的已知类视图，未知候选为现存剩余物种；记录所有筛查和缺失，不覆盖旧清单。
- [x] 从上一轮训练集分类错误和已记录浏览器额外小框生成复核清单；训练错误用于后续样本检查，浏览器样本只作回顾不混入训练。
- [x] 冻结名单、SHA及固定可视化例，检查实际数量/类别/来源分布及图像尺寸。

## Task 2: 单独物种训练

**Files:** 新建src/train/train_species_specialist.py，测试扩展test_species_specialist.py；复用training_safety.py。

**Interfaces:** specialist_loss(logits,teacher_logits,labels,oe_weight=.2,distill_weight=.5)->Tensor；原始28类ResNet50 state_dict兼容旧分类器。

- [x] 先写失败测试：未知样本不计算已知物种硬标签、均匀化有梯度、全未知/全已知批次有效、类顺序与原权重身份不匹配拒绝。
- [x] 显式载入原蜱种权重及SHA，448输入，center/full训练视图随机混合，.15区域上下文，水平/垂直翻转和轻微颜色/旋转增强。
- [x] 冻结layer1/2与BN统计；AdamW lr=2e-5，batch8/accumulate2、AMP、workers2，已知采样质量.85、未知.15，已知类使用平方根频率权重限制稀少类重复。
- [x] 4步pilot完整val验证通过后，新目录正式最多6轮，patience3；先按已知top1/macro-F1、未知最大概率分离度共同检查，保存每轮原始logits和完整日志。
- [x] 校验训练失败/AMP跳过/恢复身份，不改正在运行的源文件；冻结完整候选。

## Task 3: 校准、回归和可选接入

**Files:** 新建scripts/evaluate_species_specialist.py；必要时小范围修改classifier.py/pipeline.py/main.py及对应测试；新建独立profile。

- [ ] 对同输入旧/新分类器比较full与带上下文区域、逐种top1/macro-F1；旧模型已见来源局限明确记录。
- [ ] 仅calibration选确认门槛和未知拒识门槛，约束已知确认正确率≥95%，相对当前正确确认覆盖争取提高≥5pp；未知错误确认率争取相对降低50%，小样本不承诺达成。
- [ ] 新未知holdout只在模型/门槛冻结后评估一次；已查看80仅回归，不能据其再次选参。未通过则不换默认，记录原因并按错误分析决定下一候选。
- [ ] 候选通过后才接入：保留候选种名，拒识显示未覆盖/待确认；检测框不因拒识而丢失。严格绑定模型/类别/配置SHA，完整495检测、486物种及109回归，检查误报/计数/召回不退步。
- [ ] 独立数据库真实HTTP和浏览器成功/失败案例验证；恢复网页服务、更新部署/CONTEXT/HISTORY/VERSIONS，逐路径提交推送授权分支。

## 本轮决策与执行记录

用户已明确“开始优化吧”，沿用其全程开发授权，直接执行计划，不重复请求许可。为降低上下文与额度开销，主agent串行实现，完成时作独立只读审查。当前数据清点：已知训练2889全图+2570区域；排除旧80物种后仍有67个其他文件夹、406张图片，尚未筛查，不能先写成可用训练数。

### Task 3 ruling：先保留完整失败，再按验证集分工组合

6轮训练完整结束，第6轮按预定val分数胜出。组件val视图top1 .863834→.905229，但这混合了全图/ROI。按486图重新组合概率：原整图+原ROI=.960905，新整图+新ROI=.952675，新ROI+原整图=.967078；因此采用双专家离线候选，单框只把原模型用于整图、新模型用于上下文ROI，权重仍.5；多框只用新ROI。不是直接比较两个模型与昆虫模型的置信度。

原直接替换+绝对5%未知确认校准已失败：492已知正确确认327→170，候选448→443；26未知误确认6→1，不能以拒绝大量已知图宣称成功，未接入。该目录和冻结profile保留。下一候选在校准集约束已知确认precision≥.95、未知误确认相对旧策略至少减半，争取正确确认覆盖不下降；这是将原计划“相对降低50%”落实为校准规则，不承诺绝对5%。新的holdout24图尚未查看。代价：未知仍可能被确认，额外常驻一个原物种模型；必须实际报告。

服务支持可选whole_reference SHA绑定，未通过完整级联/来源/HTTP前默认不变。CPU用真实probabilities的numpy契约测试单框分工、多框无额外整图计算，未知裁剪不改成非蜱虫。

### Task 3 ruling：确认门槛按来源图片计收益

双专家全局门槛.799309仍正确确认327→233，未接入。随后只对至少20预测来源的大类学习门槛，稀少类候选仅原54.873%全局fallback或拒识，保留种名候选；需要确认集precision≥95%。未知预算按每类来源数相加，保守不超过旧校准误确认的一半；不把未知全图降置信度当实际裁剪结果。classwise初轮正确确认327→331、候选448→458、未知6→3，但DP按提案条数收益可能让重复框垄断预算，已留出初轮目录作历史。新测试RED复现100重复提案覆盖1图胜过20图，改为来源去重收益并惩罚错误来源，84测试GREEN；另目录run02重校准。24图holdout仍未查看。代价：部分稀少/易错物种可能只显示待确认，只有完整回归实际覆盖不退步才考虑接入。
