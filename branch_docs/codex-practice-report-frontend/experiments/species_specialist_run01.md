# 蜱种专项 run01 与保守补充识别（2026-10-27）

用户批准在codex/practice-report-frontend全程优化；独立于实践报告，不修改或合并主分支。计划[species_specialist_20261027](../plans/species_specialist_20261027.md)。源码阶段50baa109（训练/评估）、c470cce5（双专家与classwise）、dc0c40ad（最终补充路径），均已推送；最后默认/记录见VERSIONS及git log。

最终采用**保留原确认、只补充待确认**，不是直接替换全部物种判断。检测器与昆虫模型不变，增加单独区域物种模型，总计4份权重常驻。未知误确认目标未达成，如实保留该限制。

## 冻结数据与训练

根output/optimization_20261015/species_20261027/，原图片/权重/划分不覆盖、不推送。data/manifest.json SHA8325bd47910d03cef5381b0c89b4a09ae788d326579ca502841f3fd15f791a21。

|子集|已知图片/视图|未知图片/物种|
|---|---|---|
|train|2889 / 5459（2570 ROI）|145 / 31|
|val|486 / 918|53 / 10|
|calibration|492 / 935|26 / 10|
|holdout|0|24 / 10|

旧80的物种全部排除新增训练；未知物种跨4子集隔离，tick_label=1，仅做物种分布均匀化，不当非蜱虫。筛查18967历史引用、SHA/photo/observation与dHash≤4，隔离158候选，190引用缺失/不可读，不能证明完全独立。167旧错误只保存在review_queue，不混入训练。原known分组保留，原权重见过部分known val来源，所有known结果是开发/回顾。

已看known_roi_preview_01/02及unknown_preview_01/02，覆盖28已知/31未知种各一例；未知数据存在标本瓶/标签及过小主体，全图降置信度不能代表实际ROI拒识有效。epoch01_comparison与最终rescue_visual_comparison均包含改善和失败。

原ResNet50/28类448权重明确warm-start，SHA e5301d269a45e0925d21f4573a503b8cc112834374e077db360595163859ab56。冻结layer1/2与BN统计，训练layer3/4/fc；已知CE（平滑.05）+.5蒸馏T2，未知.2均匀化；已知/未知采样质量.85/.15，平方根频率平衡；lr2e-5、AdamW、batch8/accumulate2、AMP、workers2、448、.15上下文、训练随机center/full，验证mean(center/full)。

pilot仅4批、partial，top1=.839869，不能部署或续成正式训练。run01完整6轮，0 AMP跳过，第6轮按预定val分数选中。checkpoint SHA94e6c6ac6fe22c9ad6763531cf9634753b8085e715dec1b768d3b7522054be50；导出component_calibration/candidate.pth SHA d6605ee6e97f7f13e08bc44e036f2841b2bf3b9be3724efe75e2a8347b1c9f00。原始每轮预测、完整日志、全部检查点与training_summary保留。

组件val918已知视图top1 .863834→.905229、macroF1 .857807→.895310；53未知最大概率均值.651945→.137514。cal935视图top1 .858824→.875936；teacher ROI top1 .747178→.794582，但single_proxy融合 .965447→.943089。该差异解释为什么不按混合视图均值直接替换。

## 候选演进与否决

所有数均为图片文件夹标签下的覆盖，不是逐框物种真值。实际校准使用当前M1原生全图/空全图tiles与真实Pipeline，不把旧原v3结果混作本轮基线。

|方案/产物|已知正确候选|已知正确确认|错误确认图片|未知确认|结论|
|---|---|---|---|---|---|
|直接替换 cascade_calibration（492已知/26未知）|448→443|327→170|4→0|6→1|拒绝，不能大量拒识换安全|
|双专家global dual_calibration|448→458|327→233|4→0|6→3|拒绝|
|classwise_run02 calibration|448→458|327→328|4→4|6→3|仍需val|
|classwise完整val（486已知/53未知）|448→453|334→327|1→4|2→1|拒绝，完整val退步|
|最终rescue calibration|448→458|327→333|4→4|6→6|继续冻结验证|
|最终rescue完整val|448→453|334→342|1→1|2→2|采用保守补充|

双专家为新ROI+原整图各.5，多框只用新ROI；classwise初轮DP按提案重复计收益有缺陷，留在dual_classwise_calibration作历史。run02修复为来源去重、惩罚错误来源，仍完整val失败，未切默认。其cpu_smoke虽然接口通过，也不能当最终方案证据。

最终decision_mode=preserve_confirmed_rescue：原模型的同一ROI/整图参考概率和原single/region表决定是否保留原确认；原待确认时才用新ROI+原整图补充。M1 strong_box永远只用原reference概率/确认，新增确认不能复活框。

仅calibration原待确认提案选全局补充阈值.8160640001296997（81.6064%），要求新增经验precision100%、新增未知确认0；原确认来源不抬高新增精度。rescue_calibration/frozen_policy.json绑定完整profile、主/参考模型、manifest、校准报告/原始记录、evaluator/calibrator/service源码SHA。新holdout只在冻结后跑一次，没有反向调参。最终configs/inference_species_rescue.json与冻结profile字节相同，SHA b19a43ab4a7712b575a46bd9aad1f7fbdf600c7105d9a204b1c338204b955231。

## 完整回归与未解决风险

- 495开发图（188有框正例+307来源代理非蜱虫）：tp190/fp23/fn25、precision .892019、recall .883721、F1 .887850、计数MAE .127660、代理检出32/307，全部新旧一致。代理负例中已确认蜱种0→0。
- 109回顾：TP52/FN3/FP13/TN41，保持不变。
- 旧80未知种回顾：检出66、错误确认9，新旧一致。
- 新冻结24图/10种未知留出：检出19、错误确认4，新旧一致；没有新增确认，也没有减少已有错误。24图已查看，后续只作回顾，不能再称新未见评估。
- frozen_rescue_checks.json逐条核对所有5个集合的bbox/score与原已确认种名、概率、门槛一致。未知类无新增确认。独立审查另做540组CPU属性组合，保持框及原确认。

known来源存在历史重用，未知留出小且无已知/非蜱虫对照；190缺失引用限制独立性。图片目录不等于所有框都有该种，无独立逐框物种/计数真值；不声称整体所有识别能力都提高。**原未知误确认9/80及4/24仍在**，该轮未知拒识目标未完成。

## 接入、真实HTTP与浏览器

Settings/Compose同步主物种路径、profile及SPECIES_WHOLE_MODEL_PATH；原权重与balanced profile保留回滚。90项unittest通过，覆盖损失/未知隔离、图像SHA/恢复覆盖、numpy契约、来源分母/去重、冻结策略/split身份、原确认与框保持及测试DB边界。Docker compose config通过，未构建/实跑。

cpu_rescue_smoke/与gpu_rescue_smoke/均在新空测试库通过健康、非法MIME/损坏图片/10MB拒绝且不入库、JPEG可解码、真实补充确认、历史详情与四模型版本一致。CPU单图23.3933秒；CUDA单图1.2866秒，非平均/受控对比。

浏览器固定开发案例browser_cases.json：太平洋硬蜱原48.9%待确认→85.9%确认；肩板硬蜱39.5%待确认、候选/Top3保留；蚜虫86.7%兜底、无蜱虫框。首次蚜虫图5494×3662超过20MP被明确拒绝未入库，原图不改，另选有效尺寸Image_46完成；这是可用性边界，不掩盖失败。3次有效网页上传+1次HTTP，共4条记录，API逐条核验同一profile。browser_report.json、完整UI文本、browser_confirmed_verified.png及改善/失败9例对照均保留。部分紧接动画的中间截图仅作过程，最终以verified截图和完整UI文本为准。

当前后端8000 PID32036为CUDA，前端5174 PID36224；8016临时CPU已停止，无离线任务。DB为browser_rescue.db；原数据库与前轮20条库保留。启动状态services.json，进程操作前现场核验。训练/评估/图片/数据库均在ignored目录，不推送。

## 审查与后续

重要问题均先复现再修复：图片原地变更、旧checkpoint覆盖未来、known漏检分母、numpy契约、冻结模型/阈值/split、重复提案收益。没有未处理P1/P2。未知自然图/teacher框需人工复核并补同域区域标注；将标本瓶/标签单列。下一轮未知评估必须换新冻结来源，不针对旧80或本轮24选参。超20MP自动缩小属于后续可用性工作。
