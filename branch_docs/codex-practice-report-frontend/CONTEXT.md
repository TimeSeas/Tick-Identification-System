# 当前分支上下文

更新2026-10-27。只在codex/practice-report-frontend开发，不修改/合并主分支；main固定2029625d3752c14cdd6f268431eb0323e88c2bb7。不使用Claude。用户批准全程开发、重训练和推送该分支；本轮独立于实践报告。原README/docs/毕业设计材料不追加本分支实验结论。

## 已完成：蜱种专项与保守补充识别

计划[species_specialist_20261027](plans/species_specialist_20261027.md)，结果[species_specialist_run01](experiments/species_specialist_run01.md)。90项unittest通过，完整训练6轮，按预定val分数选第6轮。源码训练/评估提交50baa109、双专家c470cce5、最终补充路径dc0c40ad均已推送；最终代码/默认配置检查点a3cad33c79d27abf2efc2edd024048414a9641c2，最终记录见git log/VERSIONS。

当前默认configs/inference_species_rescue.json，SHA b19a43ab4a7712b575a46bd9aad1f7fbdf600c7105d9a204b1c338204b955231。单框新ROI模型+原整图概率各.5；多框新ROI模型。原分类器同一ROI/整图预测作为reference：**原已确认结论保留，原待确认结果才允许补充确认**。补充门槛81.60640001296997%，仅calibration原待确认提案选择，新增经验precision100%、新增未知确认0；不代表未来保证。M1强支持只读原reference的确认/概率，不能由新确认复活框。

常驻4份权重：
- M1：output/optimization_20261015/rebuild_20261024/detector_v3_finetune_run01/frozen_detector.pth，SHA 9e7a3a9b9cb32df87a37be71d85068f98a28d17bb61f828354534d4e95ed8b7c。1024/1200、.3、IoU.3、空全图才tiles2、bbox-only。
- 新物种ROI：output/optimization_20261015/species_20261027/component_calibration/candidate.pth，SHA d6605ee6e97f7f13e08bc44e036f2841b2bf3b9be3724efe75e2a8347b1c9f00，ResNet50/28类/448。
- 原物种reference：output/tick_species_clean_448/model/tick_species_best.pth，SHA e5301d269a45e0925d21f4573a503b8cc112834374e077db360595163859ab56；SPECIES_WHOLE_MODEL_PATH绑定此权重。原single/region门槛应用于reference，rescue_thresholds才应用于新模型。
- 原昆虫：output/insect_classifier/model/insect_best.pth，SHA 056c119a900b38dc3a7e679e4b036d0c2d67454475ae3b8972afdd24ef7c35c3。

## 最终效果与界限

486已知物种开发图片：正确候选448→453，正确确认334→342，错误确认图片1→1，检出476→476。不是每个框的正确率；图片文件夹标签不是逐框物种真值。组件val918视图top1 .863834→.905229，不应替代完整级联数据。

495检测开发图片：precision .892019、recall .883721、count MAE .127660、307代理非蜱虫中32张检出，全部保持原值；代理负例蜱种确认0→0。109回顾TP52/FN3/FP13/TN41不变。所有新旧框和原已确认种名/概率/门槛已逐条比较一致。

未知物种**未解决**：已查看80图检出66、错误确认9，新旧一样；冻结后24图/10种留出检出19、错误确认4，新旧一样，没有新增确认。24图已在本轮查看，今后只能回顾，不再称新未见验收，也不能用于选参。训练旧权重见过部分known val来源、190历史引用缺失，不能宣称无泄漏独立泛化。新holdout规模小且仅未知类，不能代表整体性能。

## 实际运行与交接

没有离线训练/评估在运行。后端8000 PID32036，CUDA；前端5174 PID36224。服务由Start-Process -WindowStyle Hidden启动，PID使用前须现场核验，勿碰其他项目进程。8016临时服务已停止。当前数据库output/optimization_20261015/species_20261027/browser_rescue.db，4条验证记录；原web/backend/data.db及上一轮20条browser_tests.db保留。

网页三例：太平洋硬蜱原48.9%待确认→85.9%确认；肩板硬蜱39.5%保留待确认/候选；蚜虫86.7%粗分类且无蜱虫框。20,119,028像素图片被正常拒绝、未入库，原图未修改；另用有效尺寸蚜虫完成测试。真实CUDA HTTP、无效MIME/损坏图/10MB拒绝、JPEG、历史详情与四模型版本均通过；单图1.2866秒，不是平均/受控速度比较。Docker compose config通过，未构建/实跑容器。

## 产物、失败与下一步

根output/optimization_20261015/species_20261027/（Git忽略）：data/manifest.json SHA8325bd47910d03cef5381b0c89b4a09ae788d326579ca502841f3fd15f791a21；5459/918/935已知视图，145/53/26/24未知图、31/10/10/10未知种，tick_label=1。167旧错误只复核，不入训练。训练checkpoint SHA94e6c6ac6fe22c9ad6763531cf9634753b8085e715dec1b768d3b7522054be50。未知数据含标本瓶/标签，不能把全图降置信度当真实区域拒识已成功。

rescue_calibration/冻结配置与policy；rescue_val/、rescue_detector_val/、rescue_binary/、rescue_old_unknown80/、rescue_holdout/均完整结束；frozen_rescue_checks.json逐条契约通过。cpu_rescue_smoke/、gpu_rescue_smoke/、browser_report.json、browser_cases.json、browser_confirmed_verified.png和rescue_visual_comparison.jpg保留真实证据。release_manifest.json（SHA 4e14467225084ee30b39a75cefb1d0c59f8a1706c2bd485fedc4a1077febe87a）绑定源码/权重/数据/报告/部署配置。configs/inference_species_rescue.json通过.gitattributes -text按字节保存，Git blob/工作区/冻结profile SHA一致，勿格式化换行后继续沿用原身份。

直接全替换、双专家global、双专家classwise已否决（后者val正确确认334→327、错误1→4）；全部原始产物保留，不得当已通过方案。旧recognition_run01双头/VerifiedDetector也未启用，历史见各实验。过程上下文快照在history/2026-10-27/。

下一轮优先：人工复核可见蜱虫的未知自然图与teacher框，分开标本瓶/标签；补充同域近似蜱种/螨的区域级标注，再研究独立拒识。需要新冻结未知评估来源，不能继续针对旧80或本轮24调参；稀少类与多虫混合图仍缺逐框物种真值。超20MP照片自动缩小属于可用性后续项，当前只给出明确拒绝信息。

Python C:/Users/a/.conda/envs/tick_detection/python.exe；PYTHONDONTWRITEBYTECODE=1、PYTHONIOENCODING=utf-8；一个GPU任务，训练前处理已知常驻后端，不能改活跃任务源/数据。恢复训练仅最新完整checkpoint和相同运行身份，partial不能续正式训练。

提交时只逐路径暂存本任务。用户未提交的.gitignore、.learnings、web/backend/data.db、开发进程/web开发.md、毕业设计/毕业设计系统规划.md及output/taskbook_revision_20261024/保持原状，不加入提交。
