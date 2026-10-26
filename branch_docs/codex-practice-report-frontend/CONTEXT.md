# 当前分支上下文

更新2026-10-26。仅在codex/practice-report-frontend工作；主分支保持2029625d3752c14cdd6f268431eb0323e88c2bb7，不修改、不合并。最新已推送代码c8fec462，后续v3微调与完整模型身份变更待提交。用户原有dirty文件不暂存。

## 用户目标与当前计划

整体提高蜱虫数量/定位、蜱种和昆虫识别；允许架构调整和重新训练，只使用项目已有数据，要求肉眼可见改善。优化独立于实践报告。分支文档只维护本目录，原README/docs/论文材料保留基线。

原需求：[识别重建计划](plans/recognition_rebuild_20261024.md)。当前执行：[来源差异与检测微调](plans/recognition_policy_followup_20261026.md)。用户不需重复授权已批准的分支开发。

## 当前GPU任务

**detector_v3_finetune_run01，训练中，会话86243**。日志output/optimization_20261015/rebuild_20261024/logs/detector_v3_finetune_run01.log；检查点同根detector_v3_finetune_run01/。原v3显式SHA绑定，小学习率.0003，1024/1200、batch1/accumulate2、workers2、AMP，先完整3轮。训练源码SHA a5f44e56d1dbafca2086b0628a0cab82b80477d0411f68e7a2b8cd5070c61a43；运行中不能改训练器、安全辅助、几何增强、服务detector或清单。

CPU实际v3权重严格加载与54项回归通过，窄范围独立审查无新增P1/P2。GPU必须串行，不在训练中启动评估/其他训练。Python C:/Users/a/.conda/envs/tick_detection/python.exe；RTX4060 Laptop8GB；命令设置PYTHONDONTWRITEBYTECODE=1、PYTHONIOENCODING=utf-8。

## 已完成与失败证据

- 数据审计18497图片路径，存在跨集合照片/观测、跨任务重用和冲突标签。旧模型已看过部分新val/cal，所有旧新比较明确诊断局限。原数据、权重、划分不覆盖。
- 检测增强已同步图/框/mask；新监督清单data/detector_supervision_split_v2.json为2937/495/484，SHA aee2baf8b3c0b7e9b1fd110501e346f62b638cd7f7d5f30e5185690250e13013。隔离42未经核验空图及1个2×2三角形空栅格mask的整张训练图，val/cal不变。完整3432张train/val已实际Dataset预检通过；1814训练非tick仅来源代理，不是人工逐图真负。
- 分类recognition_run01完成13轮，第14轮非有限梯度失败，best第12轮冻结。新species头不采用。存在性det=.5/presence=.95候选虽开发val/cal过门槛，但109同源图tick漏识别2→6，拒绝默认切换；低分来源门槛+区域coarse guard仍误报79→77，失败保留。
- 检测detector_run01首轮batch2760后遇空mask失败，无完成epoch/正式检查点；detector_run02通用初始化完整3轮，最佳第3轮recall=.75814、precision=.75115、计数MAE=.32979、代理误报49/307。旧v3全图.88372/.84444/.20213/252。召回/计数未过门槛，不部署；只3轮不能推断充分收敛无效。源码快照存detector_run02/source_snapshot，当前源码已改变，不能静默恢复run02。
- 63分类与39定位案例在训练/评估前冻结；组件6页、级联10页、检测每轮10页对照包含失败。第1轮分尺度诊断在data/detector_epoch01_scale_diagnostic.json，多目标漏检严重，后续须单列计数而非只看平均指标。
- 默认配置继续configs/inference_optimization.json、原v3与原分类权重。configs/inference_recognition_rebuild.json仅失败研究候选，不推荐日常启用。新物种头/存在性头未默认接入。
- profile构建已绑定当前baseline文件、模型和源清单，防止旧报告错套新配置。推理结果完整记录各模型SHA；实验profile可绑定detector.model_sha256，bbox_only只关闭不使用的mask回贴，保留Mask R-CNN权重。
- 独立CPU服务HTTP已测2轮：健康/错误MIME/损坏图片/10MB超限拒绝、真实图推理、历史和版本一致性通过。service_smoke_run02输出完整模型SHA，18.40秒是CPU数据，不能视为GPU性能提升。8016验证服务已停止，独立SQLite保存在output，原data.db未操作。

## 已冻结的补充来源

data/unsupported_tick_challenge_v2.json，80个不支持物种各1图，SHA de4e168d08c6bc72b8838e013211c82580160ffdfeb4d3d245bf594931b18080。筛查全部历史split*.json、昆虫/COCO划分及已查看109图的SHA、已知photo/observation组和dHash≤4；190历史引用缺失/不可解码，不能证明完全独立。尚未模型预测，只作冻结策略后的正例覆盖，另查是否错误确认成已知蜱种，不计数量/定位/28类准确率。v1名单遗漏旧split，未预测且已停用，保留。

## 评估入口与明确下一步

1. 等v3微调3轮完整完成，检查epochs.jsonl、best/last、失败与SHA，更新[微调登记](experiments/detector_v3_finetune_run01.md)。原v3保留，尚无新模型达到整体验收。
2. compare_detector_rebuild.py读取每轮预测与旧全图缓存，展示全部39冻结例；新float框对旧服务int框只是诊断，最终服务口径另测。
3. GPU空闲后evaluate_detector_cascade.py用实际Detector整数化/去重/空检出切块验证完整495 val，预先限定drop_in、regional_only、detector_only三策略。输入原baseline traces在cascade_val_run01_det05/traces.json；按召回/计数/代理误报共同选策略。保留旧蜱种模型，不能伪造逐框物种GT。
4. 若通过，冻结profile并绑定候选detector SHA。evaluate_detector_labelled_images.py对species_split.json的486 val作图片级正确候选/确认诊断，再对80补充正例查覆盖；不能根据80结果反复选参。原109/cal已查看，只能开发或回顾。
5. 候选通过再启动独立SQLite真实HTTP，smoke_recognition_service.py传--expected-detector验证实际版本；测GPU请求时间，必要时页面实测。没有通过就不换默认，继续按证据修改。
6. 所有文档只在本目录更新，逐路径提交推送授权分支。用户dirty .gitignore/.learnings、data.db、开发进程/web开发.md、毕业设计规划及任务书输出不提交。不要git add .。

全部大文件根output/optimization_20261015/rebuild_20261024/为Git忽略目录，原图/数据库/模型不推送。当前尚未完成整体肉眼改善目标，不以训练启动或单项误报下降作为完成。
