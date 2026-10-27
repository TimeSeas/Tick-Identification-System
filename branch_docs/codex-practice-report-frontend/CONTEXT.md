# 当前分支上下文

更新2026-10-27。当前为codex/practice-report-frontend；主分支保持2029625d3752c14cdd6f268431eb0323e88c2bb7，不修改、不合并。不使用Claude Code。分支开发、训练和维护文档已获用户授权，无需重复询问。用户已明确“开始优化吧”，执行蜱种专项训练与未知蜱种拒识；本轮独立于实践报告。

## 正在执行：2026-10-27蜱种专项

当前计划plans/species_specialist_20261027.md，实验experiments/species_specialist_run01.md。新产物根output/optimization_20261015/species_20261027/。冻结manifest SHA8325bd47910d03cef5381b0c89b4a09ae788d326579ca502841f3fd15f791a21，已知视图5459/918/935；未知图片145/53/26/24，物种31/10/10/10，旧80物种排除。167旧错误仅复核，不转训练。190历史引用缺失，独立性局限保留。

新单头ResNet50从原蜱种权重小学习率微调，加已知蒸馏和未知均匀化；现役检测器及线上物种默认保留。69项CPU回归通过；训练前只读审查发现并修复图片SHA与回退恢复覆盖两项P2。pilot已完整结束（4步、partial，不部署），正式run01正在训练，最多6轮/patience3；日志output/optimization_20261015/species_20261027/run01.log，exec会话98081。第1轮known top1=.883442（原=.863834），macroF1=.891377（原=.857807），未知53视图最大概率均值=.247461（原=.651945）；这是组件开发验证，尚无校准/级联结论。禁止并行GPU任务或修改活跃训练源码/数据。

## 当前默认与运行状态

本轮已完成可回滚的模型优化版本，本地Settings与Compose默认已切换。默认配置configs/inference_detector_balanced.json（SHA256 6164cdf41a781fa3fcc43ebc5990ad460a238b3ff4efc6bd167298fa8a2e86d0），检测权重output/optimization_20261015/rebuild_20261024/detector_v3_finetune_run01/frozen_detector.pth（SHA256 9e7a3a9b9cb32df87a37be71d85068f98a28d17bb61f828354534d4e95ed8b7c）。完整3轮，选第2轮；第3轮召回退步，拒用。

检测1024/1200、门槛.3、IoU=.3、全图后空检出才切块；关闭不使用的mask回贴。原28类448蜱种与昆虫模型、校准、规则不变。原v3及configs/inference_optimization.json完整保留回滚。configs/inference_detector_finetune.json（.5）仅研究对照，双头新物种头/存在性头及VerifiedDetector均不启用。

正式蜱种专项训练正在运行（见上）。旧20张浏览器核对已完成。为训练释放GPU，仅核验并停止本任务后端27396，再以CUDA_VISIBLE_DEVICES=-1启动CPU后端8000 PID36776；前端5174 PID36224，使用Start-Process -WindowStyle Hidden独立后台启动；端口与PID需现场检查，旧会话内进程曾随会话结束退出。后端暂用CPU，日志species_20261027/backend_cpu.stdout.log和stderr.log；训练/离线评估结束后恢复CUDA后端并更新services.json。其他项目服务不操作。8016旧验证服务已停止。当前DB_URL指向output/optimization_20261015/rebuild_20261024/browser_species_20261026_run01/browser_tests.db，原web/backend/data.db未操作。未构建/实测Docker。

上一任务是重启后端、随机选带物种文件夹名图片做浏览器测试，已完成10种20张。至少一个匹配候选20/20；正确已确认15/20；仅待确认5/20；错误已确认0/20。共22个预测框，21个候选与图片文件夹一致，7框待确认；02号有额外小框候选全沟硬蜱，与希伯来花蜱文件夹不一致，疑似误检/局部重复，缺少逐框真值不能擅自删掉。详见experiments/browser_species_20261026.md，名单、完整页面文本、逐框JSON、20张截图及档案截图均保存在同名ignored输出目录。不能把20/20图片有正确候选说成所有框全对或独立准确率。

Python C:/Users/a/.conda/envs/tick_detection/python.exe，RTX4060 Laptop8GB。设置PYTHONDONTWRITEBYTECODE=1、PYTHONIOENCODING=utf-8；环境无pytest，使用python -m unittest discover -s tests -q，62项通过。同一GPU只运行一个任务，活跃任务的源文件/权重/清单不修改。

## 完整效果与限制

495张开发val（188有标注正例+307来源代理非蜱虫）：实际完整级联定位召回87.91%→88.37%，precision84.38%→89.20%，计数MAE .20745→.12766，代理误报79→32，正确昆虫兜底确认228→272/307。不是把原全图252误报与新完整级联混合比较。39训练前冻结案例10页全部查看，既有改善也有胡蜂/果蝇误报、植株多蜱漏检。

109回顾：蜱虫检出53→52/55，螨误报32→13/54。486已知物种开发图片：检出483→476，正确候选441→448、正确确认322→334、出现错误确认图片3→1。来源检出仍有小幅损失，不声称全面提高。物种头本身未重新提升；图片标签不是逐框物种/计数真值，以上是覆盖数。

80不支持物种补充正例在最终配置冻结后才预测：检出68→66/80，新旧均9/80被错误确认为已知蜱种。未知物种问题未解决。该清单筛查全部历史split、SHA、photo/observation与dHash≤4，但190历史引用缺失/不可读，不能证明完全独立。每物种1图、无负例或框真值，不是可靠物种准确率/定位评估。该80已查看，今后只能回顾，不能再次称全新未见验收。

数据审计存在跨集合照片/观测及跨任务重用；旧模型见过部分新val来源。所有对照是开发/回顾证据，不能声称无泄漏独立测试。非蜱虫为来源文件夹代理标签，不是逐图人工不存在蜱虫的证明。

## 证据与文档入口

- 全部大产物根output/optimization_20261015/rebuild_20261024/，Git忽略，原模型/图片/数据库不推送。release_manifest.json绑定本轮配置、模型、评估报告、HTTP及部署源码SHA；版本清单在VERSIONS.json。
- 原需求plans/recognition_rebuild_20261024.md；演进plans/recognition_policy_followup_20261026.md、plans/proposal_verification_20261026.md。本轮完整登记experiments/detector_v3_finetune_run01.md和experiments/detector_calibration_run01.md；运行/回滚在operations/deployment.md。
- 有效级联cascade_detector_finetune_det03_run02/；其他核验detector_binary_det03_run02/、detector_species_balanced_run01/、detector_unsupported_balanced_run01/。原始预测、metadata和日志均保留。
- 真正CUDA HTTP service_smoke_run04/，独立service_smoke_run04.db：健康、错误MIME/损坏图片/10MB拒绝且不入库、真实图、JPEG可解码、历史详情和全模型版本一致。单图.806秒，不是平均时延或CPU对比。run03接口通过但工具JPEG解码错误，导出不可用，保留失败；修复data URI前缀和图像校验后另目录run04完成。
- 监督清单data/detector_supervision_split_v2.json，2937/495/484，SHA aee2baf8b3c0b7e9b1fd110501e346f62b638cd7f7d5f30e5185690250e13013；42未核验空图及1个空栅格mask整训练图隔离，3432 train/val实际Dataset预检通过。1814训练负例为代理，不是人工真负。

## 失败历史，禁止默默复用

recognition_run01完成13轮，第14轮非有限梯度失败；新物种头不采用，存在性候选来源召回失败。detector_run01空栅格mask首轮失败，无完成epoch；通用初始化detector_run02完整3轮召回/计数失败，其source_snapshot保留，当前源码已变不能静默恢复。v3微调.5门槛虽然开发通过，来源损失较大，最终采用开发校准后的.3。

proposal_verification_binary_run01仍未恢复召回，VerifiedDetector仅研究复现，不接main。首个cascade_detector_finetune_det03_run01缓存固定.5，报告无效，profile移入该目录withdrawn_profile.json；修复与复现回归后完整run02才有效。builder拒绝缺少阈值感知helper身份的非默认门槛报告，拒绝将双模型复核报告当单模型发布。只读评审两轮未发现剩余P1/P2。

## 下一轮优化方向与交接

本轮主要实质改善是非蜱虫误报与计数，已落入分支默认；不能宣称未知物种或所有场景满意。后续优先审计已有archive2人工框，只接收训练来源组，补近景正例覆盖；其95图COCO含121框但segmentation为空，不能伪造矩形掩码或把漏标目标当负例。再研究未知物种/错误确认的独立训练与校准，并为混合图、小目标多蜱单列验收。已查看109/486/80不能反复伪装独立测试。任何新训练另建计划、冻结清单/来源与新实验目录，保留当前回滚点。

收尾提交只选本轮路径，禁止git add .。用户dirty .gitignore、.learnings/ERRORS.md和LEARNINGS.md、web/backend/data.db、开发进程/web开发.md、毕业设计/毕业设计系统规划.md、output/taskbook_revision_20261024/全部保留不提交。代码发布提交aa4b525c已成功推送origin/codex/practice-report-frontend（使用HTTP/1.1重试成功，含此前eb68cc47）；后续文档核对提交另行推送。release_manifest.json记录完整代码提交及源文件SHA。分支文档只维护本目录，原README/docs/论文材料是历史基线，不追加本轮结论。
