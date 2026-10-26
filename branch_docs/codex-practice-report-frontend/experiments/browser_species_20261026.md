# 浏览器随机蜱种核对（2026-10-26）

用户要求重启后端，通过浏览器从带蜱种文件夹名的数据集中随机抽图，核对网页种名与目录标签。本轮仅运行、观察与记录，没有修改模型、阈值或前端功能。

## 抽样和核验方法

从现有分组物种val的28个已支持种中均匀随机抽10种，每种随机抽2张，共20张。随机种子`2941406485`，名单在预测前冻结，未根据结果换图。原文件父目录名与清单标签逐项核对，测试前后核对原图SHA。

全部20张均由浏览器选择本地图片、点击开始识别、读取可见结果，并保存逐图完整截图。以页面请求编号关联独立测试库中的原始返回，20个页面结果与20条记录一一对应。档案页显示20条；模型/配置版本均与当前balanced版本一致。没有为统计另发推理请求。

## 结果与口径

|核对项|结果|
|---|---|
|检出蜱虫的图片|20/20|
|至少一个候选种名与文件夹一致|20/20|
|有正确已确认种名的图片|15/20|
|仅有待确认候选的图片|5/20|
|含错误已确认种名的图片|0/20|
|所有预测框的候选与文件夹一致|21/22|
|所有预测框状态|15已确认，7待确认|

20/20表示每图至少有一个匹配候选，不表示全部框正确或全部蜱种已确认。文件夹是图片级参考，缺少逐框物种及数量真值。这20张来自已经使用的开发数据，结果是网页抽查，不能写成独立泛化准确率。

## 逐图对照

|编号|文件夹标签|网页候选：分类置信度、状态|
|---|---|---|
|01|ixodes_vespertilionis|ixodes_vespertilionis 91.63%，已确认|
|02|amblyomma_hebraeum|amblyomma_hebraeum 94.26%，已确认；ixodes_persulcatus 18.68%，待确认|
|03|dermacentor_albipictus|dermacentor_albipictus 54.97%，已确认|
|04|ixodes_persulcatus|ixodes_persulcatus 56.02%，已确认|
|05|amblyomma_cajennense|amblyomma_cajennense 52.31%，待确认|
|06|ixodes_pacificus|ixodes_pacificus 93.43%，已确认|
|07|ixodes_scapularis|ixodes_scapularis 69.03%，已确认|
|08|amblyomma_cajennense|amblyomma_cajennense 32.80%，待确认|
|09|ixodes_vespertilionis|ixodes_vespertilionis 94.89%，已确认|
|10|amblyomma_hebraeum|amblyomma_hebraeum 84.12%，已确认；amblyomma_hebraeum 38.15%，待确认|
|11|ixodes_pacificus|ixodes_pacificus 55.71%，已确认|
|12|rhipicephalus_pulchellus|rhipicephalus_pulchellus 90.06%，已确认|
|13|ixodes_frontalis|ixodes_frontalis 98.15%，已确认|
|14|dermacentor_albipictus|dermacentor_albipictus 47.03%，待确认|
|15|haemaphysalis_longicornis|haemaphysalis_longicornis 95.06%，已确认|
|16|ixodes_scapularis|ixodes_scapularis 52.99%，待确认|
|17|rhipicephalus_pulchellus|rhipicephalus_pulchellus 45.93%，待确认|
|18|haemaphysalis_longicornis|haemaphysalis_longicornis 84.69%，已确认|
|19|ixodes_frontalis|ixodes_frontalis 94.36%，已确认|
|20|ixodes_persulcatus|ixodes_persulcatus 83.91%，已确认|

## 不一致案例

02号图片位于`amblyomma_hebraeum`文件夹，主框确认希伯来花蜱94.26%；额外15×15小框候选为全沟硬蜱`ixodes_persulcatus`18.68%，待确认。该框与文件夹标签不一致，截图显示靠近主体上方，疑似误检或局部重复目标，尚无逐框人工真值。保留原输出，不删除后重算。请求编号`e5f402c537ad`，记录ID2，截图`case_02.png`。后续应检查局部小框与重复候选，不以放宽确认门槛掩盖问题。

## 产物与版本

产物根为`output/optimization_20261015/rebuild_20261024/browser_species_20261026_run01/`，Git忽略：

- `manifest.json`：冻结抽样名单、随机种子、原图SHA和绝对路径；SHA256 `3941f8fb100f68b637fc57b978900b5d2651671e7659652e77fb1131aa0fd3b8`。
- `browser_ui_results.json`：20张图的可见页面文本；SHA256 `5f0a503a83c019465e0a29e7673a977a4d5956f647df27501a57cad2c80a5848`。
- `report.json`、`report.md`：逐框原始结果、汇总和截图索引；`case_01.png`至`case_20.png`全部通过图像校验。
- `history_20_records.png`、`history_case20_detail.png`：档案列表和详情实际截图。
- `browser_tests.db`、后端/前端日志、`services.json`：独立测试档案与服务记录。原数据库未操作，不提交SQLite、模型和图片。

代码基线`cd656f32`，模型代码`aa4b525c`，分支`codex/practice-report-frontend`。profile SHA256 `6164cdf41a781fa3fcc43ebc5990ad460a238b3ff4efc6bd167298fa8a2e86d0`；检测模型SHA256 `9e7a3a9b9cb32df87a37be71d85068f98a28d17bb61f828354534d4e95ed8b7c`。其他模型完整SHA在report.json。当前测试不修改上一轮release_manifest。

## 服务状态与历史隔离

用户继续时发现会话内启动的服务已退出；已改为Windows `Start-Process -WindowStyle Hidden`独立后台启动。后端8000 PID27396，前端5174 PID36224；实时状态以`services.json`和端口检查为准。网页`http://127.0.0.1:5174/`，档案页显示本轮20条，详情再次核对通过。

后端DB_URL指向本目录`browser_tests.db`，与原`web/backend/data.db`分开。恢复原历史时，在正常后端启动进程中不设置该DB_URL覆盖；保留两个数据库，勿相互覆盖。独立后台服务使用当前默认balanced模型。

浏览器自动化发生文件选择事件及选择器中断后，从已落盘名单和当前页面恢复，没有重新抽图，最终请求编号20个且唯一。此次没有功能代码变更，无需重跑训练或新增镜像功能的测试。
