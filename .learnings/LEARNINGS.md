# Learnings

Corrections, insights, and knowledge gaps captured during development.

**Categories**: correction | insight | knowledge_gap | best_practice

---

## [LRN-20260912-001] best_practice

**Logged**: 2026-09-12T08:30:00Z
**Priority**: high
**Status**: promoted
**Area**: backend

### Summary
torchvision 的 GeneralizedRCNNTransform 的 min_size/max_size 不随 state_dict 保存，加载权重后必须手动恢复训练分辨率，否则 train-serve skew

### Details
- v2/v3 检测模型按 min_size=(1024,)、max_size=1200 训练，但 `load_state_dict` 后重建模型的 transform 回到默认 (800,)/1333
- 导致：评估脚本（evaluate_model.py）与 Web 推理（detector.py）都在 800 分辨率下跑 1024 训练的模型；此前 v2 报告的 mAP 0.857 是 800 分辨率测的，按训练配置重测为 0.845
- 检查方法：`any("transform" in k for k in state_dict.keys())` → False 即确认不保存
- 修复：`model.transform.min_size = (1024,); model.transform.max_size = 1200`（load_state_dict 之后）

### Suggested Action
所有"重建模型再加载权重"的推理/评估代码，先打印 model.transform.min_size 对比训练配置；evaluate_model.py 尚未修复

### Metadata
- Source: error
- Related Files: web/backend/app/services/detector.py, src/eval/evaluate_model.py, 论文研究材料/研究成果与论文素材.md §2.1
- Tags: torchvision, state_dict, train-serve-skew, min_size
- Recurrence-Count: 1
- First-Seen: 2026-09-12
- Last-Seen: 2026-09-12

**Promoted**: 项目记忆 project-current-state.md + 论文素材 §2.1

---

## [LRN-20260912-002] correction

**Logged**: 2026-09-12T08:40:00Z
**Priority**: high
**Status**: resolved
**Area**: backend

### Summary
切块推理（tile）对"小目标增强"有效但对大目标有害；级联顺序"粗分类优先"会把真蜱虫误判成昆虫——两个方向均被用户实测证伪后改为"检测优先级联"

### Details
- 用户实测反馈：①是蜱虫的图被报成其他昆虫（粗分类第一闸误判，检测没机会跑）；②大像素蜱虫被 tile=2 切碎无法识别
- 最终架构：全图检测(1024) → 未检出才切块补偿 → 仍未检出才粗分类兜底
- 三层误检防御（实测驱动）：大框否决（框面积≥5% + 全图粗分类非蛛形纲≥90%）、第二意见复判（uncertain 时对 crop 交叉验证）、螨类否决（mite/spider_mite≥80%）
- 实测：蚜虫群会被检测模型高分误检（score 0.85，框占比 6.3%/49.5%），但全图粗分类判 aphid 92-97%；真蜱虫 Top15 大框图全图粗分类 0 误伤
- 用户的验收标准（医疗场景）：绝不把肉眼看是蜱虫的报成其他昆虫；大像素蜱虫必须高精度；小目标允许犯错

### Suggested Action
级联系统设计原则：判别力最强的模型放第一道闸；否决规则只在 ≥80-90% 高置信触发；蛛形纲（tick/mite/spider）互不否决

### Metadata
- Source: user_feedback
- Related Files: web/backend/app/services/pipeline.py
- Tags: cascade, false-positive, multi-model-voting, tiling
- See Also: LRN-20260912-001
- Recurrence-Count: 1
- First-Seen: 2026-09-12
- Last-Seen: 2026-09-12

### Resolution
- **Resolved**: 2026-09-12T08:40:00Z
- **Notes**: pipeline.py 改版完成，8 图回归全过（真蜱虫大框 73.8%/普通/小目标/蚜虫×2/蓟马/叶螨/菜青虫）

---

## [LRN-20260912-004] best_practice

**Logged**: 2026-09-12T10:00:00Z
**Priority**: medium
**Status**: resolved
**Area**: backend

### Summary
Mask R-CNN 对同一大目标会重复检出嵌套框（实测重复框 IoU 0.476），全图推理路径也必须做重叠去重；去重阈值要用实测数据定，默认 0.5 拦不住嵌套框

### Details
- 现象：73.8% 大蜱虫图检出 [105,46,540,533](0.98) + [172,163,461,512](0.61)，后者完全嵌套在前者内，IoU=0.476
- 原实现只有切块路径有去重（IoU>0.5），全图路径裸返回模型输出 → 用户看到重叠标注框
- 修复：detector.py 增加模块级 `_dedup()`（贪心按分降序，IoU>DEDUP_IOU=0.3 视为同一只），全图/切块两分支统一走
- 0.3 的安全依据：相邻两只真蜱虫的框 IoU 通常 <0.1

### Suggested Action
检测类模型输出到展示层之前，任何推理路径（含全图）都要过一遍 IoU 去重；阈值取"实测重复框 IoU 的一半余量"

### Metadata
- Source: user_feedback
- Related Files: web/backend/app/services/detector.py
- Tags: nms, dedup, mask-rcnn, duplicate-detection
- See Also: LRN-20260912-002
- Recurrence-Count: 1
- First-Seen: 2026-09-12
- Last-Seen: 2026-09-12

### Resolution
- **Resolved**: 2026-09-12T10:00:00Z
- **Notes**: 5 图回归通过；原重复图从 2 框变 1 框，其余 case 无回归

---

## [LRN-20260912-003] knowledge_gap

**Logged**: 2026-09-12T09:00:00Z
**Priority**: medium
**Status**: pending
**Area**: infra

### Summary
Windows 本机 shell 两个坑：conda run 不支持 -c 参数内换行；Git Bash 的 /tmp 路径在 python -c 字符串里不会被 MSYS 转换

### Details
- `conda run -n env python -c "多行脚本"` → AssertionError（arguments contain newlines），还会卡在交互式 error-report 提示上；解法：写成 .py 文件再 `conda run`，或直接用环境绝对路径 `C:/Users/a/.conda/envs/<env>/python.exe`（最快）
- Git Bash 里 curl -o /tmp/x.json 会被 MSYS 翻译成 Windows 路径，但 python -c "open('/tmp/x.json')" 字符串内的路径不翻译 → FileNotFoundError；解法：脚本内用 `C:/Users/a/AppData/Local/Temp/` 等 Windows 原生路径

### Suggested Action
本机跑 python 一律用 `C:/Users/a/.conda/envs/tick_detection/python.exe` 绝对路径 + 临时 .py 文件，避免 conda run 和 -c 内嵌脚本

### Metadata
- Source: error
- Related Files: 无
- Tags: windows, git-bash, conda, msys, path-translation
- Recurrence-Count: 2
- First-Seen: 2026-09-12
- Last-Seen: 2026-09-12

---
