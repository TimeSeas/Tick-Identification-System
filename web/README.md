# 基于Mask R-CNN的蜱虫识别系统的设计与实现 Web 应用

前后端分离 + 容器化部署。技术栈：FastAPI + SQLAlchemy（后端）/ Vue3 + TypeScript + Element Plus（前端）/ Docker Compose（部署）。

> 完整部署教程（装什么/怎么写/怎么跑/怎么维护/怎么用）：[Docker部署指南.md](Docker部署指南.md)

## 本地开发

```bash
# 后端（需先 conda activate tick_detection）
cd backend && uvicorn app.main:app --reload --port 8000
# 接口文档: http://localhost:8000/docs

# 前端（另开终端，Node >= 18）
cd frontend && npm install && npm run dev
# 页面: http://localhost:5173（/api 由 Vite 代理到 8000）
```

## Docker 一键部署

```bash
cd web
docker compose up --build
# 浏览器打开 http://localhost:8080
```

- `backend` 容器：uvicorn:8000，模型权重与数据集经只读卷挂载（`../output` → `/app/models`、`../data/tick_species` → `/app/data`，**注意挂载后容器内路径不包含 `tick_species` 这一层**），SQLite 数据持久化在 `dbdata` 卷；torch 预训练权重缓存挂载自 `~/.cache/torch`（避免容器启动时联网下载 100MB）
- `frontend` 容器：nginx:80 静态服务 + `/api` 反向代理到 backend
- 后端镜像装 **CPU 版 torch 2.5.1**（与本地 cu118 版同大版本，权重兼容）；推理速度约 0.3-1s/张，演示足够

## 设计说明（论文/答辩素材）

- **统一响应包** `{code, msg, data}`（code=200 成功），前端 axios 拦截器统一解包与报错
- **模型常驻内存**：FastAPI lifespan 加载一次，避免每次请求重复加载；耗时受设备与分支影响
- **分物种置信阈值**：Web 决策逻辑与论文阈值方法（Villon et al. 2020）同源——低置信度输出"待确认"而非硬判
- **已知取舍**：历史记录 result_json 内含 base64 标注图，演示规模下可接受；生产环境应分离文件存储（对象存储 + 数据库只存元数据）——论文可写为"工程优化方向"

## 目录

```
web/
├── backend/            # FastAPI：app/（main/config/schemas/models/database + services + routers + utils）
├── frontend/           # Vue3+TS：src/（api/types/views/router）
├── docker-compose.yml
└── README.md
```

## 当前界面与模型配置

识别工作台支持拖放、原图与标注图切换、数量概览、逐只结果及下载；档案页支持分页和详情回看。页面主线为识别，不展示旧版就医建议。

`SPECIES_INPUT_SIZE` 默认448；本地与Docker模型已统一，完整环境变量及本轮测试见 [实践报告与模型核验说明](../docs/practice_report_validation.md)。性能数字须区分整图、切块及完整请求，不以离线组件吞吐代表端到端时延。
