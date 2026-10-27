# 基于Mask R-CNN的蜱虫识别系统的设计与实现 Docker部署指南

## 当前Codex分支版本（2026-10-27）

当前蜱种专项仍在离线验证，默认保留上一版。8000暂用CPU后端PID36776，为唯一GPU评估让出显存；前端5174 PID36224不变。实际临时服务状态在output/optimization_20261015/species_20261027/services.json，旧browser目录的services.json仅是10月26日历史，不能照旧PID操作。离线结束再恢复CUDA。

当前本地Settings与Compose默认为：

- 检测：`output/optimization_20261015/rebuild_20261024/detector_v3_finetune_run01/frozen_detector.pth`，SHA256 `9e7a3a9b9cb32df87a37be71d85068f98a28d17bb61f828354534d4e95ed8b7c`。
- 配置：`configs/inference_detector_balanced.json`，SHA256 `6164cdf41a781fa3fcc43ebc5990ad460a238b3ff4efc6bd167298fa8a2e86d0`。检测门槛.3、1024/1200、IoU=.3；原28类448物种与昆虫模型/校准不变。
- 用户随后要求重启并做浏览器抽查，当前8000/5174已运行新默认，使用独立测试数据库；见下文。独立验证端口8016已停止。此前CUDA HTTP、历史/版本及JPEG验证通过，单图0.806秒；Docker仅同步配置，未构建/实跑。

完整效果及来源代价见[门槛核验](../experiments/detector_calibration_run01.md)。495开发验证的误报79→32、MAE .20745→.12766、召回87.91%→88.37%不能描述为独立测试。未知物种9/80错误确认仍未解决，部分来源检出有少量损失。

本机原模型/数据存在时，可沿用原启动命令；不要下载或用其他同名权重代替。模型不随Git推送，他机必须取得精确权重并校验SHA。以下历史章节中的原v3路径与原配置只代表此前版本。

浏览器抽查已完成10种20张，网页`http://127.0.0.1:5174/`，档案页有20条独立测试记录。后端8000 PID27396、前端5174 PID36224以Windows独立隐藏后台进程运行，实时信息在`output/optimization_20261015/rebuild_20261024/browser_species_20261026_run01/services.json`，进程号须现场核对。后端DB_URL只在该进程中指向此目录`browser_tests.db`，原数据库未操作。恢复原历史时，用没有该DB_URL覆盖的新进程正常启动后端，保留两个数据库。完整[浏览器核对记录](../experiments/browser_species_20261026.md)包含全部结果与不一致框。

本地回滚（在web/backend目录，重启服务）：

```powershell
$env:DETECTOR_PATH='../../output/det_v3/mask_rcnn_ticks_v3.pth'
$env:INFERENCE_PROFILE_PATH='../../configs/inference_optimization.json'
python -m uvicorn app.main:app --port 8000
```

恢复当前默认前，清除以上两项进程环境覆盖，再重启。Docker回滚需把Compose这两项环境变量改回`/app/models/det_v3/mask_rcnn_ticks_v3.pth`与`/app/config/inference_optimization.json`后重建容器；不要删除dbdata。`.5`候选配置`inference_detector_finetune.json`仅研究对照，不是当前默认；旧双头/逐框复核均未启用。

> 2026-09-10 · 从零讲清：装什么、写什么、怎么跑、怎么维护、怎么用
> 本文基于本项目实际文件（`web/docker-compose.yml` 等），与代码同步维护

## 1. 部署架构（先懂全局）

```
宿主机 Windows
├─ Docker Desktop（虚拟机里跑 Linux 容器）
│   ├─ web-backend 容器：uvicorn :8000
│   │    ├─ 只读挂载 ../output        → /app/models（模型权重/阈值表）
│   │    ├─ 只读挂载 ../data/tick_species → /app/data（划分文件/中文对照表）
│   │    ├─ 只读挂载 ~/.cache/torch   → /root/.cache/torch（预训练权重缓存）
│   │    └─ 命名卷 dbdata            → /app/db（SQLite 持久化）
│   └─ web-frontend 容器：nginx :80
│        ├─ 静态文件（Vue 构建产物）
│        └─ /api/* 反向代理 → backend:8000
└─ 浏览器访问 http://localhost:8080（宿主机 8080 → nginx 容器 80）
```

**三个卷为什么分开**：模型和数据只读挂载 = 换模型**不用重建镜像**；缓存挂载 = 启动**不联网下载**；dbdata 卷 = **删容器不丢历史数据**。

## 2. 要安装什么

| 依赖                                  | 说明                                                                       |
| ------------------------------------- | -------------------------------------------------------------------------- |
| Docker Desktop（Windows）             | 唯一需要装的东西。下载安装后**必须启动桌面应用**（守护进程随它运行） |
| WSL2（Docker Desktop 安装时自动引导） | Windows 跑 Linux 容器的底层，按安装向导默认即可                            |
| 后端/前端依赖                         | **不需要本机装**——全部在镜像构建时自动装进容器                     |

验证安装：`docker --version` 和 `docker info`（后者能通 = 守护进程活着）。

## 3. 要写什么文件、为什么写

| 文件                         | 作用                             | 为什么                                                                                          |
| ---------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------- |
| `backend/Dockerfile`       | 后端镜像构建脚本                 | 把 Python+依赖+代码打包成可搬运环境；先 COPY requirements 再装依赖 = 利用层缓存，改代码重建秒级 |
| `backend/requirements.txt` | 后端依赖清单（版本锁定）         | 镜像内安装依据，`==` 锁版保证可复现                                                           |
| `backend/.dockerignore`    | 构建排除清单                     | 防`__pycache__`/测试文件/数据库进镜像                                                         |
| `frontend/Dockerfile`      | 前端镜像（**多阶段构建**） | 第一段 Node 编译出 dist，第二段只留 nginx+静态文件——最终镜像 94MB，无 Node 无源码             |
| `frontend/nginx.conf`      | nginx 路由规则                   | 静态资源 + history 路由回落 + /api 反代到 backend                                               |
| `frontend/.dockerignore`   | 同上                             | 防 node_modules/dist 打进构建上下文                                                             |
| `docker-compose.yml`       | 编排文件                         | 一条命令拉起两容器+网络+卷+环境变量+健康检查+重启策略                                           |

## 4. 部署命令全流程

```bash
cd web

# ① 构建镜像（首次 5-15 分钟：拉基础镜像 + 装 torch；之后有缓存，秒级~分钟级）
docker compose build

# ② 启动（-d = 后台；自动建容器/网络/卷）
docker compose up -d

# ③ 查看状态（Up/healthy 即正常）
docker compose ps

# ④ 看日志（-f 持续跟踪）
docker compose logs -f backend

# ⑤ 停服务（保留容器与数据）
docker compose down
# 停服务并删除数据卷（慎用！历史记录清空）
docker compose down -v
```

**日常更新流程**：

- 改了**代码** → `docker compose up -d --build`（重建镜像并重启）
- 只换了**模型文件**（output/ 里替换）→ `docker compose restart backend`（挂载直接生效，不用重建）

## 5. Docker Desktop 怎么用

| 操作            | 方法                                                                             |
| --------------- | -------------------------------------------------------------------------------- |
| 启动            | 开始菜单打开 "Docker Desktop"，等左下角鲸鱼变绿色（守护进程就绪）                |
| Containers 面板 | 看容器状态/CPU 内存占用/点进去看实时日志，右上角按钮可 start/stop/restart/delete |
| Images 面板     | 看镜像列表与体积（web-frontend 94MB / web-backend 1.6GB）                        |
| Volumes 面板    | 看 dbdata 卷——删除它 = 清空历史数据库                                          |
| 终端            | 面板底部 "Terminal" 标签可直接进容器敲命令（`ls /app` 看容器内文件）           |

**常见现象**：电脑休眠/重启后 Docker Desktop 没自动启动 → 鲸鱼灰色 → 所有 docker 命令报 `cannot connect to the docker daemon`。**解决办法：手动打开 Docker Desktop**。

## 6. 维护手册（进程掉了怎么办）

排查三步走，按顺序：

```bash
# 第 1 步：看状态（什么挂了、挂在哪）
docker compose ps
# STATUS 列解读：Up = 正常；Restarting = 崩溃循环；unhealthy = 健康检查不过

# 第 2 步：看日志（90% 的问题日志里直接写了原因）
docker compose logs backend --tail 50

# 第 3 步：按症状处理（速查表）
```

| 症状                              | 原因                                   | 处理                                                            |
| --------------------------------- | -------------------------------------- | --------------------------------------------------------------- |
| 所有 docker 命令报 cannot connect | Docker Desktop 没开（休眠/重启后常见） | 打开 Docker Desktop，等变绿                                     |
| backend 反复 Restarting 或 exited | 启动崩溃（模型路径错/挂载错）          | `logs backend` 看报错 → 改 compose 环境变量 → `up -d`     |
| unhealthy                         | 健康检查连续失败                       | `logs backend` 看启动错误；`docker compose restart backend` |
| 端口被占用                        | 8000/8080 已被别的程序用               | 改 compose 里 ports（如`"8081:80"`）                          |
| 改代码后页面没变                  | 没重建镜像                             | `docker compose up -d --build`                                |
| 历史数据想清空                    | —                                     | `docker compose down -v` 再 `up -d`（**会清库**）     |
| 崩溃自动拉起不想要                | restart 策略生效中                     | `docker compose stop backend`（手动 stop 不触发重启策略）     |

**自动化保障**（已配置）：compose 里 `restart: unless-stopped` —— 容器崩溃或电脑重启后 Docker 会自动拉起它，无需手动干预。

## 7. 部署后怎么用

| 入口         | 地址                                                                |
| ------------ | ------------------------------------------------------------------- |
| 用户页面     | http://localhost:8080 （分析页 + 历史页）                           |
| 后端接口文档 | http://localhost:8000/docs （Swagger UI，可在线调试）               |
| 健康检查     | http://localhost:8000/healthz 或经 8080：`/api/...` 由 nginx 转发 |

**curl 快速验证**：

```bash
# 上传分析（走 nginx 全链路）
curl -X POST http://localhost:8080/api/v1/analyze -F "file=@图片路径.jpg"
# 查历史
curl "http://localhost:8080/api/v1/history?page=1&size=10"
```

**历史数据位置**：不在宿主机文件系统里，而在 Docker 的 `dbdata` 卷中（容器删了数据还在）。想看内容：`docker exec web-backend-1 sqlite3 /app/db/data.db ".tables"`（若镜像无 sqlite3 命令则用 Python：`docker exec web-backend-1 python -c "import sqlite3;print(sqlite3.connect('/app/db/data.db').execute('select count(*) from analysis_records').fetchone())"`）。

## 8. 本项目踩过的三个坑（部署经验记录）

1. **管道吞退出码**：`docker compose build | tail` 在构建失败时仍返回 0——判断构建成败直接看输出内容或镜像列表（`docker images`），不用管道包住构建命令
2. **卷挂载路径错层**：`../data/tick_species → /app/data` 挂载后，容器内文件在 `/app/data/split_clean.json`（没有 `tick_species` 这一层）——挂载是"目录内容平铺进去"，不是"带原目录名复制"
3. **容器内联网下载预训练权重**：torchvision 首次加载 `resnet50(weights=IMAGENET1K_V1)` 会联网下载 100MB，国内网络可能卡死启动——已用 `~/.cache/torch` 只读挂载复用宿主机缓存解决
4. **标注图中文乱码**：容器是 Linux，不存在宿主机 `C:/Windows/Fonts` 的字体，PIL 回退默认字体（无中文字形）→ 中文变方块——已在后端 Dockerfile 安装 `fonts-noto-cjk`，visualize.py 按"容器路径优先、Windows 路径兜底"顺序查找字体。**通用教训：任何依赖宿主机资源的路径（字体/模型/数据）进容器都要显式处理**

## 2026-10-15 配置校准

本优化分支新增只读挂载 `../configs:/app/config:ro`，通过 `INFERENCE_PROFILE_PATH=/app/config/inference_optimization.json` 读取双视图、上下文与校准阈值。本地默认读取 `../../configs/inference_optimization.json`。更新配置后需重启后端，输入尺寸或类别与模型不匹配会启动失败。分类与检测模型构建使用 `weights=None` / `weights_backbone=None` 后加载完整本地权重，不再依赖下载初始化权重。模型测试与系统级联测试的区别见 [优化评估](../history/2026-10-15/optimization_results.md)。

Compose与本地默认均使用v3检测和28类448蜱种分类，新增 `SPECIES_INPUT_SIZE=448`。昆虫粗分类使用 `/app/models/insect_classifier/model/insect_best.pth`，划分文件由 `../data/insects_split.json` 单文件只读挂载到 `/app/insects_split.json`。完整变量表与测试边界见 [实践报告与模型核验说明](../history/2026-10-15/practice_report_validation.md)。本轮未执行Docker镜像构建与容器端到端测试。


## 2026-10-26切换前的验证记录（历史）

本轮切换前，本地与Compose曾保留原 `configs/inference_optimization.json`、v3及原28类448蜱种/昆虫权重，避免仅依据开发val/cal过门槛就替换。当前默认已在本文开头列出。

`configs/inference_recognition_rebuild.json`是存在性复核的研究配置，109图同源诊断出现额外漏识别，**不推荐日常使用**。新物种头也未采用。可选加载器支持独立存在性权重、模型SHA及清单校验，并在分析结果记录推理版本；这些接入代码不代表候选已部署。检测run01之后因空mask失败，后续重训与微调另目录保留，不覆盖原权重。

2026-10-26已完成默认配置真实HTTP冒烟：独立CPU服务端口8016，数据库`output/optimization_20261015/rebuild_20261024/service_smoke_run01.db`。健康、错误MIME、损坏图片、超过10MB拒绝、真实图推理、历史列表/详情与model_version一致性通过；拒绝请求未入库，成功1条。耗时28.72秒是CPU服务数据，不能作为GPU网页性能指标。产物`service_smoke_run01/report.json`、`analysis.json`和`result.jpg`，日志`logs/service_smoke_run01.log`。未操作原数据库；候选模型HTTP及Docker尚未验证。

模型、数据及完整日志在本机ignored输出目录，不随代码推送；他机不能仅凭Git代码使用本次新权重。接手先读CONTEXT与实验登记，禁止按文件名猜测“best”已通过验收。

推理版本现完整记录detector/species/insect/presence模型SHA；如果实验profile含detector.model_sha256，启动时也强制核对检测权重。53项CPU回归通过。该变更后已用service_smoke_run02重复独立CPU HTTP验证：成功1图、3类拒绝、历史和完整版本一致，耗时18.40秒（CPU，不与run01作性能结论），数据库单独命名。验证服务已停止，不占用日常端口或原数据库。
