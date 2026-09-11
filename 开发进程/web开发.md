Context

毕设 WP6 Web 应用，用户要求：技术栈贴合企业要求、体现全栈工程能力。已确认四项决策：
1. 前端 Vue3 + TypeScript + Vite + Element Plus（国内企业主流，答辩可讲选型理由）
2. 后端 SQLite + SQLAlchemy 推理历史记录（数据库能力加分）
3. Docker + docker-compose 部署（nginx 静态前端 + uvicorn 后端）
4. 先上 M1+M2（检测→蜱种→风险建议闭环已完整），M3 伤口分割预留接口热插

环境已就绪：Node v22.14 / npm 11.14 / Docker 29.5；后端依赖全部在现有 tick_detection conda 环境（torch/torchvision/opencv 等）。

自主完成原则（用户已确认"让我自己来上手，学习如何搭建系统"）：全部功能代码由用户亲手写——这是学习目标本身。Claude 的角色：本设计为蓝图 + 每个里程碑开工前讲清"这一步做什么、涉及哪些概念、怎么写" + 代码评审 + 卡住时答疑。每个里程碑交付后 Claude 评审再进入下一个。

系统架构

浏览器 (Vue3+TS+Element Plus)
   │  POST /api/v1/analyze (multipart 图片)
   ▼
Nginx（前端静态资源 + /api 反向代理，Docker 容器）
   ▼
FastAPI (uvicorn)
   ├─ Pipeline 级联：M1 蜱虫检测（2类 Mask R-CNN）
   │     → 检出蜱虫框 → 裁剪 ROI → M2 蜱种分类（31 类 ResNet-50）
   │     → 分物种置信阈值决策（confirmed / uncertain）
   │     → 风险规则引擎 → 就医建议文案
   ├─ M3 伤口分割服务接口（预留，模型训练后热插）
   └─ SQLite 历史记录（AnalysisRecord 表）

复用现有资产（不重造轮子）

┌────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────┐
│                    资产                    │                                用途                                │
├────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ detect.py 的 get_model()/加载逻辑          │ M1 检测模型封装（output/mask_rcnn_ticks.pth）                      │
├────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ src/inference/infer_tick_species.py        │ M2 分类封装（output/tick_species_31_weighted/model/tick_species_be │
│ 的加载/预处理/中文映射逻辑                 │ st.pth）                                                           │
├────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ output/tick_species_31_weighted/thresholds │ 分物种阈值 → uncertain                                             │
│ .csv                                       │ 决策（低置信度输出"无法确定，建议拍照重试/人工鉴定"）              │
├────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ data/tick_species/物种中文名对照表.csv     │ slug → 中文名展示                                                  │
├────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ 对照表"备注"列的医学信息                   │ 风险规则文案素材                                                   │
└────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────┘

后端设计（web/backend/）

app/
├── main.py            # FastAPI 实例、CORS、路由注册、lifespan 中加载模型（常驻内存）
├── config.py          # pydantic-settings：模型路径/阈值表/中文表/数据库 URL，环境变量可覆盖
├── schemas.py         # Pydantic v2 请求/响应模型（与前端 TS 类型一一对应）
├── database.py        # SQLAlchemy engine + SessionLocal
├── models.py          # ORM: AnalysisRecord(id, filename, created_at, result_json)
├── services/
│   ├── detector.py    # M1 封装：加载/推理/输出 boxes+scores
│   ├── classifier.py  # M2 封装：224 预处理 + 31 类 softmax
│   ├── pipeline.py    # 级联编排：检测→裁剪→分类→阈值→汇总
│   ├── risk_rules.py  # 物种→风险文案映射表（含 uncertain/未检出分支）
│   └── bite_seg.py    # M3 预留接口（先返回 None，WP3 后热插）
├── routers/
│   ├── analyze.py     # POST /api/v1/analyze
│   └── history.py     # GET /api/v1/history（分页）、GET /api/v1/history/{id}
├── utils/
│   ├── visualize.py   # 检测框+中文标签绘制（返回 base64 结果图）
│   └── cn_map.py      # 对照表加载缓存
tests/test_analyze.py  # pytest：pipeline 单测（真实小图）+ 阈值分支 + 历史接口
requirements.txt / Dockerfile

API 约定（企业规范）：
- 统一响应包 {code, msg, data}（code=0 成功）；全局异常处理中间件；结构化日志
- POST /api/v1/analyze 返回：{request_id, detections: [{bbox, score, species_slug, species_cn, confidence, status}], result_image_base64, risk_advice}；status ∈ {confirmed, uncertain}；未检出蜱虫返回空 detections + 提示文案
- GET /healthz 健康检查（Docker 用）

风险规则要点（risk_rules.py，措辞"辅助参考，不构成医疗建议"）：
- ixodes_scapularis→莱姆病提示；ixodes_persulcatus→森林脑炎；haemaphysalis_longicornis→SFTS；rhipicephalus_sanguineus→犬埃立克体病；dermacentor_*→蜱传斑疹热；unknown/低阈值→通用建议（保存虫体、观察伤口、及时就医）

前端设计（web/frontend/，Vue3+TS+Vite+Element Plus+Pinia）

src/
├── api/client.ts          # axios 实例 + 统一响应拦截
├── types/index.ts         # 与后端 schemas.py 对齐的 TS 接口
├── views/AnalyzeView.vue  # 主页面：拖拽上传 → 进度 → 结果卡片（检测框列表+中文蜱种+风险建议+标注图）
├── views/HistoryView.vue  # 历史列表（分页表格）+ 详情弹窗
├── components/ResultCard.vue / DetectionItem.vue
├── stores/history.ts      # Pinia 状态
└── router/ + App.vue
Dockerfile（多阶段：node build → nginx 静态 + /api 反代）

根部署层

web/docker-compose.yml   # backend(uvicorn) + frontend(nginx) 两容器
web/README.md            # 启动/构建说明（答辩运维文档）

实施里程碑（用户自主开发，Claude 评审；预计 8-10 天）

┌────────────────┬──────────────────────────────────────────────┬────────────────────────────────────────────┐
│     里程碑     │                     内容                     │                    验证                    │
├────────────────┼──────────────────────────────────────────────┼────────────────────────────────────────────┤
│ M1 后端骨架    │ main/config/schemas/database/健康检查        │ curl /healthz                              │
├────────────────┼──────────────────────────────────────────────┼────────────────────────────────────────────┤
│ M2 模型服务    │ detector/classifier/pipeline/风险规则/可视化 │ curl 上传测试图，JSON 结构正确、中文名正确 │
├────────────────┼──────────────────────────────────────────────┼────────────────────────────────────────────┤
│ M3 数据库      │ ORM + 历史接口                               │ curl 历史分页返回记录                      │
├────────────────┼──────────────────────────────────────────────┼────────────────────────────────────────────┤
│ M4 前端主流程  │ Vite 脚手架 + 上传页 + 结果卡片              │ 浏览器上传→显示结果                        │
├────────────────┼──────────────────────────────────────────────┼────────────────────────────────────────────┤
│ M5 历史页+联调 │ HistoryView + 全链路联调                     │ 前后端数据一致                             │
├────────────────┼──────────────────────────────────────────────┼────────────────────────────────────────────┤
│ M6 Docker      │ 两容器 compose + README                      │ docker compose up 全链路可用               │
└────────────────┴──────────────────────────────────────────────┴────────────────────────────────────────────┘

验证方式

1. 后端：pytest web/backend/tests（pipeline 用主数据集测试图验证：有蜱图返回 detections+中文名；昆虫图返回未检出分支；低置信度返回 uncertain 分支）
2. API：curl 上传三张代表图（蜱虫图/昆虫图/模糊图）逐分支验证
3. 前端：dev 模式浏览器手测上传/历史/错误提示；TS 严格模式零错误（vue-tsc）
4. 部署：docker compose up --build 后浏览器完整走一遍
5. 答辩演练：录屏"上传→结果→历史"全流程

待办依赖提示

- 31 类消融（448/无加权）完成后，Web 的 M2 模型路径换成最优者（config 环境变量切换，无需改代码）
- WP3 伤口分割训练完成后在 bite_seg.py 填入实现（接口已预留）










一步一步来。先建目录结构，再逐个文件讲。

第 0 步：建目录 + 装依赖

mkdir -p web/backend/app/{services,routers,utils} web/backend/tests
cd web/backend
# 用现有 conda 环境即可（torch 那些都在里面），补装 Web 依赖
conda activate tick_detection
pip install "fastapi" "uvicorn[standard]" python-multipart sqlalchemy pydantic-settings

▎ 装好后 pip freeze 里挑这几个包写进 requirements.txt（Docker 里要用）。

第 1 步：写 app/config.py（配置中心）

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 每个配置项 = 一个属性，环境变量同名可覆盖（企业 12-factor 规范）
    db_url: str = "sqlite:///./data.db"          # 数据库连接串
    detector_path: str = "../../output/mask_rcnn_ticks.pth"   # M1 模型（M2 才用到）
    species_model_path: str = "../../output/tick_species_31_weighted/model/tick_species_best.pth"
    thresholds_path: str = "../../output/tick_species_31_weighted/thresholds.csv"
    cn_table_path: str = "../../data/tick_species/物种中文名对照表.csv"

settings = Settings()   # 全局单例，其他模块 from app.config import settings

理解点：BaseSettings 会自动读环境变量——以后 Docker 里 DETECTOR_PATH=/app/models/xxx 就能覆盖，不改代码。

第 2 步：写 app/database.py（数据库引擎）

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config import settings

engine = create_engine(settings.db_url, connect_args={"check_same_thread": False})  # SQLite 特殊参数
SessionLocal = sessionmaker(bind=engine, autoflush=False)

class Base(DeclarativeBase):   # 所有 ORM 模型的父类（M3 建表用）
    pass

第 3 步：写 app/schemas.py（统一响应模型）

from typing import Optional
from pydantic import BaseModel

class ApiResponse(BaseModel):
    code: int = 0
    msg: str = "ok"
    data: Optional[dict] = None

def ok(data=None) -> dict:  return ApiResponse(data=data).model_dump()
def fail(msg: str, code: int = 500) -> dict:  return ApiResponse(code=code, msg=msg).model_dump()

理解点：ok()/fail() 是工具函数——以后所有接口 return ok({...})，格式永远统一，前端只认这一种。

第 4 步：写 app/main.py（入口，lifespan 先留空）

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import ok, fail

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("启动：模型将在 M2 里程碑加载到这里")
    yield          # yield 之前=启动时执行；之后=关闭时执行
    print("关闭")

app = FastAPI(title="蜱虫识别系统", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/healthz")
def healthz():
    return ok()

@app.exception_handler(Exception)   # 兜底：任何未捕获异常都变成统一格式，不暴露堆栈
async def global_exc(request, exc):
    return fail(f"服务器内部错误: {type(exc).__name__}")

第 5 步：启动验证

cd web/backend
uvicorn app.main:app --reload --port 8000

- 终端出现 Uvicorn running on http://127.0.0.1:8000 = 成功
- 浏览器开 http://127.0.0.1:8000/docs — 这是 FastAPI 自动生成的接口文档，点 /healthz 里的 "Try it out → Execute" 能看到返回
- 改任意代码保存，--reload 自动重启

第 6 步：别忘了 app/__init__.py

web/backend/app/ 里放一个空的 __init__.py（让它成为 Python 包，from app.config import ... 才能工作）。









进入 M2 教学：模型服务封装。核心原则一句话——模型只在启动时加载一次，常驻内存；每个请求只做推理，毫秒级返回（每次请求都加载模型的话要等 10 秒，Web 系统不可用）。

涉及的概念

┌──────────────┬──────────────────────────────────────────────────────────────────────────────┐
│     概念     │                                     理解                                     │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┤
│ 模型常驻内存 │ lifespan 里加载 → 挂到 app.state → 路由里取用。加载 1 次，推理 N 次          │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┤
│ 服务层封装   │ 每个模型一个 service 类，路由不碰模型细节——答辩可讲"分层解耦"                │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┤
│ 级联编排     │ pipeline 是"总导演"：检测 → 裁剪 → 分类 → 阈值 → 风险规则                    │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┤
│ 阈值决策     │ 用 M2 产出的 thresholds.csv：置信度 ≥ 该物种阈值 → confirmed；否则 uncertain │
└──────────────┴──────────────────────────────────────────────────────────────────────────────┘

要写的文件（复用你仓库里现成的代码）

1. services/detector.py — M1 封装

复用来源：detect.py 的 get_model()（就是 FastRCNNPredictor 换头的 2 类 Mask R-CNN）

class Detector:
    def __init__(self, path):  # 加载模型（照抄 detect.py 的 get_model + torch.load(weights_only=True)）
        ...
    def detect(self, img_rgb) -> list[dict]:
        # 输入 np.ndarray(H,W,3)，输出 [{"bbox": [x1,y1,x2,y2], "score": 0.93}, ...]
        # 用 @torch.no_grad()、model.eval()、score 过滤 0.5（与 detect.py 同口径）
关键点：detect.py 里模型输入是 tensor.unsqueeze(0) 到 device——把它抽成一个方法就是你的 service。

2. services/classifier.py — M2 封装

复用来源：src/inference/infer_tick_species.py 的加载逻辑 + eval 预处理 + 类别读取

class SpeciesClassifier:
    def __init__(self, model_path, split_json):
        # resnet50 → 换 FC → load_state_dict；classes 从 split JSON 读出（与训练脚本同款 sorted({...})）
        self.eval_tf = ...   # Resize(256)→CenterCrop(224)→ToTensor→Normalize，与训练一致！
    def classify(self, crop_rgb) -> tuple[str, float]:
        # 返回 (slug, prob)；用 @torch.no_grad()
关键点：预处理必须和训练时 eval_tf 一模一样（Resize 256 → CenterCrop 224），否则精度掉。

3. services/pipeline.py — 总导演

class Pipeline:
    def __init__(self, detector, classifier, thresholds, cn_map): ...
    def analyze(self, image_bytes) -> dict:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        dets = self.detector.detect(np.array(img))
        if not dets:
            return {"detections": [], "risk_advice": 未检出文案, ...}
        for d in dets:
            crop = img.crop(d["bbox"])            # PIL 裁剪，自动处理出界
            slug, prob = self.classifier.classify(np.array(crop))
            thr = self.thresholds.get(slug, 90.0) # 读 thresholds.csv 该物种的"MC<2% 阈值"列
            d["species_slug"] = slug; d["species_cn"] = self.cn_map.get(slug, slug)
            d["confidence"] = prob; d["status"] = "confirmed" if prob >= thr else "uncertain"
        risk_advice = 汇总规则（见下）
关键点：阈值逻辑——分物种阈值正是你的论文创新点，这里用上它，Web 系统与论文实验连成一体。

4. services/risk_rules.py — 风险规则

一个 dict 映射 + 汇总函数：ixodes_scapularis → "疑似黑腿蜱，莱姆病风险，建议就医并保存虫体"、haemaphysalis_longicornis → SFTS 提示……（素材在对照表"备注"列）；uncertain → "无法确定蜱种，建议人工鉴定"；末尾统一加**"本结果仅供参考，不构成医疗建议"**。

5. utils/visualize.py — 画标注图

复用 infer_tick_species.py 的画图方式：PIL ImageDraw + 黑体字体，画框 + 顶部中文标签（"长角血蜱 92.3%"），返回 base64 字符串（前端直接 <img src="data:image/jpeg;base64,..."> 显示）。

6. utils/cn_map.py — 中文映射

读 物种中文名对照表.csv → {slug: 中文名}，加载一次缓存。

7. 接线到路由 routers/analyze.py

@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    # 校验：content_type 是图片 + 大小 < 10MB，不合法 return fail("...", 400)
    result = app.state.pipeline.analyze(data)   # pipeline 在 lifespan 里构建好
    return ok(result)
main.py 的 lifespan 里：app.state.pipeline = Pipeline(Detector(...), SpeciesClassifier(...), ...)。

验收标准（M2 完成）

# 用主数据集测试图：
curl -X POST http://localhost:8000/api/v1/analyze -F "file=@ticks image detection.v5i.coco/test/某个蜱虫图.jpg"
# → detections 非空、species_cn 是中文、status 正确
# 用昆虫图（archive3/farm_insects/Aphids/任意一张）：
# → detections 空 + 未检出文案













前端为什么是"工程"

一个 Vue3 项目不只是写页面，标准的工程结构包含：类型系统（TS 接口与后端 schema 对齐）、API 层（axios 封装）、路由/状态（Vue Router/Pinia）、构建与代理（Vite）。你搭完这套，简历上就能写"独立搭建 Vue3+TS 前后端分离项目"。

第 1 步：脚手架（3 条命令）

cd web
npm create vite@latest frontend -- --template vue-ts    # 官方脚手架，选 vue-ts 模板
cd frontend
npm install
npm install element-plus axios vue-router@4
npm run dev        # 浏览器开 http://localhost:5173，看到 Vite 欢迎页 = 成功

第 2 步：Vite 代理（前后端分离的关键配置）

改 vite.config.ts，加：

server: {
  proxy: {
    '/api': 'http://localhost:8000'   // 前端所有 /api 请求转发到后端
  }
}

理解点（答辩必讲）：开发时前端 5173、后端 8000，浏览器直连后端就是跨域；配置代理后 /api 请求由 Vite 服务器转发，浏览器视角是同源的——这是企业前后端分离开发的标准姿势（生产环境 nginx 做同一件事，M6 你会看到）。

第 3 步：要写的文件（src/ 下）

┌──────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────┐
│     文件     │                                               内容                                               │
├──────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ types/index. │ TS 接口，与后端 schemas.py 对齐：ApiResponse<T>{code,msg,data}、Detection{bbox,score,species_cn, │
│ ts           │ confidence,status}、AnalyzeResult{detections,risk_advice,result_image_base64,request_id}         │
├──────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ api/client.t │ axios 实例：baseURL:'/api/v1' + 响应拦截器（code!==0 时弹错误提示并                              │
│ s            │ reject）——统一响应包的价值在这里兑现                                                             │
├──────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ router/index │ 两个路由：/（分析页）、/history（历史页，M5 再做）                                               │
│ .ts          │                                                                                                  │
├──────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ views/Analyz │ 主页面：el-upload 选图 + el-button 上传 → 调 /analyze → 显示标注图 + 检测卡片列表 + 风险建议     │
│ eView.vue    │                                                                                                  │
├──────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ main.ts      │ 注册 Element Plus、router、Pinia                                                                 │
├──────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ App.vue      │ 顶部导航 + <router-view>                                                                         │
└──────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────┘

第 4 步：AnalyzeView 的雏形逻辑（你先写，框架如下）

vue
<script setup lang="ts">
import { ref } from 'vue'
import { analyzeImage } from '../api/client'
import type { AnalyzeResult } from '../types'

const loading = ref(false)
const result = ref<AnalyzeResult | null>(null)

async function onUpload(file: File) {
  loading.value = true
  result.value = await analyzeImage(file)   // api 层封装 FormData
  loading.value = false
}
</script>

<template>
  <el-upload :auto-upload="false" :on-change="..." accept="image/*">
    <el-button>选择图片</el-button>
  </el-upload>
  <el-button type="primary" @click="上传" :loading="loading">开始分析</el-button>
  <img v-if="result" :src="result.result_image_base64" style="max-width: 100%" />
  <!-- 检测列表 + 风险建议卡片 -->
</template>

理解点：ref 是 Vue3 响应式核心（值变了页面自动更新）；result_image_base64 直接塞进 <img :src> 就能显示——后端返回 base64 就是为这个。

验收标准（M4 完成）

浏览器上传蜱虫图 → 看到标注图（框+中文标签）+ 检测列表 + 风险文案；上传昆虫图 → 看到"未检测到蜱虫"灰条。期间打开 F12 的 Network 面板看请求——能看到 /api/v1/analyze 的完整请求响应，这是前端调试基本功。









本文件引入的新知识点

┌──────────────────────┬──────────────────────────────┬──────────────────────────────────────────────────────────┐
│         写法         │             作用             │                       为什么这样写                       │
├──────────────────────┼──────────────────────────────┼──────────────────────────────────────────────────────────┤
│                      │ 派生值：依赖别的数据自动算出 │ previewSrc 依赖                                          │
│ computed(() => ...)  │ 来的值                       │ viewMode+result，任一变化它自动重算——你自己写 if/else    │
│                      │                              │ 更新它容易漏                                             │
├──────────────────────┼──────────────────────────────┼──────────────────────────────────────────────────────────┤
│ URL.createObjectURL( │ 本地文件生成临时预览地址     │ 选了图还没上传就能立刻看到原图，体验好；用完             │
│ file)                │                              │ revokeObjectURL 释放内存（防泄漏细节）                   │
├──────────────────────┼──────────────────────────────┼──────────────────────────────────────────────────────────┤
│ <el-button tag="a"   │ 按钮伪装成下载链接           │ base64 数据 URI 直接作为                                 │
│ :download>           │                              │ href，浏览器点击即下载——不用走后端                       │
├──────────────────────┼──────────────────────────────┼──────────────────────────────────────────────────────────┤
│ #default="{ row }"   │ el-table                     │ 置信度要格式化（toFixed(1)）、状态要变色标签——默认列只会 │
│                      │ 的插槽：自定义单元格渲染     │ 显示原始文本                                             │
├──────────────────────┼──────────────────────────────┼──────────────────────────────────────────────────────────┤
│ result!.risk_advice  │ 模板里的非空断言             │ v-if="hasResult" 只拦元素不拦类型检查，! 告诉            │
│                      │                              │ TS"我保证不为空"                                         │
└──────────────────────┴──────────────────────────────┴──────────────────────────────────────────────────────────┘

三项功能的实现方式

1. 原图/标注图切换：el-radio-button 双向绑定 viewMode；previewSrc 用 computed 在两源间切换（原图=本地 URL，标注图=后端 base64）；出结果自动跳到标注图
2. 下载标注图：tag="a" 让按钮渲染成 <a> 标签 + download 属性，文件名取 标注结果_原文件名
3. 识别结果列表：el-table 七列——序号/中文名/学名/检测置信度/分类置信度/状态标签/bbox 位置；detectionRows 用 computed 把结果整理成表格行（含 bbox 文本格式化）