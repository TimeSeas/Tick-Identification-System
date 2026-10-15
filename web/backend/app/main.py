"""
FastAPI 入口：应用实例、lifespan（模型加载）、CORS、路由注册、全局异常处理
启动：cd web/backend && uvicorn app.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import ok, fail
from app.routers import analyze, history


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载全部模型（只加载一次，常驻内存），关闭时释放"""
    from app.services.detector import Detector
    from app.services.classifier import SpeciesClassifier
    from app.services.insect_classifier import InsectClassifier
    from app.services.pipeline import Pipeline
    from app.utils.cn_map import get_cn_map

    try:
        # 建表（表不存在才创建，已存在跳过）
        from app.database import Base, engine
        from app import models  # noqa: F401   import 以注册表定义
        Base.metadata.create_all(engine)

        detector = Detector(settings.detector_path, tile=settings.detect_tile)
        classifier = SpeciesClassifier(
            settings.species_model_path, settings.species_split_json,
            input_size=settings.species_input_size)
        insect = InsectClassifier(
            settings.insect_model_path, settings.insect_split_json)
        app.state.pipeline = Pipeline(#在前面创建蜱虫检测器和蜱虫分类器这两个实例后，传入Pipeline构造函数
            insect, detector, classifier, settings.thresholds_path, get_cn_map())
    except Exception as e:
        # 模型缺失/损坏时快速失败并给出明确提示（fail-fast，比运行中崩溃好排查）
        raise RuntimeError(f"模型加载失败: {e}") from e
    print("[启动] 模型加载完成，服务就绪")
    yield
    print("[关闭] 服务停止")


app = FastAPI(title="基于Mask R-CNN的蜱虫识别系统的设计与实现", version="0.1.0", lifespan=lifespan)

# CORS：允许前端开发服务器（Vite 默认 5173）跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 演示环境放开；生产环境应改成具体域名
    allow_methods=["*"],
    allow_headers=["*"],
)
#添加 CORS 中间件
#允许跨域请求，使前端（可能不同域名）能访问后端接口。
#allow_origins=["*"]：允许所有来源
#allow_methods=["*"]：允许所有 HTTP 方法
#allow_headers=["*"]：允许所有请求头。
app.include_router(analyze.router)
app.include_router(history.router)


@app.get("/healthz", tags=["运维"])
def healthz():
    """健康检查（Docker healthcheck / 负载均衡探活用）"""
    return ok()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """兜底异常处理：统一响应格式，不向前端暴露堆栈细节"""
    return fail(f"服务器内部错误: {type(exc).__name__}", 500)
