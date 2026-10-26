"""
FastAPI 入口：应用实例、lifespan（模型加载）、CORS、路由注册、全局异常处理
启动：cd web/backend && uvicorn app.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager
import json
import hashlib
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas import ok, fail
from app.routers import analyze, history


def verify_artifact(path, expected_hash):
    """Calibration belongs to an exact model and split, not just a class count."""
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError('模型或数据划分与校准配置不一致，请重新评估并校准')


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

        with open(settings.inference_profile_path, encoding='utf-8') as f:
            profile = json.load(f)
        if profile['species']['input_size'] != settings.species_input_size:
            raise ValueError('蜱种推理尺寸与校准配置不一致')
        if profile.get('schema_version') != 1 or profile['insect']['input_size'] != 224:
            raise ValueError('推理校准配置版本或昆虫输入尺寸不受支持')
        for name, model_path, split_path in [
            ('species', settings.species_model_path, settings.species_split_json),
            ('insect', settings.insect_model_path, settings.insect_split_json),
        ]:
            verify_artifact(model_path, profile[name]['model_sha256'])
            verify_artifact(split_path, profile[name]['split_sha256'])
        detector = Detector(settings.detector_path, tile=settings.detect_tile)
        classifier = SpeciesClassifier(
            settings.species_model_path, settings.species_split_json,
            input_size=settings.species_input_size,
            view_mode=profile['species']['view_mode'])
        insect = InsectClassifier(
            settings.insect_model_path, settings.insect_split_json,
            view_mode=profile['insect']['view_mode'])
        for name, service in [('species', classifier), ('insect', insect)]:
            if set(service.classes) != set(profile[name]['thresholds']):
                raise ValueError(f'{name}校准类别与模型不一致')
        if set(classifier.classes) != set(profile['species']['region_thresholds']):
            raise ValueError('区域校准类别与蜱种模型不一致')
        if profile.get('recognition'):
            from app.services.recognition_classifier import RecognitionClassifier
            from app.services.recognition_pipeline import RecognitionPipeline
            recognition_profile = profile['recognition']
            verify_artifact(settings.recognition_model_path, recognition_profile['model_sha256'])
            verify_artifact(settings.detector_path, recognition_profile['detector_sha256'])
            recognition = RecognitionClassifier(settings.recognition_model_path)
            if (recognition.classes != classifier.classes or
                recognition.input_size != recognition_profile['input_size'] or
                recognition.metadata['split_sha256'] != recognition_profile['training_manifest_sha256'] or
                recognition_profile['species_mode'] != 'legacy'):
                raise ValueError('存在性复核模型、类别或训练清单与配置不一致')
            # Service consumes boxes only; avoid allocating full-resolution masks.
            detector.model.roi_heads.mask_roi_pool = None
            app.state.pipeline = RecognitionPipeline(insect, detector, recognition, profile,
                                                      get_cn_map(), species_classifier=classifier)
        else:
            app.state.pipeline = Pipeline(insect, detector, classifier, settings.thresholds_path,
                                          get_cn_map(), profile=profile)
        app.state.inference_version = dict(profile_sha256=hashlib.sha256(
            Path(settings.inference_profile_path).read_bytes()).hexdigest(),
            presence_model_sha256=profile.get('recognition', {}).get('model_sha256'))
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
    return JSONResponse(status_code=500, content=fail('服务器内部错误，请稍后重试', 500))
