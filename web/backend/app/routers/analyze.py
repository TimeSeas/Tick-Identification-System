"""
图片分析接口：POST /api/v1/analyze
- 接收 multipart 图片上传，校验类型与大小
- 调 pipeline 级联推理，生成可视化结果图（base64）
- M3 里程碑将在这里追加历史记录入库
"""
import json
import uuid
from threading import Lock

from fastapi import APIRouter, Depends, File, Request, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import AnalysisRecord
from app.schemas import ok, fail
from app.utils import visualize
from app.utils.images import ImageValidationError, load_rgb_image

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB 上传上限
ALLOWED_TYPES = ("image/jpeg", "image/png", "image/bmp", "image/webp")

router = APIRouter(prefix="/api/v1", tags=["分析"])
# CUDA models are shared by worker threads; serialize inference per process.
INFERENCE_LOCK = Lock()


@router.post("/analyze")
def analyze(request: Request, file: UploadFile = File(...),
                  db: Session = Depends(get_db)):
    # 1. 上传校验：类型与大小，不合法直接拒绝（企业规范：入口处拦截，不污染推理链路）
    if file.content_type not in ALLOWED_TYPES:
        return fail(f"不支持的图片类型: {file.content_type}", 400)
    data = file.file.read(MAX_FILE_SIZE + 1)
    if len(data) > MAX_FILE_SIZE:
        return fail("图片超过 10MB 限制", 400)

    # 2. 级联推理（pipeline 在 lifespan 里构建好，常驻 app.state）
    try:
        with INFERENCE_LOCK:
            load_rgb_image(data)
            result = request.app.state.pipeline.analyze(data)
    except ImageValidationError as error:
        return fail(str(error), 400)

    # 3. 生成标注可视化图（base64，前端可直接渲染）
    insect = result.get("insect")
    top_label = None
    if not result["detections"] and insect and insect.get("slug") != "tick":
        top_label = f"未检出蜱虫，{'疑似' if insect.get('status') == 'uncertain' else '候选为'}{insect['cn']}"
    result_img_b64 = visualize.draw_result(data, result["detections"], top_label)
    result["result_image_base64"] = result_img_b64
    result["request_id"] = uuid.uuid4().hex[:12]

    # 4. 历史记录入库（完整结果序列化存储；request_id 已在上面生成）
    record = AnalysisRecord(
        filename=file.filename or "unknown",
        request_id=result["request_id"],
        result_json=json.dumps(result, ensure_ascii=False),
    )
    db.add(record)
    db.commit()

    return ok(result)
