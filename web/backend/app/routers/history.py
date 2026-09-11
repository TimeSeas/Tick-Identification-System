"""
历史记录接口：
- GET /api/v1/history        分页列表（摘要，不含大字段）
- GET /api/v1/history/{id}   单条详情（含完整分析结果与标注图 base64）
"""
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import AnalysisRecord
from app.schemas import ok

router = APIRouter(prefix="/api/v1", tags=["历史"])


@router.get("/history")
def list_history(page: int = 1, size: int = 10, db: Session = Depends(get_db)):
    """分页列表：按时间倒序，只返回摘要字段（base64 大字段不进列表）"""
    total = db.query(AnalysisRecord).count()
    records = (db.query(AnalysisRecord)
               .order_by(AnalysisRecord.id.desc())
               .offset((page - 1) * size)
               .limit(size)
               .all())
    items = [{
        "id": r.id,
        "filename": r.filename,
        "request_id": r.request_id,
        "created_at": r.created_at.isoformat(),
        # 摘要：从 result_json 提取检测概况，避免列表传输大 JSON
        "detection_count": len(json.loads(r.result_json).get("detections", [])) if r.result_json else 0,
    } for r in records]
    return ok({"items": items, "total": total, "page": page, "size": size})


@router.get("/history/{record_id}")
def get_history(record_id: int, db: Session = Depends(get_db)):
    """单条详情：返回完整分析结果（含标注图 base64，供详情弹窗展示）"""
    r = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
    if r is None:
        raise HTTPException(status_code=404, detail="记录不存在")
    result = json.loads(r.result_json)
    result["record_id"] = r.id
    result["filename"] = r.filename
    result["created_at"] = r.created_at.isoformat()
    return ok(result)
