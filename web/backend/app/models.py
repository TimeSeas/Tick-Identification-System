"""ORM 模型：数据库表结构定义（SQLAlchemy 声明式）"""
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.database import Base


class AnalysisRecord(Base):
    """一次图片分析的历史记录"""
    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255))            # 上传的原文件名
    request_id = Column(String(32), index=True)  # 请求唯一 ID（前端展示/追溯用）
    created_at = Column(DateTime, default=datetime.utcnow)  # 记录时间（UTC）
    result_json = Column(Text)                # 完整分析结果（含 base64 标注图）
    # 说明：演示规模下 base64 存库足够；生产环境应分离文件存储（见 web/README.md）
