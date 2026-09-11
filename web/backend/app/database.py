"""数据库引擎"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config import settings

engine = create_engine(settings.db_url, connect_args={"check_same_thread": False})  # SQLite 特殊参数
SessionLocal = sessionmaker(bind=engine, autoflush=False)

class Base(DeclarativeBase):   # 所有 ORM 模型的父类（M3 建表用）
    pass


def get_db():
    """FastAPI 依赖：每个请求一个独立 Session，用完自动关闭（并发安全）"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()