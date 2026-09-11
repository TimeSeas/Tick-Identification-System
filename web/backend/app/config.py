"""pydantic-settings：模型路径/阈值表/中文表/数据库 URL，环境变量可覆盖"""
from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    #BaseSettings 会自动读环境变量——以后 Docker 里 DETECTOR_PATH=/app/models/xxx 就能覆盖，不改代码
    # 每个配置项 = 一个属性，环境变量同名可覆盖（企业 12-factor 规范）
    db_url: str = "sqlite:///./data.db"          # 数据库连接串
    detector_path: str = "../../output/det_v2/mask_rcnn_ticks_v2_ep10.pth"   # M1 模型（v2：1024 分辨率+增强，mAP@0.50=0.857 > v1 0.842）
    species_model_path: str = "../../output/tick_species_clean_448/model/tick_species_best.pth"
    thresholds_path: str = "../../output/tick_species_clean_448/thresholds.csv"
    cn_table_path: str = "../../data/tick_species/物种中文名对照表.csv"
    species_split_json: str = "../../data/tick_species/split_clean.json"
    detect_tile: int = 2   # 检测切块推理的切分数（1=关闭；2=2x2 提升小目标检出，代价是推理时间×4）
settings = Settings()   # 全局单例，其他模块 from app.config import settings

