"""pydantic-settings：模型路径/阈值表/中文表/数据库 URL，环境变量可覆盖"""
from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    #BaseSettings 会自动读环境变量——以后 Docker 里 DETECTOR_PATH=/app/models/xxx 就能覆盖，不改代码
    # 每个配置项 = 一个属性，环境变量同名可覆盖（企业 12-factor 规范）
    db_url: str = "sqlite:///./data.db"          # 数据库连接串
    detector_path: str = "../../output/det_v3/mask_rcnn_ticks_v3.pth"   # M1 模型（v3：1024 训练 + 小目标补标 579 图，mAP@0.50=0.866（1024 评估）> v2 0.845）
    species_model_path: str = "../../output/tick_species_clean_448/model/tick_species_best.pth"
    thresholds_path: str = "../../output/tick_species_clean_448/thresholds.csv"
    cn_table_path: str = "../../data/tick_species/物种中文名对照表.csv"
    species_split_json: str = "../../data/tick_species/split_clean.json"
    species_input_size: int = 448  # 与 clean_448 训练及阈值保持一致
    inference_profile_path: str = "../../configs/inference_optimization.json"
    detect_tile: int = 2   # 切块补偿的切分数：全图检测未检出时才启用（1=不补偿；2=2x2 小目标放大再试，多花几十秒，精度优先）
    insect_model_path: str = "../../output/insect_classifier/model/insect_best.pth"  # 昆虫粗分类（级联第一级）
    insect_split_json: str = "../../data/insects_split.json"
settings = Settings()   # 全局单例，其他模块 from app.config import settings

