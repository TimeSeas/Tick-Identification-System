"""
学名 slug → 中文名映射加载（读 data/tick_species/物种中文名对照表.csv）
进程内只加载一次：用模块级缓存，避免每个请求都读磁盘
"""
from functools import lru_cache
import csv

from app.config import settings


@lru_cache(maxsize=1)
def get_cn_map() -> dict[str, str]:
    """返回 {slug: 中文名}。lru_cache 保证整个进程只读一次 CSV"""
    cn_map = {}
    with open(settings.cn_table_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            slug = row.get("文件夹名(slug)")
            name = row.get("中文名")
            if slug and name:
                cn_map[slug] = name
    return cn_map
