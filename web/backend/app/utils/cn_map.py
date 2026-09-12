"""
学名 slug → 中文名映射加载
- 蜱种：读 data/tick_species/物种中文名对照表.csv
- 昆虫粗分类 28 类：内置 INSECT_CN 表（与 annotation_guide 的 28 类清单一致）
进程内只加载一次：用模块级缓存，避免每个请求都读磁盘
"""
from functools import lru_cache
import csv

from app.config import settings

# 昆虫粗分类 28 类 slug → 中文名（与 scripts/sam_auto_label.py 的 CLASS_MAPPING 对应）
INSECT_CN = {
    "tick": "蜱虫", "mite": "螨虫", "ant": "蚂蚁", "bee": "蜜蜂",
    "beetle": "甲虫", "butterfly": "蝴蝶", "dragonfly": "蜻蜓", "fly": "苍蝇",
    "grasshopper": "蚱蜢", "ladybug": "瓢虫", "mosquito": "蚊子", "spider": "蜘蛛",
    "wasp": "黄蜂",
    "africanized_honey_bee": "非洲化蜜蜂（杀人蜂）", "aphid": "蚜虫",
    "armyworm": "粘虫", "brown_marmorated_stink_bug": "棕纹蝽",
    "cabbage_looper": "菜青虫", "citrus_canker": "柑橘溃疡病（病害）",
    "colorado_potato_beetle": "科罗拉多甲虫", "corn_borer": "玉米螟",
    "corn_earworm": "棉铃虫", "fall_armyworm": "秋粘虫", "fruit_fly": "果蝇",
    "spider_mite": "蜘蛛螨", "thrips": "蓟马", "tomato_hornworm": "番茄天蛾",
    "western_corn_rootworm": "西方玉米根虫",
}


@lru_cache(maxsize=1)
def get_cn_map() -> dict[str, str]:
    """返回 {slug: 中文名}（蜱种对照表 + 昆虫粗分类内置表合并）。
    lru_cache 保证整个进程只读一次 CSV"""
    cn_map = dict(INSECT_CN)
    with open(settings.cn_table_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            slug = row.get("文件夹名(slug)")
            name = row.get("中文名")
            if slug and name:
                cn_map[slug] = name   # 蜱种同名键覆盖内置表（tick 用对照表里的"蜱虫"）
    return cn_map
