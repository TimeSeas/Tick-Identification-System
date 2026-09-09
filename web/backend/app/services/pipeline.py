"""
级联推理编排：M1 蜱虫检测 → 裁剪 ROI → M2 蜱种分类 → 分物种阈值决策 → 风险规则
是整个系统的"总导演"，路由层只调 pipeline.analyze() 一个方法。

依赖接口约定（其他 service 按此实现即可无缝对接）：
- Detector.detect(img_rgb: np.ndarray[H,W,3]) -> list[dict]
    返回 [{"bbox": [x1,y1,x2,y2], "score": 0.93}, ...]，已按分数过滤
- SpeciesClassifier.classify(crop_rgb: np.ndarray) -> tuple[str, float]
    返回 (物种slug, 置信度百分比 0-100)
- risk_rules.advice_for(species_slugs: list[str], statuses: list[str]) -> str
    返回风险建议文案（含未检出分支）
- cn_map 为 {slug: 中文名} 字典（由 utils/cn_map.py 加载）
"""
import csv
import io

import numpy as np
from PIL import Image

DEFAULT_THRESHOLD = 90.0  # 阈值表缺失物种时的兜底（偏严：宁可不答）


def load_thresholds(path) -> dict[str, float]:
    """读取 thresholds.csv 的"MC<2% 阈值"列 -> {slug: 阈值}"""
    thresholds = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            slug = row["类别"]
            if slug and row.get("MC<2% 阈值"):
                thresholds[slug] = float(row["MC<2% 阈值"])
    return thresholds


class Pipeline:
    def __init__(self, detector, classifier, thresholds_path, cn_map):#在Pipeline类的 __init__方法中,接收传入的其他类的实例
        self.detector = detector
        self.classifier = classifier
        self.thresholds = load_thresholds(thresholds_path)
        self.cn_map = cn_map

    def analyze(self, image_bytes: bytes) -> dict:
        """主入口：图片字节 -> 结构化结果 dict（不含可视化图，路由层另行生成）"""
        # 统一转 RGB numpy 数组，坏图直接抛异常交给全局异常处理器
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)

        detections = self.detector.detect(img_np)#调用detector类的detect方法，得到bbox和score
        if not detections:
            return {
                "detections": [],
                "risk_advice": self._no_tick_advice(),
            }

        for d in detections:
            # 裁剪检测框区域（PIL 自动处理越界），交给分类器
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            crop = img.crop((x1, y1, x2, y2))
            slug, prob = self.classifier.classify(np.array(crop))#调用classifier类的classify方法得到，蜱虫的类别和置信值

            # 分物种阈值决策：置信度达标才算 confirmed，否则 uncertain
            # （论文创新点：Villon 方法复现出的每物种阈值在这里落地）
            thr = self.thresholds.get(slug, DEFAULT_THRESHOLD)
            d["confidence"] = round(prob, 2)
            if  prob >= thr:
                d["status"] = "confirmed"
                d["species_slug"] = slug
                d["species_cn"] = self.cn_map.get(slug, slug)
                
            else:
                d["status"] = "uncertain"
                d["species_slug"] = "uncertain"
                d["species_cn"] = "无法确定品种"
                

        # 汇总风险建议：所有确认物种的规则文案合并 + 免责声明
        slugs = [d["species_slug"] for d in detections]
        statuses = [d["status"] for d in detections]
        from app.services import risk_rules
        advice = risk_rules.advice_for(slugs, statuses)

        return {
            "detections": detections,
            "risk_advice": advice,
        }

    @staticmethod
    def _no_tick_advice() -> str:
        return "未检测到蜱虫。若怀疑被叮咬，请留意皮肤红肿变化，必要时就医。" \
               "本结果仅供参考，不构成医疗建议。"
