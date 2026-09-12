"""
级联推理编排（检测优先，精度优先）：
    全图检测(1024) → 检出蜱虫 → 蜱种细分类 → 分物种阈值 → 风险规则
                 → 未检出 → 切块检测补偿（小目标在整图缩小后占比过低，切块放大再试）
                 → 仍未检出 → 粗分类兜底，只做"疑似什么昆虫"的补充说明

设计取舍（2026-09-12 改造）：
- 检测模型对蜱虫判别力最强（v3 mAP@0.50=0.866），必须是第一道闸；
  粗分类曾做第一道闸时会把蜱虫误判成其他昆虫，导致检测根本没机会跑
- 大目标绝不允许被切块破坏：全图推理永远是第一优先，切块只作未检出时的补偿
- 大框否决：检出框面积占比≥5% 且全图粗分类判非蛛形纲昆虫（≥90% 置信度）
  时判定为蚜虫群等密集昆虫误检，退回兜底说明（实测蚜虫群误检框占比 6.3%/49.5%，
  真蜱虫 Top15 大框图全图粗分类 0 误伤）
- 第二意见校验：蜱种分类 uncertain 时，昆虫粗分类对同一 crop 独立再判一次，
  两个模型都不认蜱虫才剔除（防止蚜虫等被检测模型误报成蜱虫）
- 螨类相似物否决：全图粗分类判 mite/spider_mite（≥80%）且蜱种全 uncertain
  时判为螨类而非蜱虫（叶螨与蜱虫同为螨类，检测模型难分，全图证据更可靠）
- 粗分类在兜底位置判错了也只影响补充说明，绝不反向否定检测结论

依赖接口约定：
- Detector.detect(img_rgb, tile) -> [{"bbox": [...], "score": ...}]
- SpeciesClassifier.classify(crop_rgb) -> (slug, 置信度百分数)
- InsectClassifier.classify(img_rgb) -> (slug, 置信度百分数)
- risk_rules.advice_for(slugs, statuses) -> str
- cn_map 为 {slug: 中文名}
"""
import csv
import io

import numpy as np
from PIL import Image

DEFAULT_THRESHOLD = 90.0    # 蜱种阈值表缺失时的兜底
FALLBACK_SOFT_PROB = 60.0   # 粗分类兜底：低于此置信度措辞用"未能明确识别"，不斩钉截铁下结论
VETO_PROB = 90.0            # 大框否决线：全图粗分类判非蛛形纲且置信度≥此值，才敢否决检测结果
BIG_BOX_RATIO = 0.05        # 检出框面积占比≥5% 才启动否决（蚜虫群误检框实测 6.3%/49.5%；
                            # 真蜱虫 Top15 大框图全图粗分类实测 0 误伤：非蛛形纲最高只到 75.8%）
ARACHNID = {"tick", "mite", "spider"}   # 蛛形纲：与蜱虫形态相似，粗分类判这些不否决，宁可信检测
MITE_PROB = 80.0            # 螨类相似物否决线：全图粗分类判 mite/spider_mite 且≥此值、蜱种又全 uncertain 时判螨类


def load_thresholds(path) -> dict:
    thresholds = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            slug = row["类别"]
            if slug and row.get("MC<2% 阈值"):
                thresholds[slug] = float(row["MC<2% 阈值"])
    return thresholds


class Pipeline:
    def __init__(self, insect_classifier, detector, species_classifier,
                 thresholds_path, cn_map):
        self.insect_classifier = insect_classifier
        self.detector = detector
        self.species_classifier = species_classifier
        self.thresholds = load_thresholds(thresholds_path)
        self.cn_map = cn_map

    def analyze(self, image_bytes: bytes) -> dict:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)

        # 第一级：全图检测——大目标绝不能因切块破碎漏检
        detections = self.detector.detect(img_np, tile=1)

        # 第二级：全图未检出 → 切块补偿（小目标放大再试；多花几十秒，精度优先）
        if not detections:
            detections = self.detector.detect(img_np, tile=self.detector.tile)

        # 第三级：两轮检测均空 → 粗分类兜底（只作补充说明）
        if not detections:
            return self._fallback(img_np)

        # 大框否决：蚜虫群等密集昆虫会被检测模型误判为蜱虫（实测蚜虫群误检框面积占比 33%/49%）
        if self._insect_superblock(img_np, detections):
            return self._fallback(img_np)

        # 蜱种细分类 + 分物种阈值（uncertain 时做第二意见校验，防止其他昆虫误报成蜱虫）
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            crop = img.crop((x1, y1, x2, y2))
            slug, prob = self.species_classifier.classify(np.array(crop))
            thr = self.thresholds.get(slug, DEFAULT_THRESHOLD)
            d["confidence"] = round(prob, 2)
            if prob >= thr:
                d["status"] = "confirmed"
                d["species_slug"] = slug
                d["species_cn"] = self.cn_map.get(slug, slug)
                continue
            # uncertain：第二意见——昆虫粗分类对同一 crop 独立再判一次
            # 两个独立模型都不认蜱虫（粗分类判其他昆虫且置信度更高）→ 疑似误检，剔除
            in_slug, in_prob = self.insect_classifier.classify(np.array(crop))
            if in_slug != "tick" and in_prob > max(prob, FALLBACK_SOFT_PROB):
                d["rejected"] = True
                continue
            d["status"] = "uncertain"
            d["species_slug"] = "uncertain"
            d["species_cn"] = "无法确定品种"

        # 全部检出框被第二意见剔除 → 退回粗分类兜底说明
        detections = [d for d in detections if not d.get("rejected")]
        if not detections:
            return self._fallback(img_np)

        # 螨类相似物否决：全图粗分类判 mite/spider_mite（叶螨等）且蜱种全 uncertain
        # → 判为螨类而非蜱虫（叶螨与蜱虫同为螨类，检测模型难分，全图证据更可靠）
        if all(d["status"] == "uncertain" for d in detections) and self._mite_like(img_np):
            return self._fallback(img_np)

        slugs = [d["species_slug"] for d in detections]
        statuses = [d["status"] for d in detections]
        from app.services import risk_rules
        advice = risk_rules.advice_for(slugs, statuses)

        return {
            "detections": detections,
            # 检测优先后，蜱虫分支的 insect 信息取最高检测分（前端仅无检出时才展示此字段）
            "insect": {"slug": "tick", "cn": "蜱虫",
                       "confidence": round(max(d["score"] for d in detections) * 100, 2)},
            "risk_advice": advice,
        }

    def _insect_superblock(self, img_np: np.ndarray, detections: list) -> bool:
        """大框误检否决：检测出大框（面积占比≥15%）且全图粗分类判非蛛形纲昆虫（≥90%）
        时，判定为蚜虫群等密集昆虫被误检为蜱虫，返回 True 走兜底说明。
        真蜱虫框占比通常<1%，宏观特写蜱虫粗分类也会判 tick/spider/mite，都不会被否决"""
        h, w = img_np.shape[:2]
        max_area = max((b[2] - b[0]) * (b[3] - b[1]) for b in (d["bbox"] for d in detections))
        if max_area / (h * w) < BIG_BOX_RATIO:
            return False
        slug, prob = self.insect_classifier.classify(img_np)
        return slug not in ARACHNID and prob >= VETO_PROB

    def _mite_like(self, img_np: np.ndarray) -> bool:
        """全图粗分类判螨类相似物（mite/spider_mite）且置信度足够高"""
        slug, prob = self.insect_classifier.classify(img_np)
        return slug in ("mite", "spider_mite") and prob >= MITE_PROB

    def _fallback(self, img_np: np.ndarray) -> dict:
        """两轮检测均未检出的兜底：粗分类只作补充说明，绝不反向否定检测结论"""
        slug, prob = self.insect_classifier.classify(img_np)
        cn = self.cn_map.get(slug, slug)
        if slug == "tick":
            # 粗分类觉得像蜱虫但检测器定位不到：宁可"不确定"，也不要"非蜱虫"
            return {
                "detections": [],
                "insect": {"slug": "tick", "cn": "蜱虫", "confidence": prob},
                "risk_advice": ("疑似蜱虫但检测器未能定位，建议更换清晰、对焦准确的"
                                "照片重试。本结果仅供参考，不构成医疗建议。"),
            }
        if slug in ("mite", "spider_mite"):
            # 螨虫/叶螨与蜱虫形态相似，措辞留余地
            return {
                "detections": [],
                "insect": {"slug": slug, "cn": cn, "confidence": prob},
                "risk_advice": (f"未检出蜱虫，图像识别为{cn}（与蜱虫形态相似）。"
                                "若照片来自疑似叮咬现场，建议人工复核。"
                                "本结果仅供参考，不构成医疗建议。"),
            }
        if prob < FALLBACK_SOFT_PROB:
            # 低置信度：不斩钉截铁下结论
            return {
                "detections": [],
                "insect": {"slug": slug, "cn": cn, "confidence": prob},
                "risk_advice": (f"未检出蜱虫，未能明确识别昆虫类别（疑似{cn}）。"
                                "本结果仅供参考，不构成医疗建议。"),
            }
        return {
            "detections": [],
            "insect": {"slug": slug, "cn": cn, "confidence": prob},
            "risk_advice": (f"未检出蜱虫，图像识别为{cn}（非蜱虫），无需蜱虫风险提示。"
                            "本结果仅供参考，不构成医疗建议。"),
        }
