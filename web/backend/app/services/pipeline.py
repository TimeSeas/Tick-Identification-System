"""Detection-first cascade with context-aware classification and local verification.

Global insect predictions describe the image; they must not veto every detected
object in a mixed scene. Confidence scores from different models are not compared.
"""
import csv
import numpy as np
from app.utils.images import load_rgb_image

DEFAULT_THRESHOLD = 90.0
FALLBACK_SOFT_PROB = 60.0
VETO_PROB = 90.0
ARACHNID = {'tick', 'mite', 'spider', 'spider_mite'}

def load_thresholds(path) -> dict:
    with open(path, encoding='utf-8-sig') as f:
        return {row['类别']: float(row['MC<2% 阈值']) for row in csv.DictReader(f)
                if row.get('类别') and row.get('MC<2% 阈值')}

def predict(classifier, rgb, **kwargs):
    if hasattr(classifier, 'predict'):
        return classifier.predict(rgb, **kwargs)
    slug, confidence = classifier.classify(rgb)
    return {'slug': slug, 'confidence': confidence,
            'candidates': [{'slug': slug, 'confidence': confidence}]}

class Pipeline:
    def __init__(self, insect_classifier, detector, species_classifier,
                 thresholds_path, cn_map, profile=None):
        self.insect_classifier = insect_classifier
        self.detector = detector
        self.species_classifier = species_classifier
        self.profile = profile or {}
        species = self.profile.get('species', {})
        self.thresholds = species.get('thresholds') or load_thresholds(thresholds_path)
        self.region_thresholds = species.get('region_thresholds', self.thresholds)
        self.padding = species.get('region_padding', .15)
        self.whole_weight = species.get('single_image_weight', .5)
        self.insect_thresholds = self.profile.get('insect', {}).get('thresholds', {})
        self.rejection = self.profile.get('rejection', {})
        self.cn_map = cn_map

    def analyze(self, image_bytes: bytes) -> dict:
        image = load_rgb_image(image_bytes)
        rgb = np.array(image)
        detections = self.detector.detect(rgb, tile=1)
        if not detections and self.detector.tile > 1:
            detections = self.detector.detect(rgb, tile=self.detector.tile)
        if not detections:
            return self._fallback(rgb)

        whole_insect = predict(self.insect_classifier, rgb)
        single = len(detections) == 1
        retained = []
        for source in detections:
            d = dict(source)
            x1, y1, x2, y2 = [int(v) for v in d['bbox']]
            x1, x2 = max(0, x1), min(image.width, x2)
            y1, y2 = max(0, y1), min(image.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            d['bbox'] = [x1, y1, x2, y2]
            dx, dy = (x2-x1)*self.padding, (y2-y1)*self.padding
            crop = image.crop((max(0, int(x1-dx)), max(0, int(y1-dy)),
                               min(image.width, int(x2+dx)), min(image.height, int(y2+dy))))
            crop_rgb = np.array(crop)
            species = predict(self.species_classifier, crop_rgb,
                              whole_rgb=rgb if single else None,
                              whole_weight=self.whole_weight if single else 0)
            slug, probability = species['slug'], species['confidence']
            table = self.thresholds if single else self.region_thresholds
            threshold = table.get(slug, DEFAULT_THRESHOLD)
            d.update(confidence=round(probability, 2), threshold=round(threshold, 4),
                     candidate_slug=slug, candidate_cn=self.cn_map.get(slug, slug),
                     top_candidates=[dict(c, cn=self.cn_map.get(c['slug'], c['slug']))
                                     for c in species['candidates']])
            confirmed = probability >= threshold
            insect = predict(self.insect_classifier, crop_rgb)
            # Reject this proposal, not the entire mixed image. Each confidence is
            # checked against its own criterion; scores from models are never ranked.
            local_non_arachnid = insect['slug'] not in ARACHNID
            whole_non_arachnid = (whole_insect['slug'] not in ARACHNID
                                   and whole_insect['confidence'] >= 90)
            if local_non_arachnid and (
                insect['confidence'] >= self.rejection.get('local_non_arachnid', 90)
                or (insect['confidence'] >= self.rejection.get('local_consensus', 80)
                    and whole_non_arachnid)
            ):
                continue
            whole_conflict = (whole_insect['slug'] != 'tick' and
                              whole_insect['confidence'] >= self.rejection.get('whole_non_tick', 90))
            strong_box = (confirmed and probability >= self.rejection.get('species_support', 90)
                          and d['score'] >= self.rejection.get('detector_support', .98))
            if whole_conflict and not strong_box:
                continue
            if confirmed:
                d.update(status='confirmed', species_slug=slug,
                         species_cn=self.cn_map.get(slug, slug), decision_reason='达到确认阈值')
            else:
                d.update(status='uncertain', species_slug='uncertain',
                         species_cn='无法确定品种', decision_reason='未达到确认阈值')
            retained.append(d)

        if not retained:
            return self._fallback(rgb, whole_insect)
        from app.services import risk_rules
        advice = risk_rules.advice_for([d['species_slug'] for d in retained],
                                       [d['status'] for d in retained])
        return {'detections': retained,
                'insect': {'slug': 'tick', 'cn': '蜱虫',
                           'confidence': round(max(d['score'] for d in retained)*100, 2)},
                'risk_advice': advice}

    def _fallback(self, rgb: np.ndarray, result=None) -> dict:
        result = result or predict(self.insect_classifier, rgb)
        slug, probability = result['slug'], result['confidence']
        threshold = self.insect_thresholds.get(slug, FALLBACK_SOFT_PROB)
        status = 'confirmed' if probability >= threshold and slug != 'tick' else 'uncertain'
        cn = self.cn_map.get(slug, slug)
        if slug == 'tick':
            message = '疑似蜱虫但尚未定位，请补充清晰照片或进行人工复核。'
        elif status == 'uncertain':
            message = f'未检出蜱虫，昆虫类别尚不确定，候选为{cn}，建议补拍或人工复核。'
        elif slug in ('mite', 'spider_mite'):
            message = f'未检出蜱虫，候选为{cn}，与蜱虫形态相似，建议人工复核。'
        else:
            message = f'未检出蜱虫，图像候选类别为{cn}。'
        return {'detections': [], 'insect': {
            'slug': slug, 'cn': cn, 'confidence': round(probability, 2),
            'status': status, 'threshold': round(threshold, 4),
            'top_candidates': [dict(c, cn=self.cn_map.get(c['slug'], c['slug']))
                               for c in result['candidates']]}, 'risk_advice': message}
