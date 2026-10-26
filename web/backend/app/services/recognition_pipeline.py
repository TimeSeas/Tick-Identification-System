"""Optional learned-presence cascade, enabled only after frozen model evaluation."""
import math

import numpy as np

from app.services.pipeline import Pipeline, predict, DEFAULT_THRESHOLD
from app.utils.images import load_rgb_image


def score_proposals(image,proposals,recognition_classifier):
    """Same region pixels for experiment and service. Boxes are XYXY, EXIF normalized."""
    from app.services.recognition_model import prepare_image
    scored=[]
    for source in proposals:
        x1,y1,x2,y2=map(int,source['bbox'])
        box=[max(0,x1),max(0,y1),min(image.width,x2),min(image.height,y2)]
        if box[2]<=box[0] or box[3]<=box[1]:
            continue
        crop=prepare_image(image,box,normalize_exif=False)
        result=recognition_classifier.predict(np.array(crop))
        probability=result['tick_probability']
        if not math.isfinite(probability) or not 0<=probability<=1:
            raise ValueError('Invalid recognition presence probability')
        scored.append((dict(source,bbox=box,tick_probability=probability),result,crop))
    return scored


class RecognitionPipeline(Pipeline):
    def __init__(self,insect_classifier,detector,recognition_classifier,profile,cn_map,
                 species_classifier=None):
        self.insect_classifier=insect_classifier
        self.detector=detector
        self.recognition_classifier=recognition_classifier
        self.species_classifier=species_classifier
        self.cn_map=cn_map
        self.profile=profile
        self.thresholds=profile['species']['thresholds']
        self.region_thresholds=profile['species'].get('region_thresholds',self.thresholds)
        self.insect_thresholds=profile.get('insect',{}).get('thresholds',{})
        self.tick_threshold=float(profile['recognition']['tick_threshold'])
        self.detector_threshold=float(profile['recognition']['detector_threshold'])
        if not 0<=self.tick_threshold<=1 or not 0<self.detector_threshold<=.5:
            raise ValueError('Invalid presence or detector threshold')
        # Detector normally filters at .5 before returning proposals. The learned
        # verifier must actually receive the lower-score candidates it was given.
        self.detector.CONFIDENCE=self.detector_threshold

    def analyze(self,image_bytes):
        image=load_rgb_image(image_bytes)
        rgb=np.array(image)
        proposals=[d for d in self.detector.detect(rgb,tile=1) if d['score']>=self.detector_threshold]
        if not proposals and self.detector.tile>1:
            proposals=[d for d in self.detector.detect(rgb,tile=self.detector.tile)
                       if d['score']>=self.detector_threshold]
        scored=score_proposals(image,proposals,self.recognition_classifier)
        retained=[]
        single=len(proposals)==1
        for detection,result,crop in scored:
            if detection['tick_probability']<self.tick_threshold:
                continue
            # Presence and species can be promoted independently after evaluation.
            if self.species_classifier is not None:
                result=predict(self.species_classifier,np.array(crop),
                    whole_rgb=rgb if single else None,
                    whole_weight=self.profile['species'].get('single_image_weight',.5) if single else 0)
            slug,confidence=result['slug'],result['confidence']
            threshold=(self.thresholds if single else self.region_thresholds).get(slug,DEFAULT_THRESHOLD)
            confirmed=confidence>=threshold
            detection.update(confidence=round(confidence,2),threshold=round(threshold,4),
                candidate_slug=slug,candidate_cn=self.cn_map.get(slug,slug),
                top_candidates=[dict(c,cn=self.cn_map.get(c['slug'],c['slug'])) for c in result['candidates']],
                status='confirmed' if confirmed else 'uncertain',
                species_slug=slug if confirmed else 'uncertain',
                species_cn=self.cn_map.get(slug,slug) if confirmed else '无法确定品种',
                decision_reason='达到确认阈值' if confirmed else '未达到确认阈值')
            retained.append(detection)
        if not retained:
            return self._fallback(rgb)
        from app.services import risk_rules
        return dict(detections=retained,
            insect=dict(slug='tick',cn='蜱虫',confidence=round(max(d['score'] for d in retained)*100,2)),
            risk_advice=risk_rules.advice_for([d['species_slug'] for d in retained],[d['status'] for d in retained]))
