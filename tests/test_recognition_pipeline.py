import io
import sys
from pathlib import Path
import unittest

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'web/backend'))
from app.services.recognition_pipeline import RecognitionPipeline,coarse_guard_rejects


class FakeDetector:
    tile=2
    def __init__(self):
        self.calls=[]
    def detect(self,rgb,tile=None):
        self.calls.append(tile)
        return [dict(bbox=[0,0,10,10],score=.4)]


class FakeRecognition:
    def __init__(self,probability=.9):
        self.probability=probability
    def predict(self,rgb):
        return dict(tick_probability=self.probability,slug='tick_a',confidence=95,
                    candidates=[dict(slug='tick_a',confidence=95)])


class FakeInsect:
    def predict(self,rgb):
        return dict(slug='beetle',confidence=99,candidates=[dict(slug='beetle',confidence=99)])


class RecognitionPipelineTests(unittest.TestCase):
    def test_arachnid_candidates_are_not_vetoed_by_coarse_guard(self):
        for slug in ('tick','mite','spider','spider_mite'):
            self.assertFalse(coarse_guard_rejects(dict(slug=slug,confidence=100),.2,
                dict(confidence=90,strong_presence=.95)))

    def test_guard_checks_coarse_confidence_only_for_non_arachnid_regions(self):
        service,_=self.pipeline(probability=.7)
        service.profile['recognition']['coarse_guard']=dict(confidence=90,strong_presence=.95)
        self.assertEqual(service.analyze(self.image_bytes())['detections'],[])
        service.recognition_classifier.probability=.99
        self.assertEqual(len(service.analyze(self.image_bytes())['detections']),1)

    def pipeline(self,probability=.9,threshold=90):
        detector=FakeDetector()
        service=RecognitionPipeline(FakeInsect(),detector,FakeRecognition(probability),
            dict(species=dict(thresholds={'tick_a':threshold}),
                 recognition=dict(tick_threshold=.5,detector_threshold=.3)),{})
        return service,detector

    def image_bytes(self):
        image=Image.new('RGB',(20,20)); stream=io.BytesIO(); image.save(stream,format='PNG')
        return stream.getvalue()

    def test_mixed_image_global_coarse_prediction_does_not_veto_verified_tick(self):
        service,detector=self.pipeline()
        result=service.analyze(self.image_bytes())
        self.assertEqual(len(result['detections']),1)
        self.assertEqual(result['detections'][0]['status'],'confirmed')
        self.assertEqual(detector.calls,[1])
        self.assertEqual(detector.CONFIDENCE,.3)

    def test_presence_rejection_and_species_rejection_have_different_outcomes(self):
        service,_=self.pipeline(probability=.1)
        self.assertEqual(service.analyze(self.image_bytes())['detections'],[])
        service,_=self.pipeline(threshold=99)
        detection=service.analyze(self.image_bytes())['detections'][0]
        self.assertEqual(detection['status'],'uncertain')
        self.assertEqual(detection['candidate_slug'],'tick_a')
        self.assertEqual(detection['species_slug'],'uncertain')

    def test_invalid_presence_threshold_fails_before_inference(self):
        with self.assertRaises(ValueError):
            RecognitionPipeline(None,None,None,dict(species=dict(thresholds={}),
                recognition=dict(tick_threshold=-1,detector_threshold=.3)),{})


if __name__=='__main__':
    unittest.main()
