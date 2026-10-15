import io
import sys
import uuid
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.pipeline import Pipeline

class Detector:
    tile=2
    def __init__(self,boxes): self.boxes=boxes
    def detect(self,image,tile=None):return [dict(x) for x in self.boxes]

class Species:
    def __init__(self,values):self.values=iter(values)
    def classify(self,image):return next(self.values)

class Insect:
    def __init__(self,slug,prob):self.slug,self.prob=slug,prob
    def classify(self,image):return self.slug,self.prob

class PipelineRegressions(unittest.TestCase):
    def setUp(self):
        directory=Path(__file__).resolve().parents[3]/'output/optimization_20261015'
        directory.mkdir(parents=True,exist_ok=True)
        self.threshold=directory/f'threshold_test_{uuid.uuid4().hex}.csv'
        self.threshold.write_text('类别,MC<2% 阈值\nknown,95\n',encoding='utf-8')
        stream=io.BytesIO();Image.new('RGB',(100,100),'white').save(stream,format='PNG');self.image=stream.getvalue()
    def tearDown(self):self.threshold.unlink(missing_ok=True)
    def pipeline(self,boxes,species,insect):return Pipeline(insect,Detector(boxes),Species(species),self.threshold,{'known':'已知蜱种','aphid':'蚜虫'})
    def test_mixed_image_cannot_veto_confirmed_tick_globally(self):
        boxes=[{'bbox':[1,1,20,20],'score':.98},{'bbox':[30,30,99,99],'score':.8}]
        class MixedInsect:
            def classify(self,image):
                return ('tick',99) if image.shape[0]<40 else ('aphid',99)
        result=self.pipeline(boxes,[('known',99),('known',60)],MixedInsect()).analyze(self.image)
        self.assertEqual(len(result['detections']),1)
        self.assertEqual(result['detections'][0]['status'],'confirmed')
    def test_incomparable_model_scores_do_not_discard_tick(self):
        result=self.pipeline([{'bbox':[1,1,20,20],'score':.98}],[('known',80)],Insect('aphid',85)).analyze(self.image)
        self.assertEqual(len(result['detections']),1)
        self.assertEqual(result['detections'][0]['status'],'uncertain')
    def test_uncertain_keeps_candidate_name_and_threshold(self):
        result=self.pipeline([{'bbox':[1,1,20,20],'score':.98}],[('known',80)],Insect('tick',99)).analyze(self.image)
        candidate=result['detections'][0]
        self.assertEqual(candidate['candidate_slug'],'known')
        self.assertEqual(candidate['candidate_cn'],'已知蜱种')
        self.assertEqual(candidate['threshold'],95)
        self.assertEqual(candidate['species_slug'],'uncertain')
    def test_arachnid_second_opinion_does_not_remove_strong_tick(self):
        class LocalMite:
            def classify(self,image):
                return ('tick',99) if image.shape[0]==100 else ('spider_mite',99)
        result=self.pipeline([{'bbox':[1,1,20,20],'score':.98}],[('known',80)],LocalMite()).analyze(self.image)
        self.assertEqual(len(result['detections']),1)

    def test_multiple_objects_use_region_threshold_without_global_species_fusion(self):
        class RecordingSpecies:
            def __init__(self):self.whole_inputs=[]
            def predict(self,rgb,whole_rgb=None,whole_weight=0):
                self.whole_inputs.append((whole_rgb,whole_weight))
                return {'slug':'known','confidence':85,'candidates':[{'slug':'known','confidence':85}]}
        classifier=RecordingSpecies()
        boxes=[{'bbox':[1,1,20,20],'score':.99},{'bbox':[40,40,70,70],'score':.99}]
        pipeline=Pipeline(Insect('tick',99),Detector(boxes),classifier,self.threshold,{},
                          profile={'species':{'thresholds':{'known':50},'region_thresholds':{'known':95}}})
        result=pipeline.analyze(self.image)
        self.assertEqual([d['status'] for d in result['detections']],['uncertain','uncertain'])
        self.assertEqual(classifier.whole_inputs,[(None,0),(None,0)])

    def test_single_object_uses_context_and_whole_image(self):
        class RecordingSpecies:
            def predict(self,rgb,whole_rgb=None,whole_weight=0):
                self.shape=rgb.shape;self.whole_shape=whole_rgb.shape;self.weight=whole_weight
                return {'slug':'known','confidence':99,'candidates':[{'slug':'known','confidence':99}]}
        classifier=RecordingSpecies()
        pipeline=Pipeline(Insect('tick',99),Detector([{'bbox':[20,20,60,60],'score':.99}]),
                          classifier,self.threshold,{})
        pipeline.analyze(self.image)
        self.assertEqual(classifier.shape[:2],(52,52))
        self.assertEqual(classifier.whole_shape[:2],(100,100))
        self.assertEqual(classifier.weight,.5)

if __name__=='__main__':unittest.main()
