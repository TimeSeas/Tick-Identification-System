import io
from pathlib import Path
import sys
import unittest
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'web/backend'))
from app.services.pipeline import Pipeline


def prediction(slug,confidence):
    return dict(slug=slug,confidence=confidence,candidates=[dict(slug=slug,confidence=confidence)])


class SpeciesRescueTests(unittest.TestCase):
    def analyze(self,reference_confidence,whole_conflict=False):
        class Detector:
            tile=1
            def detect(self,*args,**kwargs):return [dict(bbox=[5,5,25,25],score=.99)]
        class Species:
            def predict(self,*args,**kwargs):
                return dict(prediction('new_b',99),reference_prediction=prediction('old_a',reference_confidence))
        class Insect:
            def predict(self,rgb):return prediction('mite' if whole_conflict and rgb.shape[0]==40 else 'tick',99)
        profile=dict(species=dict(thresholds={'old_a':55,'new_b':55},region_thresholds={'old_a':55,'new_b':55},
            decision_mode='preserve_confirmed_rescue',rescue_thresholds={'old_a':90,'new_b':90}))
        pipeline=Pipeline(Insect(),Detector(),Species(),None,{},profile)
        stream=io.BytesIO();Image.new('RGB',(40,40)).save(stream,format='PNG')
        return pipeline.analyze(stream.getvalue())

    def test_confirmed_original_result_is_preserved_when_specialist_disagrees(self):
        result=self.analyze(95)
        self.assertEqual(result['detections'][0]['species_slug'],'old_a')
        self.assertEqual(result['detections'][0]['confidence'],95)

    def test_only_uncertain_original_result_can_be_rescued(self):
        result=self.analyze(40)
        self.assertEqual(result['detections'][0]['species_slug'],'new_b')
        self.assertEqual(result['detections'][0]['status'],'confirmed')

    def test_rescue_cannot_resurrect_a_proposal_vetoed_by_original_support(self):
        self.assertEqual(self.analyze(40,whole_conflict=True)['detections'],[])
        self.assertEqual(len(self.analyze(95,whole_conflict=True)['detections']),1)


if __name__=='__main__':unittest.main()
