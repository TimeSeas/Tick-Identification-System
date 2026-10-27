import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'web/backend'))
from app.services.classifier import SpeciesClassifier


class WholeReferenceTests(unittest.TestCase):
    def test_only_whole_view_uses_reference_model(self):
        classifier=SpeciesClassifier.__new__(SpeciesClassifier)
        classifier.model=object();classifier.whole_model=object()
        classifier.classes=['a','b'];classifier.input_size=448
        classifier.device='cpu';classifier.view_mode='center_full'
        crop=np.zeros((4,4,3),dtype=np.uint8);whole=np.zeros((8,8,3),dtype=np.uint8)
        calls=[]
        def infer(model,rgb,*args):
            calls.append((model,rgb.shape))
            return np.array([.9,.1]) if model is classifier.whole_model else np.array([.2,.8])
        with patch('app.services.classifier.probabilities',side_effect=infer):
            result=classifier.predict(crop,whole_rgb=whole,whole_weight=.5)
        self.assertEqual(result['slug'],'a')
        self.assertAlmostEqual(result['confidence'],55.)
        self.assertEqual(calls,[(classifier.model,crop.shape),(classifier.whole_model,whole.shape)])
        calls.clear()
        with patch('app.services.classifier.probabilities',side_effect=infer):
            result=classifier.predict(crop)
        self.assertEqual(result['slug'],'b')
        self.assertEqual(len(calls),1)


if __name__=='__main__':unittest.main()
