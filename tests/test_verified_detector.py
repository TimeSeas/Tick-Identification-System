import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'web/backend'))
from app.services.verified_detector import VerifiedDetector


class VerifiedDetectorTests(unittest.TestCase):
    def make(self,probabilities):
        service=VerifiedDetector.__new__(VerifiedDetector)
        service.device='cpu'; service.tile=2
        service.proposals=SimpleNamespace(detect=lambda rgb,tile: [
            dict(bbox=[2,4,12,14],score=.99),dict(bbox=[3,5,13,15],score=.9),
            dict(bbox=[20,20,30,30],score=.99)])
        captured={}
        def transform(images,targets):
            targets[0]['boxes']*=2
            return SimpleNamespace(tensors=images[0][None],image_sizes=[(80,80)]),targets
        def pool(features,boxes,sizes):
            captured['boxes']=boxes[0].clone(); return torch.zeros(len(boxes[0]),1)
        logits=torch.log(torch.tensor([[1-p,p] for p in probabilities]))
        service.model=SimpleNamespace(transform=transform,backbone=lambda x:{'0':x},
            roi_heads=SimpleNamespace(box_roi_pool=pool,box_head=lambda x:x,
                box_predictor=lambda x:(logits,None)))
        return service,captured

    def test_transformed_coordinates_and_verifier_scores_drive_acceptance(self):
        service,captured=self.make([.8,.9,.3])
        result=service.detect(np.zeros((40,40,3),dtype=np.uint8),tile=1)
        self.assertEqual(captured['boxes'][0].tolist(),[4,8,24,28])
        self.assertEqual(result,[dict(bbox=[3,5,13,15],score=.9)])

    def test_empty_candidates_skip_model(self):
        service,_=self.make([.8,.9,.3]); service.proposals.detect=lambda rgb,tile:[]
        service.model=None
        self.assertEqual(service.detect(np.zeros((40,40,3),dtype=np.uint8)),[])


if __name__=='__main__':
    unittest.main()
