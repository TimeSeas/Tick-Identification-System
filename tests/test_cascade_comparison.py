import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_recognition_cascade import summarize,CachedDetector


class CascadeComparisonTests(unittest.TestCase):
    def test_cached_detector_uses_the_evaluated_threshold(self):
        detector=CachedDetector(); detector.row=dict(full=[dict(score=.4)],tiles=[])
        self.assertEqual(detector.detect(None,tile=1),[])
        detector.CONFIDENCE=.3
        self.assertEqual(detector.detect(None,tile=1),[dict(score=.4)])

    def test_proxy_image_rate_is_not_box_rate_and_empty_coco_is_not_negative_truth(self):
        box=dict(bbox=[0,0,10,10],score=.9,tick_probability=.8)
        rows=[dict(annotations=[dict(bbox=[0,0,10,10])]),
              dict(annotations=[],negative_label='mite'),dict(annotations=[])]
        traces=[dict(old=[box],new=[box]),dict(old=[box,box],new=[box,box]),dict(old=[box],new=[box])]
        metrics=summarize(rows,traces,'new',.5)
        self.assertEqual((metrics['tp'],metrics['fp'],metrics['fn']),(1,0,0))
        self.assertEqual(metrics['proxy_images'],1)
        self.assertEqual(metrics['proxy_positive_images'],1)
        self.assertEqual(metrics['positive_images'],1)

    def test_rejection_reports_localization_recall_loss(self):
        rows=[dict(annotations=[dict(bbox=[0,0,10,10])])]
        box=dict(bbox=[0,0,10,10],score=.9,tick_probability=.4)
        trace=[dict(old=[box],new=[box])]
        self.assertEqual(summarize(rows,trace,'old')['recall'],1)
        self.assertEqual(summarize(rows,trace,'new',.5)['recall'],0)
        self.assertEqual(summarize(rows,trace,'new',.3)['recall'],1)


if __name__=='__main__':
    unittest.main()
