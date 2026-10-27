import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from species_threshold_calibration import calibrate_class_thresholds


class ClassThresholdTests(unittest.TestCase):
    def test_duplicate_proposals_cannot_outweigh_more_correct_source_images(self):
        rows=[dict(path='a',label=0,probabilities=[.8,.2]) for _ in range(100)]
        rows += [dict(path=f'aw{i}',label=1,probabilities=[.51,.49]) for i in range(19)]
        rows += [dict(path=f'b{i}',label=1,probabilities=[.2,.8]) for i in range(20)]
        rows += [dict(path='u0',label=-1,probabilities=[.8,.2]),
                 dict(path='u1',label=-1,probabilities=[.2,.8])]
        rows += [dict(path=f'u{i}',label=-1,probabilities=[0.,0.],detected=False) for i in range(2,20)]
        result=calibrate_class_thresholds(rows,['a','b'],1.000001,max_unknown_acceptance=.05)
        self.assertEqual(result['metrics']['known_useful_images'],20)
        self.assertEqual(result['metrics']['unknown_accepted_images'],1)

    def test_ambiguous_class_does_not_force_rejecting_supported_other_class(self):
        rows=[dict(path=f'a{i}',label=0,probabilities=[.7,.3]) for i in range(30)]
        rows += [dict(path=f'b{i}',label=1,probabilities=[.3,.7]) for i in range(30)]
        rows += [dict(path='unknown',label=-1,probabilities=[.95,.05])]
        result=calibrate_class_thresholds(rows,['a','b'],.98,max_unknown_acceptance=0.)
        self.assertEqual(result['metrics']['known_correct_accepted'],30)
        self.assertEqual(result['metrics']['unknown_accepted_images'],0)
        self.assertGreater(result['thresholds']['a'],.95)
        self.assertLessEqual(result['thresholds']['b'],.7)

    def test_tiny_class_cannot_learn_lower_threshold(self):
        rows=[dict(path='rare',label=0,probabilities=[.7,.3]),
              dict(path='unknown',label=-1,probabilities=[.9,.1])]
        self.assertIsNone(calibrate_class_thresholds(rows,['a','b'],.98,max_unknown_acceptance=0.))


if __name__=='__main__':unittest.main()
