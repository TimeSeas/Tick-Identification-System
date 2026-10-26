import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from calibrate_recognition_rebuild import acceptance_threshold,binary_threshold


class CalibrationTests(unittest.TestCase):
    def test_tied_species_scores_cannot_be_cherry_picked(self):
        records=[dict(species_confidence=.9,species_prediction=0,species_label=0) for _ in range(20)]
        records+=[dict(species_confidence=.9,species_prediction=0,species_label=1) for _ in range(10)]
        rule=acceptance_threshold(records)
        self.assertEqual(rule['accepted'],0)
        self.assertGreater(rule['threshold'],1)

    def test_small_perfect_group_does_not_imply_reliable_threshold(self):
        rule=acceptance_threshold([dict(species_confidence=.9,species_prediction=0,species_label=0)])
        self.assertEqual(rule['accepted'],0)

    def test_positive_source_strata_cannot_be_hidden_by_easy_species(self):
        rows=[dict(tick_label=1,tick_probability=.99,sample_type='species_roi') for _ in range(100)]
        rows+=[dict(tick_label=1,tick_probability=.1,sample_type='coco_tick_roi') for _ in range(10)]
        rows+=[dict(tick_label=0,tick_probability=.2,sample_type='non_tick_hard_roi')]
        result=binary_threshold(rows)
        self.assertEqual(result['threshold'],.1)
        self.assertEqual(result['observed_recall'],1)
        self.assertEqual(result['observed_proxy_fpr'],1)


if __name__=='__main__':
    unittest.main()
