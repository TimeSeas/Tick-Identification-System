import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_recognition_profile import make_profile


class ProfileTests(unittest.TestCase):
    def reports(self):
        val=dict(gates=dict(recall=True,false_positive=True),metadata=dict(subset='val',checkpoint_sha256='model',
            new_detection_threshold=.5,tick_threshold=.95,detector_sha256='det',old_species_sha256='species',old_insect_sha256='insect',
            split_sha256='split',old_profile_sha256='profile'))
        cal=copy.deepcopy(val); cal['metadata']['subset']='calibration'
        return val,cal

    def test_failed_or_changed_policy_cannot_create_default_profile(self):
        val,cal=self.reports(); cal['gates']['recall']=False
        with self.assertRaises(ValueError):
            make_profile({},val,cal,'model','manifest','profile')
        val,cal=self.reports(); cal['metadata']['tick_threshold']=.2
        with self.assertRaises(ValueError):
            make_profile({},val,cal,'model','manifest','profile')

    def test_new_presence_keeps_original_species_and_original_profile(self):
        val,cal=self.reports()
        baseline=dict(species=dict(model_sha256='species'),insect=dict(model_sha256='insect'))
        result=make_profile(baseline,val,cal,'model','manifest','profile')
        self.assertNotIn('recognition',baseline)
        self.assertEqual(result['species'],baseline['species'])
        self.assertEqual(result['recognition']['species_mode'],'legacy')

    def test_coarse_guard_is_frozen_and_preserved(self):
        val,cal=self.reports()
        val['metadata']['coarse_guard']=True
        with self.assertRaises(ValueError):
            make_profile({},val,cal,'model','manifest','profile')
        cal['metadata']['coarse_guard']=True
        result=make_profile(dict(species=dict(model_sha256='species'),insect=dict(model_sha256='insect')),
            val,cal,'model','manifest','profile')
        self.assertEqual(result['recognition']['coarse_guard'],{'confidence':90,'strong_presence':.95})

    def test_current_baseline_and_split_must_match_evaluated_artifacts(self):
        val,cal=self.reports()
        baseline=dict(species=dict(model_sha256='different'),insect=dict(model_sha256='insect'))
        with self.assertRaises(ValueError):
            make_profile(baseline,val,cal,'model','manifest','profile')
        baseline['species']['model_sha256']='species'
        with self.assertRaises(ValueError):
            make_profile(baseline,val,cal,'model','manifest','changed_threshold_profile')
        cal['metadata']['split_sha256']='different_split'
        with self.assertRaises(ValueError):
            make_profile(baseline,val,cal,'model','manifest','profile')


if __name__=='__main__':
    unittest.main()
