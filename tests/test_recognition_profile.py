import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_recognition_profile import make_profile


class ProfileTests(unittest.TestCase):
    def reports(self):
        val=dict(gates=dict(recall=True,false_positive=True),metadata=dict(subset='val',checkpoint_sha256='model',
            new_detection_threshold=.5,tick_threshold=.95,detector_sha256='det',old_species_sha256='species',old_insect_sha256='insect'))
        cal=copy.deepcopy(val); cal['metadata']['subset']='calibration'
        return val,cal

    def test_failed_or_changed_policy_cannot_create_default_profile(self):
        val,cal=self.reports(); cal['gates']['recall']=False
        with self.assertRaises(ValueError):
            make_profile({},val,cal,'model','manifest')
        val,cal=self.reports(); cal['metadata']['tick_threshold']=.2
        with self.assertRaises(ValueError):
            make_profile({},val,cal,'model','manifest')

    def test_new_presence_keeps_original_species_and_original_profile(self):
        val,cal=self.reports()
        baseline=dict(species=dict(model_sha256='species'))
        result=make_profile(baseline,val,cal,'model','manifest')
        self.assertNotIn('recognition',baseline)
        self.assertEqual(result['species'],baseline['species'])
        self.assertEqual(result['recognition']['species_mode'],'legacy')

    def test_coarse_guard_is_frozen_and_preserved(self):
        val,cal=self.reports()
        val['metadata']['coarse_guard']=True
        with self.assertRaises(ValueError):
            make_profile({},val,cal,'model','manifest')
        cal['metadata']['coarse_guard']=True
        result=make_profile({},val,cal,'model','manifest')
        self.assertEqual(result['recognition']['coarse_guard'],{'confidence':90,'strong_presence':.95})


if __name__=='__main__':
    unittest.main()
