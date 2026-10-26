import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_detector_profile import build_profile
from build_recognition_profile import digest


class DetectorProfileTests(unittest.TestCase):
    def test_detection_threshold_is_bound_to_evaluated_run(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name); base,weights,_=self.prepare(folder)
            metadata_path=folder/'metadata.json'; report_path=folder/'report.json'
            metadata=json.loads(metadata_path.read_text('utf-8')); metadata['score_threshold']=.3
            metadata['cache_helper_sha256']='threshold-aware helper source'
            metadata_path.write_text(json.dumps(metadata),'utf-8')
            report=json.loads(report_path.read_text('utf-8')); report['metadata']=metadata
            report_path.write_text(json.dumps(report),'utf-8')
            self.assertEqual(build_profile(folder,'regional_only',weights,base)['detector']['score_threshold'],.3)
            metadata['proposal_weights_sha256']='different proposal artifact'
            metadata_path.write_text(json.dumps(metadata),'utf-8')
            with self.assertRaises(ValueError):
                build_profile(folder,'regional_only',weights,base)

    def prepare(self,folder):
        base=folder/'baseline.json'; weights=folder/'weights.pth'
        baseline=dict(species=dict(model_sha256='species'),insect=dict(model_sha256='insect'))
        base.write_text(json.dumps(baseline),'utf-8'); weights.write_bytes(b'actual detector artifact')
        metadata=dict(old_profile_sha256=digest(base),weights_sha256=digest(weights),
            species_sha256='species',insect_sha256='insect')
        (folder/'metadata.json').write_text(json.dumps(metadata),'utf-8')
        profile=copy.deepcopy(baseline); profile['rejection']=dict(whole_non_tick=101)
        policy=folder/'regional_only_profile.json'; policy.write_text(json.dumps(profile),'utf-8')
        report=dict(metadata=metadata,variants=dict(regional_only=dict(profile_sha256=digest(policy),
            gates=dict(recall=True,proxy_fp=True,count=True))))
        (folder/'report.json').write_text(json.dumps(report),'utf-8')
        return base,weights,policy

    def test_preserves_classifiers_and_binds_actual_detector_and_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); base,weights,policy=self.prepare(folder)
            profile=build_profile(folder,'regional_only',weights,base)
            self.assertEqual(profile['detector']['model_sha256'],digest(weights))
            self.assertEqual(profile['species']['model_sha256'],'species')
            policy.write_text('{}','utf-8')
            with self.assertRaises(ValueError):
                build_profile(folder,'regional_only',weights,base)

    def test_changed_detector_or_failed_gate_cannot_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); base,weights,_=self.prepare(folder)
            weights.write_bytes(b'different checkpoint')
            with self.assertRaises(ValueError):
                build_profile(folder,'regional_only',weights,base)
            self.prepare(folder)
            report=json.loads((folder/'report.json').read_text('utf-8'))
            report['variants']['regional_only']['gates']['recall']=False
            (folder/'report.json').write_text(json.dumps(report),'utf-8')
            with self.assertRaises(ValueError):
                build_profile(folder,'regional_only',weights,base)


if __name__=='__main__':
    unittest.main()
