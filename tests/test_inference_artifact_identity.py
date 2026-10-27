import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'web/backend'))
from app.main import artifact_sha256,verify_artifact,configure_detector,verified_whole_reference


class ArtifactIdentityTests(unittest.TestCase):
    def test_optional_whole_reference_is_verified_before_loading(self):
        self.assertIsNone(verified_whole_reference(dict(species={}),'/missing'))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'reference.pth';path.write_bytes(b'original whole model')
            profile=dict(species=dict(whole_reference=dict(model_sha256=artifact_sha256(path))))
            self.assertEqual(verified_whole_reference(profile,str(path)),str(path))
            path.write_bytes(b'different model')
            with self.assertRaises(ValueError):verified_whole_reference(profile,str(path))

    def test_evaluated_detector_configuration_is_applied_or_rejected(self):
        detector=SimpleNamespace(CONFIDENCE=.5,model=SimpleNamespace(roi_heads=SimpleNamespace(mask_roi_pool='pool')))
        config=dict(input_min_size=1024,input_max_size=1200,dedup_iou=.3,score_threshold=.3,bbox_only=True)
        configure_detector(detector,config)
        self.assertEqual(detector.CONFIDENCE,.3)
        self.assertIsNone(detector.model.roi_heads.mask_roi_pool)
        for key,value in [('dedup_iou',.5),('score_threshold',float('nan')),('input_min_size',800)]:
            with self.assertRaises(ValueError):
                configure_detector(detector,dict(config,**{key:value}))

    def test_same_filename_changed_bytes_cannot_reuse_model_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'best_weights.pth'
            path.write_bytes(b'original model bytes')
            expected=hashlib.sha256(path.read_bytes()).hexdigest()
            verify_artifact(path,expected)
            path.write_bytes(b'retrained model bytes')
            self.assertNotEqual(artifact_sha256(path),expected)
            with self.assertRaises(ValueError):
                verify_artifact(path,expected)


if __name__=='__main__':
    unittest.main()
