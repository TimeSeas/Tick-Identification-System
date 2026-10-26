import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'web/backend'))
from app.main import artifact_sha256,verify_artifact


class ArtifactIdentityTests(unittest.TestCase):
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
