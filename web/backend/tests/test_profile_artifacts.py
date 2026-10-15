"""A calibrated profile must not silently accept a replacement model."""
import hashlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import verify_artifact


class ArtifactProfileTests(unittest.TestCase):
    def test_matching_artifact_is_accepted(self):
        content = b'calibrated model contents'
        with patch('builtins.open', return_value=io.BytesIO(content)):
            verify_artifact('model.pth', hashlib.sha256(content).hexdigest())

    def test_replaced_artifact_requires_recalibration(self):
        with patch('builtins.open', return_value=io.BytesIO(b'replacement model')):
            with self.assertRaisesRegex(ValueError, '重新评估并校准'):
                verify_artifact('model.pth', hashlib.sha256(b'original model').hexdigest())


if __name__ == '__main__':
    unittest.main()
