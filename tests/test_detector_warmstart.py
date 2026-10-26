from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/train'))
from train_detector_rebuild import load_warm_start,file_sha256


class WarmStartTests(unittest.TestCase):
    def test_explicit_artifact_and_complete_state_required(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'deployed.pth'
            original=torch.nn.Linear(3,2)
            torch.save(original.state_dict(),path)
            target=torch.nn.Linear(3,2)
            with self.assertRaises(ValueError):
                load_warm_start(target,path,'wrong-artifact')
            load_warm_start(target,path,file_sha256(path))
            self.assertTrue(torch.equal(target.weight,original.weight))
            torch.save({'weight':original.weight},path)
            with self.assertRaises(RuntimeError):
                load_warm_start(target,path,file_sha256(path))


if __name__=='__main__':
    unittest.main()
