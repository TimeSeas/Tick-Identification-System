import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src/train'))
from prepare_species_specialist import build_unknown_assignment,validate_specialist_manifest
from train_species_specialist import specialist_loss,validate_frozen_images,validate_resume_epoch


class SpecialistDataTests(unittest.TestCase):
    def test_changed_or_conflicting_image_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);path=root/'image';path.write_bytes(b'original')
            row=dict(path='image',sha256=hashlib.sha256(b'original').hexdigest())
            manifest=dict(train=[row],val=[dict(row)])
            validate_frozen_images(manifest,root)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_frozen_images(manifest,root)
            manifest['val'][0]['sha256']='conflict'
            with self.assertRaises(ValueError):validate_frozen_images(manifest,root)

    def test_resume_rejects_earlier_epoch_or_unjournalled_later_artifact(self):
        with tempfile.TemporaryDirectory() as name:
            out=Path(name)
            (out/'epochs.jsonl').write_text('\n'.join(json.dumps(dict(epoch=i)) for i in range(1,4)),'utf-8')
            (out/'checkpoint_epoch_003.pth').touch()
            validate_resume_epoch(out,2)
            with self.assertRaises(ValueError):validate_resume_epoch(out,0)
            (out/'val_epoch_004.json').touch()
            with self.assertRaises(ValueError):validate_resume_epoch(out,2)

    def manifest(self):
        result=dict(classes=['known_a','known_b'])
        for number,subset in enumerate(('train','val','calibration','holdout')):
            result[subset]=[dict(path=f'{subset}.jpg',sha256=str(number),group_id=str(number),
                species_label=None if subset=='holdout' else 'known_a',
                source_species='unknown_d' if subset=='holdout' else 'known_a',bbox=None)]
        return result

    def test_unknown_assignment_is_reproducible_and_species_disjoint(self):
        labels=[f'unknown_{i}' for i in range(30)]
        mapping=build_unknown_assignment(labels,42)
        self.assertEqual(mapping,build_unknown_assignment(list(reversed(labels)),42))
        self.assertEqual(set(mapping),set(labels))
        self.assertEqual(set(mapping.values()),{'train','val','calibration','holdout'})

    def test_cross_split_hash_or_group_rejected(self):
        for key in ('sha256','group_id'):
            manifest=self.manifest(); manifest['val'][0][key]=manifest['train'][0][key]
            with self.assertRaises(ValueError):validate_specialist_manifest(manifest)

    def test_unknown_species_cannot_cross_splits_or_be_given_known_label(self):
        manifest=self.manifest(); validate_specialist_manifest(manifest)
        manifest['train'][0].update(species_label=None,source_species='unknown_d')
        with self.assertRaises(ValueError):validate_specialist_manifest(manifest)
        manifest=self.manifest();manifest['train'][0]['species_label']='unrecognized'
        with self.assertRaises(ValueError):validate_specialist_manifest(manifest)


class SpecialistLossTests(unittest.TestCase):
    def test_unknown_uniform_loss_reduces_confident_known_prediction(self):
        logits=torch.tensor([[4.,0.,0.]],requires_grad=True)
        loss=specialist_loss(logits,torch.zeros_like(logits),torch.tensor([-1]))
        loss.backward()
        self.assertGreater(logits.grad[0,0].item(),0)
        self.assertLess(logits.grad[0,1].item(),0)

    def test_teacher_has_no_gradient_and_mixed_or_known_batches_finite(self):
        for labels in (torch.tensor([0,1]),torch.tensor([0,-1])):
            logits=torch.randn(2,3,requires_grad=True);teacher=torch.randn(2,3,requires_grad=True)
            loss=specialist_loss(logits,teacher,labels);loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertIsNone(teacher.grad)
            self.assertTrue(torch.isfinite(logits.grad).all())


if __name__=='__main__':unittest.main()
