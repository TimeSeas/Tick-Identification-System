import sys
from pathlib import Path
import unittest

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/train'))
from train_recognition_rebuild import (sampling_weights, metrics_from_predictions, validate_manifest)
from app.services.recognition_model import prepare_image, letterbox_tensor, recognition_loss


class RecognitionTests(unittest.TestCase):
    def test_negative_only_batch_has_finite_loss_and_zero_species_gradient(self):
        tick = torch.randn(4,2,requires_grad=True)
        species = torch.randn(4,28,requires_grad=True)
        loss = recognition_loss(dict(tick_logits=tick,species_logits=species),torch.zeros(4,dtype=torch.long),
                                torch.full((4,),-1,dtype=torch.long))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(species.grad.abs().sum().item(),0)
        self.assertGreater(tick.grad.abs().sum().item(),0)

    def test_unknown_species_does_not_train_species_head(self):
        species = torch.randn(2,3,requires_grad=True)
        loss = recognition_loss(dict(tick_logits=torch.randn(2,2,requires_grad=True),species_logits=species),
                                torch.tensor([1,1]),torch.tensor([2,-1]))
        loss.backward()
        self.assertGreater(species.grad[0].abs().sum().item(),0)
        self.assertEqual(species.grad[1].abs().sum().item(),0)

    def test_letterbox_preserves_extreme_aspect_ratio_subject(self):
        image = Image.new('RGB',(200,20),'red')
        tensor = letterbox_tensor(image,100)
        self.assertEqual(tuple(tensor.shape),(3,100,100))
        self.assertLess(tensor[:,0,:].abs().max(),.02)
        self.assertGreater(tensor[0,50,:].min(),2)

    def test_exif_and_raw_box_coordinate_contract(self):
        image = Image.new('RGB',(8,4))
        image.getexif()[274] = 6
        self.assertEqual(prepare_image(image).size,(4,8))
        self.assertEqual(prepare_image(image,[0,0,8,4],context=0,normalize_exif=False).size,(8,4))
        with self.assertRaises(ValueError):
            prepare_image(image,[2,2,1,1])

    def test_sampler_gives_requested_bucket_masses_despite_class_imbalance(self):
        rows = [dict(species_label='a',tick_label=1,sample_type='species_full')]*100
        rows += [dict(species_label=None,tick_label=1,sample_type='coco_tick_roi')]*10
        rows += [dict(species_label=None,tick_label=0,sample_type='non_tick_full',negative_label='mite')]*1000
        weights = sampling_weights(rows)
        self.assertAlmostEqual(sum(weights[:100]),.45)
        self.assertAlmostEqual(sum(weights[100:110]),.15)
        self.assertAlmostEqual(sum(weights[110:]),.40)

    def test_binary_failure_not_hidden_by_species_accuracy(self):
        records = [dict(tick_label=0,tick_probability=.9,species_label=-1,species_prediction=0),
                   dict(tick_label=1,tick_probability=.9,species_label=1,species_prediction=1)]
        result = metrics_from_predictions([],records,2)
        self.assertEqual(result['non_tick_fpr'],1)
        self.assertEqual(result['balanced_accuracy'],.5)
        self.assertEqual(result['species_top1'],1)

    def test_manifest_rejects_cross_task_group_leak(self):
        row = dict(group_id='g',sha256='a',tick_label=1,species_label='tick_a')
        split = dict(classes=['tick_a'],train=[row],val=[dict(row,sha256='b')],calibration=[dict(row,group_id='h',sha256='c')])
        with self.assertRaisesRegex(ValueError,'Cross-split'):
            validate_manifest(split)


if __name__=='__main__':
    unittest.main()
