"""CPU regressions for detection supervision, validation and training failures."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.roi_heads import RoIHeads

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/train'))
from train_torchvision import JointDetectionTransform, collate_fn, train_one_epoch, validate
from train_detector_rebuild import RebuildDataset, bbox_only_evaluation, evaluate, train_epoch, validate_split, service_dedup


class JointTransformTests(unittest.TestCase):
    def test_forced_flip_moves_image_boxes_masks_without_mutating_target(self):
        transform = JointDetectionTransform(train=True, flip_probability=1)
        transform.color = torch.nn.Identity()
        image = torch.zeros((3, 4, 8))
        image[:, 1:3, 1:3] = 1
        masks = torch.zeros((1, 4, 8), dtype=torch.uint8)
        masks[:, 1:3, 1:3] = 1
        target = {'boxes': torch.tensor([[1., 1., 3., 3.]]), 'masks': masks}
        flipped, updated = transform(image, target)
        self.assertTrue(torch.equal(updated['boxes'], torch.tensor([[5., 1., 7., 3.]])))
        self.assertTrue(torch.equal(updated['masks'][0].bool(), flipped[0].bool()))
        self.assertTrue(torch.equal(target['boxes'], torch.tensor([[1., 1., 3., 3.]])))
        restored, restored_target = transform(flipped, updated)
        self.assertTrue(torch.equal(restored, image))
        self.assertTrue(torch.equal(restored_target['boxes'], target['boxes']))
        self.assertTrue(torch.equal(restored_target['masks'], target['masks']))

    def test_empty_target_stays_legal_after_flip(self):
        transform = JointDetectionTransform(train=True, flip_probability=1)
        transform.color = torch.nn.Identity()
        image, target = transform(Image.new('RGB', (8, 4)), {
            'boxes': torch.zeros((0, 4)), 'masks': torch.zeros((0, 4, 8), dtype=torch.uint8)})
        self.assertEqual(tuple(image.shape), (3, 4, 8))
        self.assertEqual(tuple(target['boxes'].shape), (0, 4))
        self.assertEqual(tuple(target['masks'].shape), (0, 4, 8))

    def test_validation_transform_is_not_geometric(self):
        image = torch.rand(3, 4, 8)
        target = {'boxes': torch.tensor([[1., 1., 3., 3.]]),
                  'masks': torch.ones((1, 4, 8), dtype=torch.uint8)}
        result, updated = JointDetectionTransform(train=False)(image, target)
        self.assertTrue(torch.equal(image, result))
        self.assertTrue(torch.equal(updated['boxes'], target['boxes']))


class LossModel(torch.nn.Module):
    def __init__(self, error=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.error = error

    def forward(self, images, targets):
        if self.error:
            raise self.error
        return {'loss': self.weight*images[0].mean()}


class TrainingFailureTests(unittest.TestCase):
    def batches(self, values):
        return [([torch.full((3, 2, 2), float(value))], [{'boxes': torch.zeros((0, 4))}])
                for value in values]

    def test_legacy_training_does_not_swallow_batch_exception(self):
        model = LossModel(RuntimeError('bad batch'))
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        with self.assertRaisesRegex(RuntimeError, 'bad batch'):
            train_one_epoch(model, optimizer, self.batches([1]), torch.device('cpu'), 1)

    def test_rebuild_training_does_not_swallow_batch_exception(self):
        model = LossModel(RuntimeError('bad batch'))
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        with self.assertRaisesRegex(RuntimeError, 'bad batch'):
            train_epoch(model, self.batches([1]), optimizer, scaler, torch.device('cpu'))

    def test_non_finite_loss_fails_instead_of_saving_success(self):
        model = LossModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        with self.assertRaises(FloatingPointError):
            train_one_epoch(model, optimizer, self.batches([float('nan')]), torch.device('cpu'), 1)

    def test_empty_training_is_not_zero_loss_success(self):
        model = LossModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        with self.assertRaises(ValueError):
            train_one_epoch(model, optimizer, [], torch.device('cpu'), 1)

    def test_last_short_gradient_window_uses_actual_window_size(self):
        model = LossModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        _, batches, skipped = train_epoch(model, self.batches([1, 3, 5]), optimizer, scaler,
                                  torch.device('cpu'), accumulate=2)
        self.assertEqual(batches, 3)
        # First update averages gradients 1 and 3; final singleton gradient is 5.
        self.assertAlmostEqual(model.weight.item(), .3, places=6)

    def test_partial_step_cap_does_not_read_an_extra_batch(self):
        batch = self.batches([1])[0]
        class Loader:
            def __len__(self):
                return 2
            def __iter__(self):
                yield batch
                raise RuntimeError('extra batch was read')
        model = LossModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        _, count, skipped = train_epoch(model, Loader(), optimizer, scaler, torch.device('cpu'),
                               accumulate=2, max_batches=1)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(model.weight.item(), .9, places=6)


class FakeValidationDataset(Dataset):
    rows = [dict(path='positive', group_id='p'), dict(path='empty_coco', group_id='e'),
            dict(path='folder_proxy', group_id='n', provenance='folder_proxy_non_tick', negative_label='aphid')]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        truth = [[0., 0., 2., 2.]] if index == 0 else []
        return torch.zeros((3, 2, 2)), {'boxes': torch.tensor(truth).reshape(-1, 4),
                                       'image_id': torch.tensor([index])}


class PredictionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.images_seen = 0

    def forward(self, images):
        self.images_seen += len(images)
        return [{'boxes': torch.tensor([[0., 0., 2., 2.]]), 'scores': torch.tensor([.9]),
                 'labels': torch.tensor([1])} for _ in images]


class MaskAwarePredictionModel(PredictionModel):
    def __init__(self, raise_error=False):
        super().__init__()
        self.roi_heads = torch.nn.Module()
        self.roi_heads.mask_roi_pool = torch.nn.Identity()
        self.roi_heads.mask_head = torch.nn.Linear(1, 1)
        self.roi_heads.mask_predictor = torch.nn.Linear(1, 1)
        self.raise_error = raise_error
        self.mask_branch_skipped = False

    def forward(self, images):
        if self.roi_heads.mask_roi_pool is not None:
            raise AssertionError('Unused full-resolution masks would be generated')
        self.mask_branch_skipped = True
        if self.raise_error:
            raise RuntimeError('validation forward failed')
        return super().forward(images)


class EvaluationTests(unittest.TestCase):
    def test_detection_selection_uses_service_overlap_deduplication(self):
        boxes,scores=service_dedup([[0.,0.,10.,10.],[4.,0.,14.,10.]], [.9,.8])
        self.assertEqual(boxes,[[0.,0.,10.,10.]])
        self.assertEqual(scores,[.9])

    def test_torchvision_mask_gate_is_disabled_only_inside_bbox_context(self):
        model = MaskAwarePredictionModel()
        self.assertTrue(RoIHeads.has_mask(model.roi_heads))
        with bbox_only_evaluation(model):
            self.assertFalse(RoIHeads.has_mask(model.roi_heads))
        self.assertTrue(RoIHeads.has_mask(model.roi_heads))

    def test_bbox_evaluation_disables_masks_and_restores_training_modes(self):
        for evaluator in (evaluate, validate):
            with self.subTest(evaluator=evaluator.__name__):
                model = MaskAwarePredictionModel()
                model.train()
                model.roi_heads.mask_head.eval()  # Preserve mixed modes too.
                pool = model.roi_heads.mask_roi_pool
                modes = {name: module.training for name, module in model.named_modules()}
                parameters = {name: p.requires_grad for name, p in model.named_parameters()}
                loader = DataLoader(FakeValidationDataset(), batch_size=1, collate_fn=collate_fn)
                evaluator(model, loader, torch.device('cpu'))
                self.assertTrue(model.mask_branch_skipped)
                self.assertEqual(model.images_seen, 3)
                self.assertIs(model.roi_heads.mask_roi_pool, pool)
                self.assertEqual({name: module.training for name, module in model.named_modules()}, modes)
                self.assertEqual({name: p.requires_grad for name, p in model.named_parameters()}, parameters)

    def test_bbox_evaluation_restores_mask_branch_even_after_forward_exception(self):
        for evaluator in (evaluate, validate):
            with self.subTest(evaluator=evaluator.__name__):
                model = MaskAwarePredictionModel(raise_error=True)
                model.train()
                model.roi_heads.mask_head.eval()
                pool = model.roi_heads.mask_roi_pool
                modes = {name: module.training for name, module in model.named_modules()}
                loader = DataLoader(FakeValidationDataset(), batch_size=1, collate_fn=collate_fn)
                with self.assertRaisesRegex(RuntimeError, 'validation forward failed'):
                    evaluator(model, loader, torch.device('cpu'))
                self.assertTrue(model.mask_branch_skipped)
                self.assertIs(model.roi_heads.mask_roi_pool, pool)
                self.assertEqual({name: module.training for name, module in model.named_modules()}, modes)

    def test_legacy_validation_evaluates_every_batch(self):
        model = PredictionModel()
        loader = DataLoader(FakeValidationDataset(), batch_size=1, collate_fn=collate_fn)
        metrics = validate(model, loader, torch.device('cpu'))
        self.assertEqual(model.images_seen, 3)
        self.assertEqual((metrics['tp'], metrics['fp'], metrics['fn']), (1, 2, 0))
        self.assertAlmostEqual(metrics['precision'], 1/3)

    def test_proxy_negative_metrics_are_separate_from_unverified_empty_coco(self):
        model = PredictionModel()
        loader = DataLoader(FakeValidationDataset(), batch_size=1, collate_fn=collate_fn)
        metrics, predictions = evaluate(model, loader, torch.device('cpu'))
        self.assertEqual(model.images_seen, 3)
        self.assertEqual(len(predictions), 3)
        self.assertEqual((metrics['tp'], metrics['fp'], metrics['fn']), (1, 1, 0))
        self.assertEqual(metrics['negative_proxy_images'], 1)
        self.assertEqual(metrics['negative_proxy_positive_images'], 1)
        self.assertEqual(metrics['count_mae'], .5)

    def test_group_leakage_is_rejected_before_loading_images(self):
        row = dict(path='x', group_id='g', annotations=[], width=8, height=4, image_id=1)
        split = dict(train=[row], val=[dict(row, path='other')], calibration=[])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_split(split)

    def test_negative_dataset_has_valid_empty_masks(self):
        row = dict(path='unused', width=8, height=4, annotations=[])
        dataset = RebuildDataset([row], train=False)
        with patch('train_detector_rebuild.Image.open', return_value=Image.new('RGB', (8, 4))):
            image, target = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 4, 8))
        self.assertEqual(tuple(target['masks'].shape), (0, 4, 8))
        self.assertEqual(tuple(target['boxes'].shape), (0, 4))

    def test_folder_proxy_exif_matches_manifest_without_rotating_coco(self):
        source = Image.new('RGB', (8, 4))
        source.getexif()[274] = 6
        row = dict(path='unused', width=4, height=8, annotations=[], negative_label='mite')
        with patch('train_detector_rebuild.Image.open', return_value=source):
            image, _ = RebuildDataset([row], train=False)[0]
        self.assertEqual(tuple(image.shape), (3, 8, 4))
        row.pop('negative_label')
        with patch('train_detector_rebuild.Image.open', return_value=Image.new('RGB', (8, 4))):
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                RebuildDataset([row], train=False)[0]


if __name__ == '__main__':
    unittest.main()
