"""Leakage-controlled detector retraining from a local COCO pretrained cache.

Only train/val are used. Calibration image contents and historical test are never
read here. Partial pilots produce explicitly partial artifacts, not final models.
"""
import argparse
import hashlib
import json
import math
import random
import sys
from contextlib import contextmanager
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_torchvision import JointDetectionTransform, collate_fn
from training_safety import optimizer_step, validate_resume_directory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'web/backend'))
from app.services.detector import _dedup, DEDUP_IOU
DEFAULT_PRETRAINED = Path.home()/'.cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), 'utf-8')
    temporary.replace(path)


def validate_split(split):
    """Reject path/group overlap before loading any training or validation image."""
    seen_groups, seen_paths = {}, {}
    for subset in ('train', 'val', 'calibration'):
        rows = split[subset]
        if subset in ('train', 'val') and not rows:
            raise ValueError(f'{subset} is empty')
        for row in rows:
            for key in ('path', 'annotations', 'width', 'height', 'image_id', 'group_id'):
                if key not in row:
                    raise ValueError(f'Missing {key} in {subset} row')
            for seen, value in ((seen_groups, row['group_id']), (seen_paths, row['path'])):
                if value in seen and seen[value] != subset:
                    raise ValueError(f'{value}: overlap between {seen[value]} and {subset}')
                seen[value] = subset
            if row.get('negative_label') and row['annotations']:
                raise ValueError('Negative proxy row contains positive annotations')


class RebuildDataset(Dataset):
    def __init__(self, rows, root=ROOT, train=False):
        self.rows, self.root = rows, Path(root)
        self.transform = JointDetectionTransform(train=train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root/row['path']) as opened:
            # Folder proxies have no coordinate labels; their manifest dimensions
            # were measured after EXIF normalization. COCO labels remain raw.
            image = (ImageOps.exif_transpose(opened) if row.get('negative_label')
                     else opened).convert('RGB')
        width, height = image.size
        # Source COCO boxes/masks are in raw pixel coordinates. Do not apply EXIF
        # to only the image; reject metadata mismatches rather than shifting labels.
        if (width, height) != (row['width'], row['height']):
            raise ValueError(f'Image dimensions do not match annotations: {row["path"]}')
        boxes, masks = [], []
        for ann in row['annotations']:
            if ann['category_id'] != 1 or ann.get('iscrowd', 0):
                raise ValueError(f'Unsupported category/crowd annotation: {row["path"]}')
            x, y, w, h = ann['bbox']
            if not all(math.isfinite(float(v)) for v in (x, y, w, h)):
                raise ValueError('Non-finite annotation box')
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x+w), min(height, y+h)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f'Invalid annotation box: {row["path"]}')
            segmentation = ann.get('segmentation')
            if not segmentation:
                raise ValueError(f'Missing segmentation; do not silently invent mask labels: {row["path"]}')
            if isinstance(segmentation, list):
                rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
            elif isinstance(segmentation['counts'], list):
                rle = mask_utils.frPyObjects(segmentation, height, width)
            else:
                rle = dict(segmentation)
                if isinstance(rle['counts'], str):
                    rle['counts'] = rle['counts'].encode('ascii')
            mask = mask_utils.decode(rle)
            if mask.shape != (height, width) or not mask.any():
                raise ValueError(f'Invalid segmentation: {row["path"]}')
            boxes.append([x1, y1, x2, y2]); masks.append(mask)
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {'boxes': boxes, 'labels': torch.ones(len(boxes), dtype=torch.int64),
                  'masks': torch.tensor(np.stack(masks), dtype=torch.uint8) if masks else
                           torch.zeros((0, height, width), dtype=torch.uint8),
                  'image_id': torch.tensor([index]),
                  'area': (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]),
                  'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)}
        return self.transform(image, target)


def build_model(pretrained, freeze_low_stages):
    # Explicit FrozenBatchNorm matches COCO checkpoint architecture; no implicit
    # network download and no accidental initialization from an old tick model.
    backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=None,
                                  norm_layer=FrozenBatchNorm2d,
                                  trainable_layers=5-freeze_low_stages)
    model = MaskRCNN(backbone, num_classes=91, min_size=1024, max_size=1200)
    model.load_state_dict(torch.load(pretrained, map_location='cpu', weights_only=True))
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2)
    return model


def box_iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0.0


def match_detections(boxes, scores, truth, threshold=.5):
    remaining, tp = set(range(len(truth))), 0
    for index in sorted(range(len(boxes)), key=lambda i: -scores[i]):
        if not remaining:
            break
        best = max(remaining, key=lambda j: box_iou(boxes[index], truth[j]))
        if box_iou(boxes[index], truth[best]) >= threshold:
            remaining.remove(best); tp += 1
    return tp, len(boxes)-tp, len(truth)-tp


@contextmanager
def bbox_only_evaluation(model):
    """Temporarily skip the unused mask branch, preserving every module's mode.

    Torchvision's RoIHeads.has_mask() requires mask_roi_pool. Disabling that pool
    during bbox-only validation also prevents full-resolution mask postprocessing,
    whose allocation depends on original image size rather than model input size.
    This context is for the serial offline trainer, not concurrent serving.
    """
    modes = [(module, module.training) for module in model.modules()]
    roi_heads = getattr(model, 'roi_heads', None)
    has_mask_pool = roi_heads is not None and hasattr(roi_heads, 'mask_roi_pool')
    original_pool = roi_heads.mask_roi_pool if has_mask_pool else None
    try:
        model.eval()
        if has_mask_pool:
            roi_heads.mask_roi_pool = None
        yield
    finally:
        if has_mask_pool:
            roi_heads.mask_roi_pool = original_pool
        # Restore individual flags, including children intentionally in eval mode.
        for module, training in modes:
            module.training = training


@torch.inference_mode()
def evaluate(model, loader, device, score_threshold=.5):
    with bbox_only_evaluation(model):
        return _evaluate_bbox_outputs(model, loader, device, score_threshold)


def service_dedup(boxes, scores):
    retained = _dedup([dict(bbox=box,score=score) for box,score in zip(boxes,scores)])
    return [row['bbox'] for row in retained], [row['score'] for row in retained]


def _evaluate_bbox_outputs(model, loader, device, score_threshold):
    records = []
    tp = fp = fn = absolute_error = detection_images = 0
    negative_images = negative_positive = 0
    negative_by_label = {}
    for images, targets in loader:
        outputs = model([image.to(device) for image in images])
        for output, target in zip(outputs, targets):
            row = loader.dataset.rows[int(target['image_id'].item())]
            keep = (output['scores'] >= score_threshold) & (output['labels'] == 1)
            boxes = output['boxes'][keep].cpu().tolist()
            scores = output['scores'][keep].cpu().tolist()
            boxes, scores = service_dedup(boxes, scores)
            truth = target['boxes'].tolist()
            a, b, c = match_detections(boxes, scores, truth)
            if row.get('negative_label'):
                negative_images += 1; negative_positive += bool(boxes)
                label_stats = negative_by_label.setdefault(row['negative_label'], dict(images=0, positive_images=0))
                label_stats['images'] += 1; label_stats['positive_images'] += bool(boxes)
            else:
                detection_images += 1; tp += a; fp += b; fn += c
                absolute_error += abs(len(boxes)-len(truth))
            records.append(dict(path=row['path'], group_id=row['group_id'],
                                provenance=row.get('provenance'), negative_label=row.get('negative_label'),
                                boxes=boxes, scores=scores, truth=truth, tp=a, fp=b, fn=c))
    if not records or not detection_images:
        raise ValueError('Full validation requires nonempty detection images')
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    metrics = dict(images=len(records), detection_images=detection_images, tp=tp, fp=fp, fn=fn,
                   precision=precision, recall=recall, service_dedup_iou=DEDUP_IOU,
                   f1=2*precision*recall/(precision+recall) if precision+recall else 0.0,
                   count_mae=absolute_error/detection_images,
                   negative_proxy_images=negative_images,
                   negative_proxy_positive_images=negative_positive,
                   negative_proxy_positive_rate=negative_positive/negative_images if negative_images else None,
                   negative_proxy_by_label=negative_by_label,
                   score_threshold=score_threshold, matching_iou=.5)
    return metrics, records


def train_epoch(model, loader, optimizer, scaler, device, accumulate=1, max_batches=None):
    model.train()
    count = min(len(loader), max_batches) if max_batches is not None else len(loader)
    if count < 1:
        raise ValueError('No training batches')
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    skipped=consecutive=0
    for batch_index, (images, targets) in enumerate(islice(loader, count)):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        window_start = batch_index//accumulate*accumulate
        window_size = min(accumulate, count-window_start)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f'Non-finite loss at batch {batch_index}: {loss_dict}')
        scaler.scale(loss/window_size).backward()
        total_loss += loss.item()
        if (batch_index+1) % accumulate == 0 or batch_index+1 == count:
            parameters = [p for group in optimizer.param_groups for p in group['params']]
            finite=optimizer_step(parameters,optimizer,scaler,max_norm=float('inf'))
            consecutive=0 if finite else consecutive+1
            skipped+=int(not finite)
            if not finite:
                print(f'AMP skipped batch={batch_index+1}, scale={scaler.get_scale()}',flush=True)
            if consecutive>=5:
                raise FloatingPointError('Five consecutive AMP gradient overflows')
        if (batch_index+1) % 20 == 0:
            print(f'batch {batch_index+1}/{count} mean_loss={total_loss/(batch_index+1):.5f}', flush=True)
    return total_loss/count, count, skipped


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--pretrained', type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accumulate', type=int, default=2)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--freeze-low-stages', type=int, choices=range(6), default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--max-steps', type=int, default=0,
                        help='Cap training minibatches across this run; partial pilot artifacts only')
    parser.add_argument('--negative-penalty', type=float, default=.25,
                        help='Best selection: val F1 minus this coefficient times proxy positive rate')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.accumulate) < 1 or args.max_steps < 0 or args.workers < 0:
        parser.error('epochs/batch-size/accumulate must be positive; max-steps/workers nonnegative')
    if args.lr <= 0 or args.negative_penalty < 0:
        parser.error('lr must be positive and negative-penalty nonnegative')
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(4)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; explicitly select --device cpu if intended')
    device = torch.device(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    split = json.loads(args.split.read_text('utf-8'))
    validate_split(split)
    if not args.pretrained.is_file():
        raise FileNotFoundError(f'Local COCO pretrained cache missing: {args.pretrained}')
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(split_sha256=file_sha256(args.split), pretrained_sha256=file_sha256(args.pretrained),
                  min_size=1024, max_size=1200, class_names=['background', 'tick'],
                  service_dedup_iou=DEDUP_IOU,
                  service_detector_sha256=file_sha256(ROOT/'web/backend/app/services/detector.py'),
                  trainer_sha256=file_sha256(Path(__file__)),
                  training_safety_sha256=file_sha256(Path(__file__).with_name('training_safety.py')),
                  joint_transform_sha256=file_sha256(Path(__file__).with_name('train_torchvision.py')))
    # COCO cache identity prevents accidentally passing the deployed two-class v3.
    if config['pretrained_sha256'] != 'bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0':
        raise ValueError('Expected the verified torchvision COCO pretrained cache, not a tick checkpoint')
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(args.out.iterdir()):
        raise ValueError('--out must be empty for a new experiment; use --resume explicitly')
    model = build_model(args.pretrained, args.freeze_low_stages).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr, momentum=.9, weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=.1)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == 'cuda', init_scale=1024)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(RebuildDataset(split['train'], args.root, train=True),
                              batch_size=args.batch_size, shuffle=True, generator=generator,
                              num_workers=args.workers, collate_fn=collate_fn)
    val_loader = DataLoader(RebuildDataset(split['val'], args.root), batch_size=1,
                            shuffle=False, num_workers=args.workers, collate_fn=collate_fn)
    start_epoch, steps, best_score = 0, 0, -math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        prior_config = checkpoint['config']
        validate_resume_directory(prior_config,args.out,args.resume)
        for key in ('split_sha256', 'pretrained_sha256', 'min_size', 'max_size', 'batch_size',
                    'accumulate', 'freeze_low_stages', 'lr', 'negative_penalty', 'seed', 'workers', 'amp',
                    'device', 'max_steps',
                    'trainer_sha256', 'joint_transform_sha256', 'service_dedup_iou', 'service_detector_sha256',
                    'training_safety_sha256'):
            if prior_config[key] != config[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
        if checkpoint['partial']:
            raise ValueError('Partial pilot checkpoints cannot silently resume as full training')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer']); scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        generator.set_state(checkpoint['loader_rng'])
        torch.set_rng_state(checkpoint['torch_rng']); random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        start_epoch, steps, best_score = checkpoint['epoch']+1, checkpoint['steps'], checkpoint['best_score']
        if start_epoch >= args.epochs:
            raise ValueError('--epochs must extend beyond the resumed checkpoint epoch')
    write_json(args.out/'config.json', config)
    try:
        for epoch in range(start_epoch, args.epochs):
            remaining = args.max_steps-steps if args.max_steps else None
            if remaining is not None and remaining <= 0:
                break
            loss, batches, skipped = train_epoch(model, train_loader, optimizer, scaler, device,
                                         args.accumulate, remaining)
            steps += batches
            metrics, predictions = evaluate(model, val_loader, device)
            scheduler.step()
            partial = bool(args.max_steps)
            proxy_rate = metrics['negative_proxy_positive_rate'] or 0.0
            score = metrics['f1']-args.negative_penalty*proxy_rate
            improved = score > best_score
            best_score = max(best_score, score)
            record = dict(epoch=epoch, steps=steps, train_loss=loss, amp_skipped_updates=skipped, validation=metrics,
                          selection_score=score, partial=partial, best=improved)
            with (args.out/'epochs.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False)+'\n')
            write_json(args.out/f'val_epoch_{epoch+1:03d}.json', predictions)
            checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                              scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                              config=config, epoch=epoch, steps=steps, best_score=best_score,
                              partial=partial, metrics=metrics, loader_rng=generator.get_state(),
                              torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                              numpy_rng=np.random.get_state(),
                              cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
            torch.save(checkpoint, args.out/f'checkpoint_epoch_{epoch+1:03d}.pth')
            torch.save(checkpoint, args.out/'last_checkpoint.pth')
            if improved:
                torch.save(checkpoint, args.out/('best_partial_checkpoint.pth' if partial else 'best_checkpoint.pth'))
                torch.save(model.state_dict(), args.out/('best_partial_weights.pth' if partial else 'best_weights.pth'))
            print(json.dumps(record, ensure_ascii=False), flush=True)
    except Exception as error:
        write_json(args.out/'failure.json', dict(type=type(error).__name__, message=str(error),
                                                steps=steps, partial=bool(args.max_steps)))
        raise


if __name__ == '__main__':
    main()
