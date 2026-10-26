"""Train shared tick/species heads from local ImageNet weights on frozen groups."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'web/backend'))
from app.services.recognition_model import (RecognitionModel, prepare_image,
    letterbox_tensor, letterbox_image, recognition_loss, PREPROCESSING, MEAN)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), 'utf-8')
    tmp.replace(path)


def validate_resume_directory(prior_config, output, checkpoint):
    original=Path(prior_config['out']).resolve()
    if Path(output).resolve()!=original or Path(checkpoint).resolve().parent!=original:
        raise ValueError('Resume must use the original experiment directory and its checkpoint')


def optimizer_step(parameters, optimizer, scaler):
    parameters=list(parameters)
    scaler.unscale_(optimizer)
    norm=nn.utils.clip_grad_norm_(parameters,5,error_if_nonfinite=not scaler.is_enabled())
    finite=bool(torch.isfinite(norm))
    if finite:
        scaler.step(optimizer); scaler.update()
    else:
        # The loss can be finite while scaled FP16 gradients overflow. Never
        # update parameters in that case; record it and reduce the dynamic scale.
        scaler.update(new_scale=scaler.get_scale()*.5)
    optimizer.zero_grad(set_to_none=True)
    return finite


def validate_manifest(split):
    classes = split['classes']
    if len(classes) != len(set(classes)) or not classes:
        raise ValueError('Species classes must be nonempty and unique')
    seen = {}
    for subset in ('train', 'val', 'calibration'):
        if not split[subset]:
            raise ValueError(f'Empty {subset}')
        for row in split[subset]:
            for key in ('group_id', 'sha256'):
                token = key, row[key]
                if token in seen and seen[token] != subset:
                    raise ValueError('Cross-split identity')
                seen[token] = subset
            if row['tick_label'] not in (0,1):
                raise ValueError('Invalid tick label')
            label = row['species_label']
            if label is not None and (label not in classes or row['tick_label'] != 1):
                raise ValueError('Species label requires a tick and known species')
    return classes


class RecognitionDataset(Dataset):
    def __init__(self, rows, classes, root=ROOT, size=448, train=False):
        self.rows, self.root, self.size, self.train = rows, Path(root), size, train
        self.class_indices = {name:i for i,name in enumerate(classes)}
        self.augment = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
            T.RandomRotation(15, fill=tuple(round(v*255) for v in MEAN)),
            T.ColorJitter(brightness=.15, contrast=.15, saturation=.12)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root/row['path']) as opened:
            image = prepare_image(opened, row['bbox'],
                                  normalize_exif=row['coordinate_frame'] != 'raw')
        if self.train:
            # Augment the actual model-size view, not a multi-megapixel source.
            # This also preserves subjects at extreme aspect ratios before rotation.
            image = self.augment(letterbox_image(image, self.size))
        return (letterbox_tensor(image, self.size), row['tick_label'],
                self.class_indices.get(row['species_label'], -1), index)


def sampling_weights(rows):
    def bucket(row):
        return 'species' if row['species_label'] is not None else ('positive' if row['tick_label'] else 'negative')
    def label(row):
        return row['species_label'] or row.get('negative_label', row['sample_type'])
    frequencies = Counter((bucket(row), label(row)) for row in rows)
    raw = [1/math.sqrt(frequencies[bucket(row), label(row)]) for row in rows]
    totals = defaultdict(float)
    for row, weight in zip(rows, raw):
        totals[bucket(row)] += weight
    masses = dict(species=.45, positive=.15, negative=.40)
    return [weight*masses[bucket(row)]/totals[bucket(row)] for row, weight in zip(rows, raw)]


def metrics_from_predictions(rows, records, num_classes):
    tick = np.zeros((2,2), dtype=np.int64)
    species = np.zeros((num_classes,num_classes), dtype=np.int64)
    for record in records:
        tick[record['tick_label'], int(record['tick_probability'] >= .5)] += 1
        if record['species_label'] >= 0:
            species[record['species_label'], record['species_prediction']] += 1
    tp, fn, fp, tn = int(tick[1,1]), int(tick[1,0]), int(tick[0,1]), int(tick[0,0])
    recall = tp/(tp+fn) if tp+fn else None
    specificity = tn/(tn+fp) if tn+fp else None
    denominator = species.sum(0)+species.sum(1)
    f1 = np.divide(2*species.diagonal(), denominator, out=np.zeros(num_classes), where=denominator>0)
    supported = species.sum(1)>0
    return dict(images=len(records), tick_tp=tp,tick_fn=fn,tick_fp=fp,tick_tn=tn,
        tick_recall=recall, non_tick_fpr=1-specificity if specificity is not None else None,
        balanced_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None,
        species_images=int(species.sum()), species_top1=float(species.trace()/species.sum()) if species.sum() else None,
        species_macro_f1=float(f1[supported].mean()) if supported.any() else None,
        species_supported_classes=int(supported.sum()))


@torch.inference_mode()
def evaluate(model, loader, device, rows, num_classes):
    model.eval()
    records = []
    for images, tick, species, indices in loader:
        output = model(images.to(device))
        tick_prob = output['tick_logits'].softmax(1)[:,1].cpu().tolist()
        species_prob = output['species_logits'].softmax(1).cpu()
        for j, index in enumerate(indices.tolist()):
            records.append(dict(index=index, tick_label=int(tick[j]), species_label=int(species[j]),
                tick_probability=tick_prob[j], species_prediction=int(species_prob[j].argmax()),
                species_confidence=float(species_prob[j].max()), sample_type=rows[index]['sample_type']))
    metrics = metrics_from_predictions(rows, records, num_classes)
    metrics['by_sample_type'] = {kind:metrics_from_predictions(rows,
        [r for r in records if r['sample_type']==kind], num_classes) for kind in sorted({r['sample_type'] for r in records})}
    return metrics, records


def train_epoch(model, loader, optimizer, scaler, device, accumulate, max_steps=0):
    model.train()
    # Small mixed-source batches must not rewrite ImageNet BN running statistics.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    count = min(len(loader), max_steps) if max_steps else len(loader)
    optimizer.zero_grad(set_to_none=True)
    total = 0
    skipped=consecutive=0
    for step, (images, tick, species, _) in enumerate(loader):
        if step >= count:
            break
        images, tick, species = images.to(device), tick.to(device), species.to(device)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            loss = recognition_loss(model(images), tick, species)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite loss at batch {step}')
        window = min(accumulate, count-(step//accumulate)*accumulate)
        scaler.scale(loss/window).backward()
        total += loss.item()
        if (step+1)%accumulate == 0 or step+1 == count:
            finite=optimizer_step(model.parameters(),optimizer,scaler)
            consecutive=0 if finite else consecutive+1
            skipped+=int(not finite)
            if not finite:
                print(f'AMP skipped optimizer update at batch={step+1}, scale={scaler.get_scale()}',flush=True)
            if consecutive>=5:
                raise FloatingPointError('Five consecutive AMP gradient overflows; investigate inputs and model')
        if (step+1)%50 == 0:
            print(f'batch={step+1}/{count} loss={total/(step+1):.5f}', flush=True)
    return total/count, count, skipped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pretrained', type=Path, default=Path.home()/'.cache/torch/hub/checkpoints/resnet50-0676ba61.pth')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--accumulate', type=int, default=2)
    p.add_argument('--size', type=int, default=448)
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    if min(args.epochs,args.batch_size,args.accumulate,args.size,args.patience) < 1 or args.max_steps < 0:
        p.error('Positive epochs, batch, accumulate, size, patience required')
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    split = json.loads(args.split.read_text('utf-8'))
    classes = validate_manifest(split)
    config = {k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(split_sha256=sha256(args.split), pretrained_sha256=sha256(args.pretrained),
        trainer_sha256=sha256(Path(__file__)), model_sha256=sha256(ROOT/'web/backend/app/services/recognition_model.py'),
        preprocessing=PREPROCESSING, architecture='resnet50_tick2_species_v1',
        source_limitations=split.get('metadata',{}))
    if not config['pretrained_sha256'].startswith('0676ba61'):
        raise ValueError('Expected verified generic ImageNet ResNet50 checkpoint')
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(args.out.iterdir()):
        raise ValueError('New experiment requires an empty output directory')
    model = RecognitionModel(len(classes))
    pretrained = torch.load(args.pretrained, map_location='cpu', weights_only=True)
    pretrained.pop('fc.weight'); pretrained.pop('fc.bias')
    model.encoder.load_state_dict(pretrained, strict=True)
    for name, parameter in model.encoder.named_parameters():
        parameter.requires_grad = name.startswith(('layer3.', 'layer4.'))
    model.to(device)
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in model.encoder.parameters() if p.requires_grad], lr=args.lr*.1),
        dict(params=list(model.tick_head.parameters())+list(model.species_head.parameters()), lr=args.lr)], weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*.01)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type=='cuda', init_scale=1024)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(sampling_weights(split['train']), len(split['train']), generator=generator)
    train = DataLoader(RecognitionDataset(split['train'],classes,size=args.size,train=True),
        batch_size=args.batch_size,sampler=sampler,num_workers=args.workers)
    val = DataLoader(RecognitionDataset(split['val'],classes,size=args.size),
        batch_size=args.batch_size,num_workers=args.workers)
    start, best, stale = 0, -math.inf, 0
    if args.resume:
        checkpoint = torch.load(args.resume,map_location='cpu',weights_only=False)
        validate_resume_directory(checkpoint['metadata'],args.out,args.resume)
        for key in config:
            if key in ('resume','out'):
                continue
            if checkpoint['metadata'][key] != config[key]:
                raise ValueError(f'Resume config mismatch: {key}')
        if checkpoint['partial']:
            raise ValueError('Partial pilot cannot resume as formal training')
        model.load_state_dict(checkpoint['model_state']); optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']); scaler.load_state_dict(checkpoint['scaler'])
        generator.set_state(checkpoint['loader_rng']); torch.set_rng_state(checkpoint['torch_rng'])
        np.random.set_state(checkpoint['numpy_rng']); random.setstate(checkpoint['python_rng'])
        if device.type=='cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        start, best, stale = checkpoint['epoch']+1, checkpoint['best_score'], checkpoint['stale']
    save_json(args.out/'config.json',config)
    try:
        for epoch in range(start,args.epochs):
            begin = time.monotonic()
            loss, steps, skipped = train_epoch(model,train,optimizer,scaler,device,args.accumulate,args.max_steps)
            metrics, records = evaluate(model,val,device,split['val'],len(classes))
            score = (metrics['balanced_accuracy']+metrics['species_macro_f1'])/2
            improved = score > best
            best, stale = max(best,score), 0 if improved else stale+1
            scheduler.step()
            record = dict(epoch=epoch,train_loss=loss,steps=steps,amp_skipped_updates=skipped,validation=metrics,
                selection_score=score,best=improved,partial=bool(args.max_steps),seconds=time.monotonic()-begin)
            with (args.out/'epochs.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
            save_json(args.out/f'val_epoch_{epoch+1:03d}.json',records)
            checkpoint = dict(model_state=model.state_dict(),classes=classes,
                architecture=config['architecture'],input_size=args.size,preprocessing=PREPROCESSING,metadata=config,
                epoch=epoch,best_score=best,stale=stale,partial=bool(args.max_steps),
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                loader_rng=generator.get_state(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
                python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
            torch.save(checkpoint,args.out/'last_checkpoint.pth')
            if improved:
                torch.save(checkpoint,args.out/('best_partial_checkpoint.pth' if args.max_steps else 'best_checkpoint.pth'))
            print(json.dumps(record,ensure_ascii=False),flush=True)
            if args.max_steps or stale>=args.patience:
                break
    except Exception as error:
        save_json(args.out/'failure.json',dict(type=type(error).__name__,message=str(error)))
        raise


if __name__=='__main__':
    main()
