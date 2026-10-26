"""Freeze full/region views, preserving source groups and original labels.

Teacher boxes are proposals, not new human annotations. Folder negatives remain
proxy labels. Does not read any historical test image or change input manifests.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'web/backend'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), 'utf-8')
    tmp.replace(path)


def base_row(row, kind, tick, species=None, bbox=None, raw=False):
    value = {key: row[key] for key in ('path', 'group_id', 'sha256')}
    value.update(sample_type=kind, tick_label=tick, species_label=species,
                 bbox=bbox, coordinate_frame='raw' if raw else 'exif_normalized')
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--teacher', type=Path, default=ROOT/'output/det_v3/mask_rcnn_ticks_v3.pth')
    p.add_argument('--negative-per-class', type=int, default=20)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = {name: args.data/f'{name}_split.json' for name in ('species', 'insect', 'detector')}
    data = {name: json.loads(path.read_text('utf-8')) for name, path in sources.items()}
    config = dict(inputs={name: digest(path) for name, path in sources.items()},
                  teacher_sha256=digest(args.teacher), negative_per_class=args.negative_per_class,
                  script_sha256=digest(Path(__file__)), threshold=.7, max_hard_boxes=2)
    cache_path = args.out/'proposals.json'
    cache = json.loads(cache_path.read_text('utf-8')) if cache_path.exists() else dict(config=config, boxes={})
    if cache['config'] != config:
        raise ValueError('Proposal cache configuration changed; use a new output directory')
    from app.services.detector import Detector
    import torch
    torch.set_num_threads(4)
    teacher = Detector(str(args.teacher))
    # This private instance is used only for boxes, never service inference/masks.
    teacher.model.roi_heads.mask_roi_pool = None
    teacher.CONFIDENCE = .7
    output = dict(classes=sorted({r['label'] for r in data['species']['train']}), metadata=config)
    processed = 0

    def propose(row):
        nonlocal processed
        key = row['sha256']
        if key not in cache['boxes']:
            with Image.open(ROOT/row['path']) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                boxes = teacher.detect(np.array(image))
                cache['boxes'][key] = [b for b in boxes if
                    min(b['bbox'][2]-b['bbox'][0], b['bbox'][3]-b['bbox'][1]) >= 16]
            processed += 1
            if processed % 20 == 0:
                save(cache_path, cache)
                print(f'proposals cached={len(cache["boxes"])}', flush=True)
        return cache['boxes'][key]

    for subset in ('train', 'val', 'calibration'):
        rows = []
        for row in data['species'][subset]:
            rows.append(base_row(row, 'species_full', 1, row['label']))
            boxes = propose(row)
            if len(boxes) == 1 and boxes[0]['score'] >= .8:
                rows.append(base_row(row, 'species_roi', 1, row['label'], boxes[0]['bbox']))
        negative_counts = Counter()
        for row in data['insect'][subset]:
            tick = int(row['label'] == 'tick')
            item = base_row(row, 'insect_tick_full' if tick else 'non_tick_full', tick)
            if not tick:
                item['negative_label'] = row['label']
            rows.append(item)
            # Freeze an ordered class-balanced subset, not a test-driven sample.
            limit = args.negative_per_class if subset == 'train' else min(8, args.negative_per_class)
            if not tick and negative_counts[row['label']] < limit:
                negative_counts[row['label']] += 1
                for proposal in propose(row)[:2]:
                    item = base_row(row, 'non_tick_hard_roi', 0, bbox=proposal['bbox'])
                    item.update(negative_label=row['label'], label_quality='folder_proxy',
                                teacher_score=proposal['score'])
                    rows.append(item)
        for row in data['detector'][subset]:
            for ann in row['annotations']:
                if ann.get('iscrowd') or ann['category_id'] != 1:
                    continue
                x, y, w, h = ann['bbox']
                rows.append(base_row(row, 'coco_tick_roi', 1, bbox=[x,y,x+w,y+h], raw=True))
        # Exact full/box duplicate views do not get extra sampling weight.
        unique = {(r['path'], tuple(r['bbox']) if r['bbox'] else None, r['sample_type']):r for r in rows}
        output[subset] = list(unique.values())
        print(subset, dict(Counter(r['sample_type'] for r in output[subset])), flush=True)
        save(cache_path, cache)
    seen = {}
    for subset in ('train', 'val', 'calibration'):
        for row in output[subset]:
            for key in ('group_id', 'sha256'):
                token = key, row[key]
                if token in seen and seen[token] != subset:
                    raise ValueError('Cross-split identity in recognition manifest')
                seen[token] = subset
    save(args.out/'recognition_split.json', output)
    save(args.out/'summary.json', dict(config=config, split_sha256=digest(args.out/'recognition_split.json'),
        counts={s:dict(Counter(r['sample_type'] for r in output[s])) for s in ('train','val','calibration')},
        limitations=['teacher proposals not human boxes', 'folder negatives are proxies',
                     'multiple views in same split are correlated']))


if __name__ == '__main__':
    main()
