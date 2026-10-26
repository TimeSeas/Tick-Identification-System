"""Build new manifests without changing images or historical splits (CPU only)."""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

from PIL import Image, ImageOps

SPLITS = ('train', 'val', 'calibration')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Groups:
    def __init__(self):
        self.parents = {}

    def find(self, item):
        self.parents.setdefault(item, item)
        if self.parents[item] != item:
            self.parents[item] = self.find(self.parents[item])
        return self.parents[item]

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        self.parents[max(a, b)] = min(a, b)


def stable_order(value, seed):
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()


def assign_groups(rows, seed=42):
    """Stratify connected groups jointly across tasks; never split a group."""
    strata = collections.defaultdict(set)
    for row in rows:
        strata[(row['task'], row['label'])].add(row['group_id'])
    assigned = {}
    for label, ids in sorted(strata.items(), key=lambda pair: (len(pair[1]), pair[0])):
        if len(ids) < 3:
            raise ValueError(f'{label} has fewer than three independent groups')
        target = max(1, round(len(ids) * .125))
        ordered = sorted(ids, key=lambda x: stable_order(x, seed))
        for split in ('val', 'calibration'):
            needed = target - sum(assigned.get(g) == split for g in ids)
            available = [g for g in ordered if g not in assigned]
            for group in available[:max(0, needed)]:
                assigned[group] = split
        # Small overlapping strata must still have a training group.
        if not any(assigned.get(g, 'train') == 'train' for g in ids):
            raise ValueError(f'Overlapping strata leave no training group for {label}')
    return {g: assigned.get(g, 'train') for ids in strata.values() for g in ids}


def assert_disjoint(manifests):
    seen = {}
    for task, manifest in manifests.items():
        for split, rows in manifest.items():
            for row in rows:
                for key in ('group_id', 'sha256'):
                    identity = (key, row[key])
                    if identity in seen and seen[identity] != split:
                        raise AssertionError(f'{task}: {identity} crosses splits')
                    seen[identity] = split
    return True


def image_fingerprint(path):
    """EXIF-normalized 64-bit dHash and quality metadata; no model inference."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        width, height = image.size
        thumbnail = image.resize((9, 8)).convert('L')
        pixels = [thumbnail.getpixel((x, y)) for y in range(8) for x in range(9)]
        value = 0
        for y in range(8):
            for x in range(8):
                value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
        return value, width, height


def near_matches(value, index, max_distance=4):
    candidates = set()
    for offset in range(8):
        candidates.update(index.get((offset, (value >> (offset * 8)) & 255), ()))
    return [candidate for candidate in candidates
            if (candidate ^ value).bit_count() <= max_distance]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, default=Path('output/optimization_20261015/rebuild_20261024/data'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out if args.out.is_absolute() else root / args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output directory must be empty; preserve previous manifests')
    audit_path = root / 'output/optimization_20261015/rebuild_20261024/data_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    for path, expected in audit['source_files'].items():
        if sha256(root / path) != expected:
            raise ValueError(f'Input changed since audit: {path}')
    groups = Groups()
    rows, hashes, quality, excluded = [], {}, {}, []
    for task, source in [('species', 'data/tick_species/split_clean.json'),
                         ('insect', 'data/insects_split.json')]:
        data = json.loads((root / source).read_text(encoding='utf-8'))
        for split, items in data.items():
            rows.extend(dict(item, task=task, original_split=split, source=source)
                        for item in items)
    for split in ('train', 'valid', 'test'):
        source = f'ticks image detection.v5i.coco/{split}/_annotations.coco.json'
        data = json.loads((root / source).read_text(encoding='utf-8'))
        annotations = collections.defaultdict(list)
        for ann in data['annotations']:
            annotations[ann['image_id']].append(ann)
        for image in data['images']:
            rows.append(dict(task='detector', original_split=split, source=source,
                path=f"ticks image detection.v5i.coco/{split}/{image['file_name']}",
                label='tick' if annotations[image['id']] else 'empty',
                annotations=annotations[image['id']], width=image['width'],
                height=image['height'], image_id=image['id'],
                provenance='original_coco' if annotations[image['id']] else 'unverified_coco_empty'))
    first_hash = {}
    for i, row in enumerate(rows):
        path = row['path']
        groups.find(path)
        digest = hashes.setdefault(path, sha256(root / path))
        row['sha256'] = digest
        if digest in first_hash:
            groups.union(path, first_hash[digest])
        first_hash[digest] = path
        try:
            quality[path] = image_fingerprint(root / path)
        except (OSError, ValueError) as error:
            quality[path] = None
            excluded.append(dict(path=path, reason='decode_failed', detail=str(error)))
        if (i + 1) % 2000 == 0:
            print(f'Fingerprint {i + 1}/{len(rows)}', flush=True)
    first_photo, first_observation = {}, {}
    for path, metas in audit['photo_id_audit']['path_provenance'].items():
        for meta in metas:
            photo = meta.get('photo_id')
            match = re.search(r'inaturalist.org/observations/(\d+)', meta.get('observation_reference', ''))
            for identity, table in [(photo, first_photo),
                                    (match.group(1) if match else None, first_observation)]:
                if identity:
                    if identity in table:
                        groups.union(path, table[identity])
                    table[identity] = path
    components = collections.defaultdict(list)
    for row in rows:
        components[groups.find(row['path'])].append(row)
    reasons = collections.defaultdict(set)
    for key, component in components.items():
        if any(row['original_split'] != 'train' for row in component):
            reasons[key].add('connected_to_historical_val_or_test')
    # Label conflicts need SHA/photo evidence; an observation alone may contain multiple species.
    for task, section in audit['sha_audit']['per_task'].items():
        for conflict in section['conflicting_label_groups']:
            for row in conflict:
                reasons[groups.find(row['path'])].add('same_image_conflicting_labels')
    for conflict in audit['photo_id_audit']['conflicting_species_groups'].values():
        for row in conflict:
            reasons[groups.find(row['path'])].add('same_photo_conflicting_species')
    eligible = []
    for row in rows:
        key = groups.find(row['path'])
        fingerprint = quality[row['path']]
        if not fingerprint or min(fingerprint[1:]) < 32:
            reasons[key].add('decode_failed_or_dimension_below_32')
    for row in rows:
        key = groups.find(row['path'])
        if row['original_split'] != 'train' or reasons[key]:
            excluded.append(dict(path=row['path'], task=row['task'],
                original_split=row['original_split'], reasons=sorted(reasons[key])))
            continue
        row['group_id'] = hashlib.sha256(key.encode()).hexdigest()[:20]
        row.setdefault('provenance', 'folder_species_label' if row['task'] == 'species' else 'folder_insect_label')
        eligible.append(row)
    # Retain one view of each source group per task/class, avoiding oversampling resized copies.
    deduplicated = {}
    for row in sorted(eligible, key=lambda r: (-quality[r['path']][1] * quality[r['path']][2], r['path'])):
        key = (row['task'], row['label'], row['group_id'])
        if key not in deduplicated:
            deduplicated[key] = row
        else:
            excluded.append(dict(path=row['path'], reason='same_task_class_group_representative',
                                 representative=deduplicated[key]['path']))
    eligible = list(deduplicated.values())
    assignment = assign_groups(eligible, args.seed)
    manifests = {task: {s: [] for s in SPLITS} for task in ('species', 'insect', 'detector')}
    for row in eligible:
        manifests[row['task']][assignment[row['group_id']]].append(row)
    negatives = collections.defaultdict(list)
    for row in eligible:
        if row['task'] == 'insect' and row['label'] != 'tick':
            negatives[row['label']].append(row)
    for label, items in negatives.items():
        # Cap total per class, approximately retaining the same train/val/calibration proportions.
        selected = []
        for split, cap in [('train', 75), ('val', 13), ('calibration', 12)]:
            choices = sorted((r for r in items if assignment[r['group_id']] == split),
                             key=lambda r: stable_order(r['path'], args.seed))[:cap]
            selected.extend(choices)
        for row in selected:
            _, width, height = quality[row['path']]
            negative = dict(row, task='detector', label='empty', negative_label=label,
                annotations=[], width=width, height=height, image_id='proxy_' + row['group_id'],
                provenance='folder_proxy_non_tick')
            manifests['detector'][assignment[row['group_id']]].append(negative)
    assert_disjoint(manifests)
    # Old archive2 test images are new only relative to audited known training sources.
    training_sources = {r['path'] for r in rows if r['original_split'] == 'train'}
    for source in sorted((root / 'data/tick_species').glob('split*.json')):
        training_sources.update(r['path'] for r in json.loads(source.read_text(encoding='utf-8')).get('train', []))
    historical_hashes, dhash_paths, bucket_index = set(), collections.defaultdict(list), collections.defaultdict(set)
    source_errors = []
    for path in sorted(training_sources):
        try:
            historical_hashes.add(hashes.get(path) or sha256(root / path))
            fingerprint = quality.get(path) or image_fingerprint(root / path)
            value = fingerprint[0]
            dhash_paths[value].append(path)
            for offset in range(8):
                bucket_index[(offset, (value >> (8 * offset)) & 255)].add(value)
        except (OSError, ValueError) as error:
            source_errors.append(dict(path=path, error=str(error)))
    unseen, unseen_excluded, unseen_groups = [], [], set()
    for folder in ('tick', 'mite'):
        for path in sorted((root / f'archive2/dataset/test/{folder}').iterdir()):
            if path.suffix.lower() not in EXTENSIONS:
                continue
            relative = path.relative_to(root).as_posix()
            digest = sha256(path)
            value, width, height = image_fingerprint(path)
            matches = near_matches(value, bucket_index)
            if digest in historical_hashes or matches or digest in unseen_groups:
                unseen_excluded.append(dict(path=relative, sha256=digest,
                    reason='exact_or_dhash_distance_le_4_to_known_training_or_duplicate',
                    matching_training_paths=[p for h in matches for p in dhash_paths[h]][:20]))
                continue
            unseen_groups.add(digest)
            unseen.append(dict(path=relative, label=folder, binary_label=1 if folder == 'tick' else 0,
                group_id='archive2test_' + digest[:20], sha256=digest, width=width, height=height,
                original_split='archive2_test_not_in_insects_split', source='archive2/dataset/test',
                provenance='folder_binary_label_same_source', dhash=f'{value:016x}'))
    summary = dict(seed=args.seed, builder_sha256=sha256(__file__), audit_sha256=sha256(audit_path),
        input_sha256=audit['source_files'], group_method='SHA256 + canonical photo_id + known observation_id across all tasks',
        counts={task: {s: len(rr) for s, rr in manifest.items()} for task, manifest in manifests.items()},
        class_support={task: {s: dict(collections.Counter(r.get('negative_label', r['label']) for r in rr))
                             for s, rr in manifest.items()} for task, manifest in manifests.items()},
        leakage_assertions={'global_group_and_sha_disjoint': True,
                            'training_pool_old_train_only': True,
                            'all_connected_historical_val_test_groups_excluded': True},
        binary_test=dict(included=len(unseen), excluded=len(unseen_excluded),
            known_training_source_paths=len(training_sources), source_errors=source_errors,
            labels=dict(collections.Counter(r['label'] for r in unseen)),
            status=('historical_sources_incomplete_same_source_comparison_only' if source_errors
                    else 'screened_against_known_training_sources_not_open_world_independent'),
            limitations=['Same archive2 source; folder labels, not independently adjudicated.',
                         'SHA and 64-bit dHash distance<=4 are screening, not a proof of independence.',
                         'Archive2 source observation/photo identity unavailable; other crop/rotation near duplicates may remain.',
                         'Historical training paths with missing files cannot be fully checked; see source_errors.',
                         'No new independent species test set is claimed.']),
        limitations=['Training/validation/calibration use old train pool and source groups, not a new external test.',
                     'Observation mapping incomplete outside API reference fields.',
                     'Detector negative rows are folder proxies, not human-confirmed absence.',
                     'Original COCO empty annotations are unverified, not strict negatives.'])
    out.mkdir(parents=True, exist_ok=True)
    for task, manifest in manifests.items():
        (out / f'{task}_split.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    for name, data in [('unseen_binary_test', unseen), ('excluded', excluded),
                       ('unseen_binary_excluded', unseen_excluded), ('summary', summary)]:
        (out / f'{name}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    summary['output_sha256'] = {p.name: sha256(p) for p in out.glob('*.json') if p.name != 'summary.json'}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(out), 'counts': summary['counts'],
        'binary_test': {k: v for k, v in summary['binary_test'].items() if k != 'source_errors'},
        'historical_source_errors': len(source_errors)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
