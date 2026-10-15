"""Compare frozen and optimized cascades without rerunning the detector.

Validation is for policy selection; test is for the frozen policy only. This
evaluation does not attach image-level species labels to individual boxes.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/optimization_20261015'


def digest_file(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), 'utf-8')
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CachedDetector:
    """Reproduce baseline thresholding/deduplication using cached proposals."""
    tile = 2

    def __init__(self, dedup):
        self.row = None
        self.dedup = dedup

    def detect(self, rgb, tile=None):
        key = 'full' if tile == 1 else 'tiles'
        return self.dedup(copy.deepcopy([
            item for item in self.row[key] if item['score'] >= .5
        ]))


class CachedClassifier:
    """Evaluation-only prediction cache, separate from production services."""
    def __init__(self, classifier, name, model_sha, cache):
        self.classifier = classifier
        self.name = name
        self.model_sha = model_sha
        self.cache = cache
        self.hits = 0
        self.misses = 0

    def predict(self, rgb, **kwargs):
        identity = [self.name, self.model_sha, self.classifier.view_mode,
                    self.classifier.input_size, list(rgb.shape),
                    self.classifier.classes, kwargs.get('whole_weight', 0.0)]
        hasher = hashlib.sha256(json.dumps(identity).encode())
        hasher.update(rgb.tobytes())
        whole = kwargs.get('whole_rgb')
        if whole is not None:
            hasher.update(json.dumps(list(whole.shape)).encode())
            hasher.update(whole.tobytes())
        key = hasher.hexdigest()
        if key not in self.cache:
            self.cache[key] = self.classifier.predict(rgb, **kwargs)
            self.misses += 1
        else:
            self.hits += 1
        return copy.deepcopy(self.cache[key])

    def classify(self, rgb):
        result = self.predict(rgb)
        return result['slug'], round(result['confidence'], 2)


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(
        0, min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_boxes(predictions, truth):
    """Score-ordered one-to-one matching, detection IoU >= 0.5."""
    remaining = set(range(len(truth)))
    matched = 0
    for prediction in sorted(predictions, key=lambda d: -d['score']):
        if not remaining:
            break
        index = max(remaining, key=lambda j: iou(prediction['bbox'], truth[j]))
        if iou(prediction['bbox'], truth[index]) >= .5:
            remaining.remove(index)
            matched += 1
    return matched, len(predictions)-matched, len(truth)-matched


def summarize(records):
    detection = [r for r in records if r['task'] == 'detection']
    negative = [r for r in records if r['task'] == 'negative']
    tp = sum(r['metrics']['tp'] for r in detection)
    fp = sum(r['metrics']['fp'] for r in detection)
    fn = sum(r['metrics']['fn'] for r in detection)
    detections = [d for r in records for d in r['result']['detections']]
    negative_positive = sum(bool(r['result']['detections']) for r in negative)
    return {
        'evaluated_images': len(records), 'detection_images': len(detection),
        'tp': tp, 'fp': fp, 'fn': fn, 'matching_iou': .5,
        'precision': tp/(tp+fp) if tp+fp else None,
        'recall': tp/(tp+fn) if tp+fn else None,
        'count_mae': sum(r['metrics']['count_absolute_error'] for r in detection)
                     / len(detection) if detection else None,
        'negative_proxy_images': len(negative),
        'negative_proxy_positive_images': negative_positive,
        'negative_proxy_positive_rate': negative_positive/len(negative) if negative else None,
        'confirmed_boxes_all_images': sum(d['status'] == 'confirmed' for d in detections),
        'uncertain_boxes_all_images': sum(d['status'] == 'uncertain' for d in detections),
        'seconds_total': sum(r['seconds'] for r in records),
        'timing_note': 'Includes image decoding/classification; detector is cached; resumed prediction hits are faster.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subset', choices=['val', 'test'], default='val')
    parser.add_argument('--limit', type=int, default=0,
                        help='Smoke-test image limit; partial results are explicitly marked.')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    sys.path.insert(0, str(ROOT / 'web/backend'))
    import torch
    from app.services.classifier import SpeciesClassifier
    from app.services.insect_classifier import InsectClassifier
    from app.services.pipeline import Pipeline

    torch.set_num_threads(4)
    cache_path = OUT / f'detection_{args.subset}_cache.json'
    all_rows = json.loads(cache_path.read_text('utf-8'))
    expected_detection_count = 332 if args.subset == 'val' else 134
    if not args.limit and (sum(r['task'] == 'detection' for r in all_rows) != expected_detection_count
                           or sum(r['task'] == 'negative' for r in all_rows) != 135):
        raise ValueError('Cache is incomplete: need all COCO images plus 135 predetermined negative proxies')
    if len({r['path'] for r in all_rows}) != len(all_rows):
        raise ValueError('Duplicate image paths in detector cache')
    rows = all_rows[:args.limit] if args.limit else all_rows
    if any(r['task'] not in ('detection', 'negative') for r in rows):
        raise ValueError('Detection/negative cache required, not image-level species labels')
    profile_path = ROOT / 'configs/inference_optimization.json'
    profile = json.loads(profile_path.read_text('utf-8'))
    baseline_path = OUT / 'baseline_app/services/pipeline.py'
    detector_path = OUT / 'baseline_app/services/detector.py'
    species_path = ROOT / 'output/tick_species_clean_448/model/tick_species_best.pth'
    insect_path = ROOT / 'output/insect_classifier/model/insect_best.pth'
    thresholds_path = ROOT / 'output/tick_species_clean_448/thresholds.csv'
    dependency_paths = [Path(__file__).resolve(), profile_path, baseline_path, detector_path, species_path,
                        insect_path, thresholds_path,
                        ROOT / 'data/tick_species/split_clean.json',
                        ROOT / 'data/insects_split.json']
    dependency_paths += sorted((ROOT / 'web/backend/app/services').glob('*.py'))
    dependency_paths += sorted((ROOT / 'web/backend/app/utils').glob('*.py'))
    fingerprints = {str(p.relative_to(ROOT)): digest_file(p) for p in dependency_paths}
    manifest = {'subset': args.subset, 'limit': args.limit,
                'cache_sha256': digest_file(cache_path), 'inputs': fingerprints,
                'image_count': len(rows), 'available_cache_images': len(all_rows)}
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    suffix = f'_limit{args.limit}' if args.limit else ''
    prefix = OUT / f'cascade_{args.subset}{suffix}'
    predictions_path = OUT / f'cascade_{args.subset}_predictions.json'
    preprocessing_sha = hashlib.sha256(json.dumps({
        k: v for k, v in fingerprints.items()
        if k.endswith(('classifier.py', 'classification_views.py'))
    }, sort_keys=True).encode()).hexdigest()
    cached_predictions = {}
    if predictions_path.exists():
        prior = json.loads(predictions_path.read_text('utf-8'))
        if prior.get('preprocessing_sha256') == preprocessing_sha:
            cached_predictions = prior['predictions']
    species = SpeciesClassifier(str(species_path), str(ROOT/'data/tick_species/split_clean.json'))
    insect = InsectClassifier(str(insect_path), str(ROOT/'data/insects_split.json'))
    species_cached = CachedClassifier(species, 'species', digest_file(species_path), cached_predictions)
    insect_cached = CachedClassifier(insect, 'insect', digest_file(insect_path), cached_predictions)
    baseline = load_module('frozen_baseline_pipeline', baseline_path)
    detector_module = load_module('frozen_baseline_detector', detector_path)
    detector = CachedDetector(detector_module._dedup)
    summaries = {}
    for mode in ('baseline', 'optimized'):
        species.view_mode = 'center' if mode == 'baseline' else profile['species']['view_mode']
        insect.view_mode = 'center' if mode == 'baseline' else profile['insect']['view_mode']
        pipeline_type = baseline.Pipeline if mode == 'baseline' else Pipeline
        kwargs = {} if mode == 'baseline' else {'profile': profile}
        pipeline = pipeline_type(insect_cached, detector, species_cached,
                                 str(thresholds_path), {}, **kwargs)
        result_path = Path(f'{prefix}_{mode}.json')
        records = []
        if result_path.exists():
            previous = json.loads(result_path.read_text('utf-8'))
            if previous.get('signature') != signature:
                raise ValueError(f'{result_path.name}: source/config/cache changed; archive this result before rerunning')
            records = previous['records']
        completed = {r['path'] for r in records}

        def checkpoint():
            save_json(predictions_path, {'preprocessing_sha256': preprocessing_sha,
                                        'predictions': cached_predictions})
            save_json(result_path, {'signature': signature, 'manifest': manifest,
                                   'mode': mode, 'records': records})

        try:
            for row in rows:
                if row['path'] in completed:
                    continue
                detector.row = row
                started = time.perf_counter()
                result = pipeline.analyze((ROOT / row['path']).read_bytes())
                elapsed = time.perf_counter() - started
                tp, fp, fn = match_boxes(result['detections'], row['boxes'])
                records.append({'path': row['path'], 'task': row['task'],
                                'proxy_label': row.get('label'), 'seconds': elapsed,
                                'result': result, 'metrics': {
                                    'tp': tp, 'fp': fp, 'fn': fn,
                                    'count_absolute_error': abs(len(result['detections'])-len(row['boxes']))}})
                if len(records) % 20 == 0:
                    checkpoint()
                    print(f'{args.subset} {mode}: {len(records)}/{len(rows)}', flush=True)
        finally:
            checkpoint()
        summaries[mode] = summarize(records)
        print(mode, json.dumps(summaries[mode], ensure_ascii=False), flush=True)
    save_json(Path(f'{prefix}_summary.json'), {
        'signature': signature, 'manifest': manifest,
        'partial': bool(args.limit and args.limit < len(all_rows)),
        'complete_for_cache': all(x['evaluated_images'] == len(all_rows) for x in summaries.values()),
        'summaries': summaries,
        'limitations': [
            'Non-tick folder labels are image-level proxies, not independently verified negative boxes.',
            'COCO box labels assess localization/count only; confirmed species correctness is not assessed.',
            'Confidence/coverage are not medical accuracy; risk-advice text is not evaluated.',
            'Elapsed time excludes GPU detector inference and includes variable prediction-cache hits.',
        ],
        'classifier_cache_hits': species_cached.hits + insect_cached.hits,
        'classifier_cache_misses': species_cached.misses + insect_cached.misses,
    })


if __name__ == '__main__':
    main()
