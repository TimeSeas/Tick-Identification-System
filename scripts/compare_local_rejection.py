"""Validation-only ablation of local coarse-classifier rejection policies.

Only further filters boxes retained by the recorded optimized cascade. A chosen
policy still requires a complete cascade rerun before test evaluation/deployment.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

from evaluate_optimized_cascade import CachedClassifier, digest_file, match_boxes, save_json, summarize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/optimization_20261015'
INITIAL_CASCADE = OUT / 'initial_cascade_val'
ARACHNID = {'tick', 'mite', 'spider', 'spider_mite'}


def non_arachnid(result, threshold):
    return result['slug'] not in ARACHNID and result['confidence'] >= threshold


def retained(detections, evidence, policy):
    if policy == 'current':
        return detections
    keep = []
    for d, local in zip(detections, evidence['local']):
        consensus = non_arachnid(local, 80) and non_arachnid(evidence['whole'], 90)
        if policy == 'B_local80_whole90':
            reject = consensus
        else:
            family, threshold = policy.split('_')
            reject = non_arachnid(local, float(threshold))
            if family == 'C':
                reject = reject or consensus
        if not reject:
            keep.append(d)
    return keep


def main():
    sys.path.insert(0, str(ROOT/'web/backend'))
    import numpy as np
    import torch
    from app.services.insect_classifier import InsectClassifier
    from app.utils.images import load_rgb_image

    torch.set_num_threads(4)
    source_path = INITIAL_CASCADE/'cascade_val_optimized.json'
    source = json.loads(source_path.read_text('utf-8'))
    rows = source['records']
    if len(rows) != 467:
        raise ValueError('Need all 467 optimized validation results')
    detector_rows = {r['path']: r for r in json.loads((OUT/'detection_val_cache.json').read_text('utf-8'))}
    model_path = ROOT/'output/insect_classifier/model/insect_best.pth'
    fingerprint_paths = [source_path, model_path,
                         ROOT/'web/backend/app/services/insect_classifier.py',
                         ROOT/'web/backend/app/services/classification_views.py',
                         ROOT/'web/backend/app/utils/images.py']
    # Preserve the original logical identity after moving the unchanged source
    # into its archive, so existing local-evidence checkpoints remain reusable.
    manifest = {str((OUT/p.name if p == source_path else p).relative_to(ROOT)):
                digest_file(p) for p in fingerprint_paths}
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    evidence_path = OUT/'local_rejection_val_predictions.json'
    evidence = {}
    if evidence_path.exists():
        previous = json.loads(evidence_path.read_text('utf-8'))
        if previous['signature'] != signature:
            raise ValueError('Validation evidence input changed; archive prior evidence before rerunning')
        evidence = previous['images']
    prediction_path = OUT/'cascade_val_predictions.json'
    prior = json.loads(prediction_path.read_text('utf-8'))
    # Reuse the existing evaluation cache only when its classification source
    # fingerprint still matches, never overwriting that prior cache.
    current_sources = {
        str(p.relative_to(ROOT)): digest_file(p)
        for p in sorted((ROOT/'web/backend/app/services').glob('*.py'))
        if p.name.endswith(('classifier.py', 'classification_views.py'))
    }
    preprocessing_sha = hashlib.sha256(json.dumps(current_sources, sort_keys=True).encode()).hexdigest()
    predictions = prior['predictions'] if prior['preprocessing_sha256'] == preprocessing_sha else {}
    service = InsectClassifier(str(model_path), str(ROOT/'data/insects_split.json'), view_mode='center_full')
    classifier = CachedClassifier(service, 'insect', digest_file(model_path), predictions)

    def checkpoint():
        save_json(evidence_path, {'signature': signature, 'manifest': manifest, 'padding': .15,
                                 'view_mode': 'center_full', 'images': evidence})
        save_json(OUT/'local_rejection_val_classifier_cache.json', {
            'preprocessing_sha256': preprocessing_sha, 'predictions': predictions})

    additional_path = OUT/'local_rejection_val_classifier_cache.json'
    if additional_path.exists():
        previous = json.loads(additional_path.read_text('utf-8'))
        if previous['preprocessing_sha256'] == preprocessing_sha:
            predictions.update(previous['predictions'])
    try:
        for index, row in enumerate(rows):
            if row['path'] in evidence:
                continue
            detections = row['result']['detections']
            if not detections:
                evidence[row['path']] = {'whole': None, 'local': []}
                continue
            image = load_rgb_image((ROOT/row['path']).read_bytes())
            rgb = np.array(image)
            whole = classifier.predict(rgb)
            local = []
            for d in detections:
                x1, y1, x2, y2 = d['bbox']
                dx, dy = (x2-x1)*.15, (y2-y1)*.15
                crop = image.crop((max(0, int(x1-dx)), max(0, int(y1-dy)),
                                   min(image.width, int(x2+dx)), min(image.height, int(y2+dy))))
                local.append(classifier.predict(np.array(crop)))
            evidence[row['path']] = {'whole': whole, 'local': local}
            if (index+1) % 20 == 0:
                checkpoint()
                print(f'validation local evidence {index+1}/{len(rows)}; classifier misses={classifier.misses}', flush=True)
    finally:
        checkpoint()
    policies = ['current', 'A_90', 'A_95', 'A_98', 'B_local80_whole90', 'C_90', 'C_95', 'C_98']
    metrics = {}
    for policy in policies:
        results = []
        for row in rows:
            record = copy.deepcopy(row)
            detections = retained(record['result']['detections'], evidence[row['path']], policy)
            record['result']['detections'] = detections
            truth = detector_rows[row['path']]['boxes']
            tp, fp, fn = match_boxes(detections, truth)
            record['metrics'] = {'tp': tp, 'fp': fp, 'fn': fn,
                                 'count_absolute_error': abs(len(detections)-len(truth))}
            results.append(record)
        metrics[policy] = summarize(results)
    def dominates(a, b):
        left = [a['precision'], a['recall'], -a['negative_proxy_positive_images'], -a['count_mae']]
        right = [b['precision'], b['recall'], -b['negative_proxy_positive_images'], -b['count_mae']]
        return all(x >= y for x, y in zip(left, right)) and any(x > y for x, y in zip(left, right))
    pareto = [name for name, value in metrics.items()
              if not any(dominates(other, value) for other in metrics.values())]
    output = {'subset': 'val', 'source_signature': signature, 'manifest': manifest,
              'policies': metrics, 'pareto': pareto,
              'pareto_objectives': ['maximize precision', 'maximize recall',
                                    'minimize proxy positive images', 'minimize count MAE'],
              'limitations': ['Only filters retained optimized boxes; cannot recover previously rejected boxes.',
                              'Full selected cascade must be rerun before independent test evaluation.',
                              'Non-tick categories are image-level negative proxies, not verified box ground truth.',
                              'No risk-advice evaluation; original text is not recomputed after this offline filter.']}
    save_json(OUT/'local_rejection_val_comparison.json', output)
    for name, value in metrics.items():
        print(name, {k: value[k] for k in ('tp', 'fp', 'fn', 'precision', 'recall',
                                          'count_mae', 'negative_proxy_positive_images')}, flush=True)
    print('Pareto:', pareto, flush=True)


if __name__ == '__main__':
    main()
