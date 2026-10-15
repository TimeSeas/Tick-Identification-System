"""CPU-only validation grid using already recorded classification evidence.

Each detected box is evaluated independently. Strong non-arachnid whole-image
evidence requires a confirmed species and, optionally, stronger local support.
This is an ablation, not a replacement for a complete cascade rerun.
"""
import copy
import json

from compare_local_rejection import ARACHNID, INITIAL_CASCADE, OUT, non_arachnid
from evaluate_optimized_cascade import match_boxes, save_json, summarize


def c90_reject(local, whole):
    return non_arachnid(local, 90) or (
        non_arachnid(local, 80) and non_arachnid(whole, 90))


def main():
    source = json.loads((INITIAL_CASCADE/'cascade_val_optimized.json').read_text('utf-8'))
    rows = source['records']
    if len(rows) != 467:
        raise ValueError('Full validation source required')
    evidence_source = json.loads((OUT/'local_rejection_val_predictions.json').read_text('utf-8'))
    evidence = evidence_source['images']
    truth = {r['path']: r['boxes'] for r in json.loads(
        (OUT/'detection_val_cache.json').read_text('utf-8'))}
    policies = [('current', None, None, False), ('C90', None, None, True)]
    for apply_c90 in (False, True):
        suffix = '_C90' if apply_c90 else ''
        policies.append(('global90_confirmed_only'+suffix, 0, 0, apply_c90))
        for confidence in (70, 80, 90):
            for score in (.95, .98, .99):
                policies.append((f'global90_species{confidence}_or_det{score}{suffix}',
                                 confidence, score, apply_c90))
    metrics = {}
    definitions = {}
    for name, confidence, score, apply_c90 in policies:
        results = []
        for row in rows:
            record = copy.deepcopy(row)
            item_evidence = evidence[row['path']]
            whole = item_evidence['whole']
            kept = []
            for d, local in zip(record['result']['detections'], item_evidence['local']):
                if apply_c90 and c90_reject(local, whole):
                    continue
                if confidence is not None and non_arachnid(whole, 90):
                    confirmed = d['status'] == 'confirmed'
                    strong_support = (confidence == 0 or d['confidence'] >= confidence or d['score'] >= score)
                    if not (confirmed and strong_support):
                        continue
                kept.append(d)
            record['result']['detections'] = kept
            tp, fp, fn = match_boxes(kept, truth[row['path']])
            record['metrics'] = {'tp': tp, 'fp': fp, 'fn': fn,
                                 'count_absolute_error': abs(len(kept)-len(truth[row['path']]))}
            results.append(record)
        value = summarize(results)
        negative = [r for r in results if r['task'] == 'negative']
        value['negative_proxy_confirmed_images'] = sum(
            any(d['status'] == 'confirmed' for d in r['result']['detections']) for r in negative)
        value['negative_proxy_confirmed_boxes'] = sum(
            d['status'] == 'confirmed' for r in negative for d in r['result']['detections'])
        # On image-level proxy labels, inspect false tick confirmations even
        # when the coarse classifier correctly recognized the actual category.
        original_by_path = {r['path']: r for r in rows}
        correct_coarse = []
        for record in negative:
            whole = evidence[record['path']]['whole']
            if whole is None:
                whole = original_by_path[record['path']]['result']['insect']
            if whole['slug'] == record['proxy_label']:
                correct_coarse.append((record, whole))
        value['negative_proxy_coarse_correct_images'] = len(correct_coarse)
        value['negative_proxy_coarse_correct_but_tick_confirmed_images'] = sum(
            any(d['status'] == 'confirmed' for d in r['result']['detections'])
            for r, whole in correct_coarse)
        value['negative_proxy_coarse_correct_ge90_but_tick_confirmed_images'] = sum(
            whole['confidence'] >= 90 and any(d['status'] == 'confirmed'
                                             for d in r['result']['detections'])
            for r, whole in correct_coarse)
        value['negative_proxy_by_label'] = {}
        for r in negative:
            label = r['proxy_label']
            group = value['negative_proxy_by_label'].setdefault(label, {
                'images': 0, 'positive_images': 0, 'confirmed_images': 0, 'confirmed_boxes': 0})
            group['images'] += 1
            group['positive_images'] += bool(r['result']['detections'])
            group['confirmed_images'] += any(d['status'] == 'confirmed' for d in r['result']['detections'])
            group['confirmed_boxes'] += sum(d['status'] == 'confirmed' for d in r['result']['detections'])
        metrics[name] = value
        definitions[name] = {'apply_local_c90': apply_c90,
                             'whole_trigger': None if confidence is None else 'coarse non-ARACHNID confidence >=90',
                             'per_box_exemption': None if confidence is None else (
                                 'species status confirmed' if confidence == 0 else
                                 f'species status confirmed AND (species confidence >= {confidence} OR detector score >= {score})'),
                             'trigger_effect': 'Remove only the individual box that fails exemption; keep independently supported other boxes'}
    candidates = [name for name, v in metrics.items()
                  if v['recall'] > 258/318 and v['negative_proxy_positive_images'] <= 60]
    def dominates(a, b):
        aa = [a['precision'], a['recall'], -a['negative_proxy_positive_images'],
              -a['negative_proxy_confirmed_images'], -a['count_mae']]
        bb = [b['precision'], b['recall'], -b['negative_proxy_positive_images'],
              -b['negative_proxy_confirmed_images'], -b['count_mae']]
        return all(x >= y for x, y in zip(aa, bb)) and any(x > y for x, y in zip(aa, bb))
    pareto = [name for name, value in metrics.items()
              if not any(dominates(other, value) for other in metrics.values())]
    output = {'subset': 'val', 'source_cascade_signature': source['signature'],
              'source_evidence_signature': evidence_source['signature'],
              'definitions': definitions, 'metrics': metrics, 'pareto': pareto,
              'desired_candidates': candidates,
              'desired_criteria': {'recall_greater_than': 258/318, 'negative_proxy_positive_images_maximum': 60},
              'limitations': ['Image-level non-tick labels are proxies, not verified object-level negatives.',
                              'Positive-image species correctness is not measured by these metrics.',
                              'Only further filters prior optimized retained boxes; requires full rerun.',
                              'Uses saved rounded species confidence, which could differ at exact cutoff boundaries.']}
    save_json(OUT/'context_rejection_val_comparison.json', output)
    for name, v in metrics.items():
        print(name, {k: v[k] for k in ('tp', 'fp', 'fn', 'precision', 'recall', 'count_mae',
                                       'negative_proxy_positive_images', 'negative_proxy_confirmed_images')})
    print('Desired candidates:', candidates)
    print('Pareto:', pareto)


if __name__ == '__main__':
    main()
