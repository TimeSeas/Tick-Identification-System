"""Final CPU validation ablation: strong non-tick context, with local tick protection."""
import copy
import json

from compare_local_rejection import INITIAL_CASCADE, OUT
from compare_context_rejection import c90_reject
from evaluate_optimized_cascade import match_boxes, save_json, summarize


def main():
    source = json.loads((INITIAL_CASCADE/'cascade_val_optimized.json').read_text('utf-8'))
    rows = source['records']
    if len(rows) != 467:
        raise ValueError('Need full validation source')
    evidence_source = json.loads((OUT/'local_rejection_val_predictions.json').read_text('utf-8'))
    evidence = evidence_source['images']
    truth = {r['path']: r['boxes'] for r in json.loads(
        (OUT/'detection_val_cache.json').read_text('utf-8'))}
    summaries = {}
    for protect_local in (False, True):
        name = 'whole_non_tick90' + ('_protect_local_tick90' if protect_local else '')
        output_rows = []
        for row in rows:
            record = copy.deepcopy(row)
            observed = evidence[row['path']]
            whole = observed['whole']
            kept = []
            for d, local in zip(record['result']['detections'], observed['local']):
                if c90_reject(local, whole):
                    continue
                trigger = whole['slug'] != 'tick' and whole['confidence'] >= 90
                confirmed_support = (d['status'] == 'confirmed'
                                     and (d['confidence'] >= 90 or d['score'] >= .98))
                local_support = (protect_local and d['status'] == 'uncertain'
                                 and local['slug'] == 'tick' and local['confidence'] >= 90)
                if trigger and not (confirmed_support or local_support):
                    continue
                kept.append(d)
            record['result']['detections'] = kept
            tp, fp, fn = match_boxes(kept, truth[row['path']])
            record['metrics'] = {'tp': tp, 'fp': fp, 'fn': fn,
                                 'count_absolute_error': abs(len(kept)-len(truth[row['path']]))}
            output_rows.append(record)
        summary = summarize(output_rows)
        negative = [r for r in output_rows if r['task'] == 'negative']
        summary['negative_proxy_confirmed_images'] = sum(
            any(d['status'] == 'confirmed' for d in r['result']['detections']) for r in negative)
        summary['negative_proxy_by_label'] = {}
        for r in negative:
            group = summary['negative_proxy_by_label'].setdefault(r['proxy_label'], {
                'images': 0, 'positive_images': 0, 'confirmed_images': 0})
            group['images'] += 1
            group['positive_images'] += bool(r['result']['detections'])
            group['confirmed_images'] += any(d['status'] == 'confirmed' for d in r['result']['detections'])
        summaries[name] = summary
    baseline_rows = json.loads((INITIAL_CASCADE/'cascade_val_baseline.json').read_text('utf-8'))['records']
    baseline = summarize(baseline_rows)
    baseline['negative_proxy_confirmed_images'] = sum(
        r['task'] == 'negative' and any(d['status'] == 'confirmed' for d in r['result']['detections'])
        for r in baseline_rows)
    desired = [name for name, v in summaries.items()
               if v['recall'] > baseline['recall'] and v['negative_proxy_positive_images'] <= 60]
    save_json(OUT/'non_tick_context_val_comparison.json', {
        'subset': 'val', 'source_signature': source['signature'],
        'evidence_signature': evidence_source['signature'],
        'definitions': {
            'both': 'Keep original C90 local rejection (ARACHNID unchanged). When whole coarse slug != tick and confidence >=90, each box must be confirmed AND (species confidence >=90 OR detector score >=.98).',
            'protect_local_tick90': 'Additionally retain uncertain boxes when their own local coarse slug is tick with confidence >=90.',
        },
        'baseline': baseline, 'policies': summaries, 'desired_candidates': desired,
        'limitations': ['Validation only; no GPU or test data used.',
                        'Negative folder labels are proxies; positive COCO labels assess localization only.',
                        'Only further filters retained optimized boxes; full cascade rerun still required.'],
    })
    for name, v in [('baseline', baseline), *summaries.items()]:
        print(name, {k: v[k] for k in ('tp', 'fp', 'fn', 'precision', 'recall', 'count_mae',
                                       'negative_proxy_positive_images', 'negative_proxy_confirmed_images')})
    print('Desired candidates:', desired)


if __name__ == '__main__':
    main()
