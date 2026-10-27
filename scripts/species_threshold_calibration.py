"""Support-limited class thresholds with a conservative unknown-image budget."""
import math
import numpy as np
from evaluate_species_specialist import gate_metrics


def calibrate_class_thresholds(records,classes,global_threshold,min_precision=.95,
                              max_unknown_acceptance=.05,min_support=20,
                              min_threshold=.5487344360351562):
    unknown={r['path'] for r in records if r['label']<0}
    if not unknown or not any(r['label']>=0 for r in records):return None
    budget=math.floor(len(unknown)*max_unknown_acceptance+1e-9)
    states={0:(0,0,{})}
    support={}
    for index,slug in enumerate(classes):
        rows=[r for r in records if r.get('detected',True) and int(np.argmax(r['probabilities']))==index]
        known_sources={r['path'] for r in rows if r['label']>=0};support[slug]=len(known_sources)
        candidates={float(global_threshold),1.000001}
        if len(known_sources)>=min_support:
            candidates.update(max(float(max(r['probabilities'])),min_threshold) for r in rows)
        options={}
        for threshold in sorted(candidates):
            accepted=[r for r in rows if max(r['probabilities'])>=threshold]
            known=[r for r in accepted if r['label']>=0]
            correct_sources={r['path'] for r in known if r['label']==index}
            wrong_sources={r['path'] for r in known if r['label']!=index}
            correct=len(correct_sources);wrong=len(wrong_sources)
            cost=len({r['path'] for r in accepted if r['label']<0})
            if cost>budget or (correct+wrong and correct/(correct+wrong)<min_precision):continue
            # Correct sources belong to one truth class. Wrong accepted sources
            # may spoil another class's useful image, so penalize them rather
            # than rewarding repeated proposals from the same source.
            rank=(correct-wrong,-wrong,threshold)
            if cost not in options or rank>options[cost][0]:options[cost]=(rank,threshold)
        updated={}
        for used,(score,negative_wrong,table) in states.items():
            for cost,(rank,threshold) in options.items():
                total=used+cost
                if total>budget:continue
                candidate=(score+rank[0],negative_wrong+rank[1],dict(table,**{slug:threshold}))
                if total not in updated or candidate[:2]>updated[total][:2]:updated[total]=candidate
        states=updated
    if not states:return None
    _,_,table=max(states.values(),key=lambda value:value[:2])
    metrics=gate_metrics(records,{i:table[c] for i,c in enumerate(classes)})
    if not metrics['known_correct_accepted']:return None
    if metrics['unknown_false_acceptance']>max_unknown_acceptance+1e-9:raise AssertionError('Unknown budget exceeded')
    return dict(thresholds=table,metrics=metrics,method='support_limited_class_dp',
        min_support=min_support,min_threshold=min_threshold,predicted_class_support=support,
        limitation='Source gain penalizes wrong-source confirmations; unknown budget sums class source counts conservatively; folder labels are proxies')
