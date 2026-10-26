"""Calibrate frozen candidate outputs; report coverage and sampling uncertainty."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def acceptance_threshold(records,target=.95,minimum=20):
    ordered=sorted(records,key=lambda r:-r['species_confidence'])
    correct=0; chosen=None
    for i,row in enumerate(ordered):
        correct+=row['species_prediction']==row['species_label']
        n=i+1
        # Include ties together; a threshold cannot select half a tied score.
        if i+1<len(ordered) and ordered[i+1]['species_confidence']==row['species_confidence']:
            continue
        if n>=minimum and correct/n>=target:
            chosen=dict(threshold=row['species_confidence'],accepted=n,correct=correct,
                        precision=correct/n,wilson_lower=wilson_lower(correct,n))
    return chosen or dict(threshold=1.000001,accepted=0,correct=0,precision=None,wilson_lower=None)


def wilson_lower(successes,n,z=1.96):
    p=successes/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


def binary_threshold(records,target=.98):
    thresholds=[]; groups={}
    for kind in ('species_roi','coco_tick_roi','insect_tick_full'):
        values=sorted(r['tick_probability'] for r in records if r['tick_label']==1 and r['sample_type']==kind)
        if values:
            threshold=values[len(values)-math.ceil(target*len(values))]
            thresholds.append(threshold)
            groups[kind]=dict(views=len(values),quantile_threshold=threshold)
    if not thresholds:
        raise ValueError('No positive region strata for binary calibration')
    threshold=min(thresholds)
    positive=[r for r in records if r['tick_label']==1]
    negative=[r for r in records if r['tick_label']==0]
    return dict(threshold=threshold,target_recall=target,strata=groups,
        observed_recall=sum(r['tick_probability']>=threshold for r in positive)/len(positive),
        observed_proxy_fpr=sum(r['tick_probability']>=threshold for r in negative)/len(negative))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--classes',type=Path,required=True,help='Frozen recognition manifest')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads((args.comparison/'report.json').read_text('utf-8'))
    if report['subset']!='calibration':
        raise ValueError('Only frozen calibration outputs may determine these thresholds')
    split=json.loads(args.classes.read_text('utf-8'))
    if hashlib.sha256(args.classes.read_bytes()).hexdigest()!=report['manifest_sha256']:
        raise ValueError('Calibration manifest identity mismatch')
    records=json.loads((args.comparison/'new_predictions.json').read_text('utf-8'))
    regions=[r for r in records if r['sample_type']=='species_roi' and r['species_label']>=0]
    global_rule=acceptance_threshold(regions,minimum=50)
    thresholds={}; details={}
    for i,name in enumerate(split['classes']):
        rows=[r for r in regions if r['species_prediction']==i]
        rule=acceptance_threshold(rows)
        if rule['accepted']==0:
            rule=dict(global_rule,source='pooled_global_fallback')
        else:
            rule=dict(rule,source='class_calibration')
        thresholds[name]=rule['threshold']*100
        details[name]=dict(rule,predicted_views=len(rows))
    accepted=[r for r in regions if r['species_confidence']*100>=thresholds[split['classes'][r['species_prediction']]]]
    result=dict(checkpoint_sha256=report['checkpoint_sha256'],manifest_sha256=report['manifest_sha256'],
        predictions_sha256=hashlib.sha256((args.comparison/'new_predictions.json').read_bytes()).hexdigest(),
        binary=binary_threshold(records),species_thresholds=thresholds,species_calibration=details,
        species_summary=dict(views=len(regions),accepted=len(accepted),
            correct_accepted=sum(r['species_prediction']==r['species_label'] for r in accepted),
            coverage=len(accepted)/len(regions) if regions else 0),
        limitations=['calibration precision is empirical, not guaranteed future accuracy',
            'views share source groups','pooled fallback is not class-specific evidence',
            'rejected species keep candidates; acceptance is not top1 improvement'])
    if args.out.exists():
        raise ValueError('Calibration result already exists; keep frozen artifacts')
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(dict(binary=result['binary'],species_summary=result['species_summary']),indent=2))


if __name__=='__main__':
    main()
