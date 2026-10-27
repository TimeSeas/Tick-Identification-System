"""Freeze a specialist candidate and calibrate source-aware abstention offline.

Component calibration does not authorize deployment: actual detector proposals,
whole-image fusion, insect vetoes and independent regressions remain mandatory.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/train'))
from train_species_specialist import SpeciesViews,evaluate,compute_metrics,WARM_SHA
from prepare_species_specialist import digest,save,validate_specialist_manifest


def validate_candidate(checkpoint,manifest_sha):
    if checkpoint['partial'] or not checkpoint['metrics']['eligible']:
        raise ValueError('Partial or ineligible candidate cannot be exported')
    if checkpoint['config']['manifest_sha256']!=manifest_sha:
        raise ValueError('Checkpoint manifest identity mismatch')


def gate_metrics(records,threshold):
    known=defaultdict(list);unknown=defaultdict(list);accepted=correct=0
    for row in records:
        scores=np.asarray(row['probabilities']);prediction=int(scores.argmax())
        cutoff=threshold[prediction] if isinstance(threshold,dict) else threshold
        accept=bool(row.get('detected',True) and scores[prediction]>=cutoff)
        if row['label']<0:unknown[row['path']].append(accept)
        else:
            right=prediction==row['label']
            known[row['path']].append((accept,right))
            accepted+=accept;correct+=accept and right
    useful=sum(any(a and c for a,c in views) and not any(a and not c for a,c in views)
               for views in known.values())
    unknown_accepted=sum(any(views) for views in unknown.values())
    return dict(known_images=len(known),known_accepted_views=accepted,
        known_correct_accepted=correct,known_accepted_precision=correct/accepted if accepted else None,
        known_useful_images=useful,known_useful_coverage=useful/len(known) if known else None,
        unknown_images=len(unknown),unknown_accepted_images=unknown_accepted,
        unknown_false_acceptance=unknown_accepted/len(unknown) if unknown else None)


def select_threshold(records,min_precision=.95,max_unknown_acceptance=.05):
    if not any(r['label']>=0 for r in records) or not any(r['label']<0 for r in records):return None
    thresholds=sorted({float(max(r['probabilities'])) for r in records})
    best=None
    for threshold in thresholds:
        metrics=gate_metrics(records,threshold)
        if not metrics['known_accepted_views'] or metrics['known_accepted_precision']<min_precision:
            continue
        if metrics['unknown_false_acceptance']>max_unknown_acceptance:continue
        rank=(metrics['known_useful_images'],metrics['known_correct_accepted'],
              -metrics['unknown_accepted_images'],metrics['known_accepted_precision'],threshold)
        if best is None or rank>best[0]:best=(rank,dict(threshold=threshold,metrics=metrics))
    return best[1] if best else None


def attach_rows(predictions,rows):
    result=[]
    for prediction in predictions:
        row=rows[prediction['index']]
        result.append(dict(prediction,path=row['path'],bbox=row['bbox'],
                           source_species=row['source_species']))
    return result


def modes(records):
    grouped=defaultdict(list)
    for row in records:grouped[row['path']].append(row)
    single=[];regions=[]
    for path,views in grouped.items():
        full=[row for row in views if row['bbox'] is None]
        crops=[row for row in views if row['bbox'] is not None]
        if len(full)!=1:raise ValueError('Exactly one full view required per source: '+path)
        for crop in crops:
            regions.append(crop)
        if len(crops)==1:
            probabilities=((np.asarray(full[0]['probabilities'])+np.asarray(crops[0]['probabilities']))/2).tolist()
            single.append(dict(full[0],probabilities=probabilities))
        else:single.append(full[0])
    return dict(all_views=records,single_proxy=single,region=regions)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',choices=('calibration','holdout'),default='calibration')
    p.add_argument('--policy',type=Path,help='Frozen calibration summary required for holdout')
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();torch.set_num_threads(4)
    if args.out.exists():raise ValueError('Preserve previous evaluation; require new output')
    manifest=json.loads(args.manifest.read_text('utf-8'));classes=validate_specialist_manifest(manifest)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    validate_candidate(checkpoint,digest(args.manifest))
    if checkpoint['config']['classes']!=classes:raise ValueError('Candidate class order mismatch')
    if checkpoint['config']['warm_start_sha256']!=WARM_SHA:raise ValueError('Unexpected teacher identity')
    warm=Path(checkpoint['config']['warm_start'])
    if digest(warm)!=WARM_SHA:raise ValueError('Teacher content changed')
    policy=json.loads(args.policy.read_text('utf-8')) if args.policy else None
    if args.phase=='holdout':
        if not policy or policy['checkpoint_sha256']!=digest(args.checkpoint) or policy['manifest_sha256']!=digest(args.manifest):
            raise ValueError('Holdout requires a matching frozen calibration policy')
        if policy['phase']!='calibration':raise ValueError('Invalid policy phase')
    rows=manifest[args.phase]
    for row in rows:
        if digest(ROOT/row['path'])!=row['sha256']:raise ValueError('Evaluation image content changed')
    args.out.mkdir(parents=True)
    loader=DataLoader(SpeciesViews(rows,classes),batch_size=8,shuffle=False,num_workers=args.workers)
    report=dict(phase=args.phase,manifest_sha256=digest(args.manifest),checkpoint_sha256=digest(args.checkpoint),
        evaluator_sha256=digest(__file__),checkpoint_epoch=checkpoint['epoch']+1,classes=classes,
        policy_sha256=digest(args.policy) if args.policy else None,
        limitations=['Component comparison, not end-to-end release acceptance',
                    'Known teacher regions are inherited image labels, not manual box species truth',
                    'Unknown full-image-only rows cannot calibrate actual region or fusion rejection',
                    'Some known validation sources were seen by the original teacher'],models={})
    torch.save(checkpoint['model'],args.out/'candidate.pth')
    report['candidate_sha256']=digest(args.out/'candidate.pth')
    for name,state in (('baseline',torch.load(warm,map_location='cpu',weights_only=True)),('candidate',checkpoint['model'])):
        model=resnet50(weights=None);model.fc=nn.Linear(model.fc.in_features,len(classes))
        model.load_state_dict(state,strict=True);model.to(args.device).eval()
        metrics,predictions=evaluate(model,loader,torch.device(args.device),classes)
        records=attach_rows(predictions,rows)
        save(args.out/(name+'_predictions.json'),records)
        details={}
        for mode,selected in modes(records).items():
            mode_metrics=compute_metrics(selected,classes)
            selected_policy=select_threshold(selected) if args.phase=='calibration' else policy['models'][name]['modes'][mode]['policy']
            details[mode]=dict(metrics=mode_metrics,policy=selected_policy,
                frozen_gate_metrics=gate_metrics(selected,selected_policy['threshold']) if selected_policy else None)
        report['models'][name]=dict(metrics=metrics,modes=details)
        del model
        if args.device=='cuda':torch.cuda.empty_cache()
        print(name+' '+json.dumps({k:v for k,v in metrics.items() if k!='by_class'}),flush=True)
    save(args.out/'summary.json',report)
    print('Saved '+str(args.out/'summary.json'),flush=True)


if __name__=='__main__':main()
