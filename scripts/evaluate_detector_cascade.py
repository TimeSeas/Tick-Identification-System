"""Evaluate a retrained detector with three predeclared cascade policies on val.

Identical decoded pixels, actual service detector integerization/dedup/tiles,
original species model retained. All outputs remain development evidence.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'web/backend'))
from app.services.detector import Detector
from app.services.classifier import SpeciesClassifier
from app.services.insect_classifier import InsectClassifier
from app.services.pipeline import Pipeline
from evaluate_recognition_cascade import CachedDetector,read_image,summarize
from build_recognition_profile import digest


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),'utf-8'); temporary.replace(path)


def proxy_routing(rows,records,key):
    count=correct=confirmed_correct=0
    for row,record in zip(rows,records):
        if not row.get('negative_label'):
            continue
        result=record[key]; count+=1
        matches=result['slug']==row['negative_label']
        correct+=matches
        confirmed_correct+=matches and result.get('status')=='confirmed'
    return dict(proxy_images=count,correct_fallback_candidate_images=correct,
        correct_fallback_confirmed_images=confirmed_correct,
        correct_candidate_rate=correct/count if count else None,
        correct_confirmed_rate=confirmed_correct/count if count else None)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--baseline-traces',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--proposal-weights',type=Path,help='Optional v3 proposals verified by retrained ROI head')
    p.add_argument('--score-threshold',type=float,choices=(.3,.4,.5),default=.5)
    p.add_argument('--variants',nargs='+',choices=('drop_in','regional_only','detector_only'),
        default=['drop_in','regional_only','detector_only'])
    args=p.parse_args(); torch.set_num_threads(4)
    manifest=json.loads(args.split.read_text('utf-8')); rows=manifest['val']
    baseline=json.loads(args.baseline_traces.read_text('utf-8'))
    profile_path=ROOT/'configs/inference_optimization.json'
    species_path=ROOT/'output/tick_species_clean_448/model/tick_species_best.pth'
    insect_path=ROOT/'output/insect_classifier/model/insect_best.pth'
    expected=manifest.get('metadata',{}).get('source_sha256',digest(args.split))
    for key,value in dict(split_sha256=expected,subset='val',old_profile_sha256=digest(profile_path),
        old_species_sha256=digest(species_path),old_insect_sha256=digest(insect_path),
        old_pipeline_sha256=digest(ROOT/'web/backend/app/services/pipeline.py')).items():
        if baseline['metadata'][key]!=value:
            raise ValueError('Baseline traces do not match current artifacts: '+key)
    if len(baseline['rows'])!=len({r['path'] for r in baseline['rows']}):
        raise ValueError('Duplicate baseline trace paths')
    old={r['path']:r['old'] for r in baseline['rows']}
    metadata=dict(weights_sha256=digest(args.weights),split_sha256=digest(args.split),
        old_profile_sha256=digest(profile_path),baseline_traces_sha256=digest(args.baseline_traces),
        species_sha256=digest(species_path),insect_sha256=digest(insect_path),
        evaluator_sha256=digest(Path(__file__)),pipeline_sha256=digest(ROOT/'web/backend/app/services/pipeline.py'),
        detector_sha256=digest(ROOT/'web/backend/app/services/detector.py'),subset='val')
    metadata['classification_sources']={name:digest(ROOT/'web/backend/app/services'/name)
        for name in ('classifier.py','insect_classifier.py','classification_views.py')}
    metadata.update(score_threshold=args.score_threshold,variants=args.variants)
    metadata['cache_helper_sha256']=digest(ROOT/'scripts/evaluate_recognition_cascade.py')
    if args.proposal_weights:
        metadata.update(proposal_weights_sha256=digest(args.proposal_weights),
            verifier_source_sha256=digest(ROOT/'web/backend/app/services/verified_detector.py'))
    for row in rows:
        if digest(ROOT/row['path'])!=row['sha256']:
            raise ValueError('Frozen image changed: '+row['path'])
    args.out.mkdir(parents=True,exist_ok=True)
    metadata_path=args.out/'metadata.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text('utf-8'))!=metadata:
        raise ValueError('Changed experiment inputs; use a new directory')
    save(metadata_path,metadata)
    cache_path=args.out/'detections.json'
    cache=json.loads(cache_path.read_text('utf-8')) if cache_path.exists() else {}
    if any(r['path'] not in cache for r in rows):
        if args.proposal_weights:
            from app.services.verified_detector import VerifiedDetector
            detector=VerifiedDetector(str(args.proposal_weights),str(args.weights),tile=2)
        else:
            detector=Detector(str(args.weights),tile=2)
        detector.model.roi_heads.mask_roi_pool=None
        detector.CONFIDENCE=args.score_threshold
        for index,row in enumerate(rows):
            if row['path'] in cache:
                continue
            image=read_image(row); rgb=np.array(image)
            full=detector.detect(rgb,tile=1)
            tiles=detector.detect(rgb,tile=2) if not full else []
            cache[row['path']]=dict(full=full,tiles=tiles)
            if (index+1)%20==0:
                save(cache_path,cache); print(f'detector {index+1}/{len(rows)}',flush=True)
        save(cache_path,cache); del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    species=SpeciesClassifier(str(species_path),str(ROOT/'data/tick_species/split_clean.json'),448,'center_full')
    insect=InsectClassifier(str(insect_path),str(ROOT/'data/insects_split.json'),view_mode='center_full')
    cached=CachedDetector(); original=json.loads(profile_path.read_text('utf-8'))
    cached.CONFIDENCE=args.score_threshold
    baseline_fallback=Pipeline(insect,cached,species,str(ROOT/'output/tick_species_clean_448/thresholds.csv'),{},original)
    old_fallbacks={}
    reports={}
    old_result=summarize(rows,[dict(old=old[r['path']]) for r in rows],'old')
    for name in args.variants:
        profile=copy.deepcopy(original)
        if name!='drop_in':
            profile.setdefault('rejection',{})['whole_non_tick']=101
        if name=='detector_only':
            profile['rejection'].update(local_non_arachnid=101,local_consensus=101)
        save(args.out/(name+'_profile.json'),profile)
        pipeline=Pipeline(insect,cached,species,str(ROOT/'output/tick_species_clean_448/thresholds.csv'),{},profile)
        trace_path=args.out/(name+'_traces.json')
        records=json.loads(trace_path.read_text('utf-8')) if trace_path.exists() else []
        if len(records)>len(rows) or any(record['path']!=row['path'] for record,row in zip(records,rows)):
            raise ValueError('Trace order or length changed')
        for index,row in enumerate(rows):
            if index<len(records):
                continue
            image=read_image(row); cached.row=cache[row['path']]
            with patch('app.services.pipeline.load_rgb_image',return_value=image):
                result=pipeline.analyze(b'')
            record=dict(path=row['path'],old=old[row['path']],new=result['detections'],insect=result['insect'])
            if row.get('negative_label'):
                if row['path'] not in old_fallbacks:
                    old_fallbacks[row['path']]=(dict(slug='tick') if old[row['path']]
                        else baseline_fallback._fallback(np.array(image))['insect'])
                record['old_insect']=old_fallbacks[row['path']]
            records.append(record)
            if (index+1)%20==0:
                save(trace_path,records); print(f'{name} {index+1}/{len(rows)}',flush=True)
        save(trace_path,records)
        current=summarize(rows,[dict(old=r['new']) for r in records],'old')
        reports[name]=dict(profile_sha256=digest(args.out/(name+'_profile.json')),old=old_result,new=current,gates=dict(
            recall_loss_at_most_2pp=current['recall']>=old_result['recall']-.02,
            relative_proxy_fp_reduction_30pct=current['proxy_positive_rate']<=old_result['proxy_positive_rate']*.7,
            count_mae_not_worse=current['count_mae']<=old_result['count_mae']))
        reports[name]['folder_label_routing']=dict(old=proxy_routing(rows,records,'old_insect'),
            new=proxy_routing(rows,records,'insect'))
        print(json.dumps({name:reports[name]},ensure_ascii=False),flush=True)
    save(args.out/'report.json',dict(metadata=metadata,variants=reports,limitations=[
        'development val used for policy selection, not independent final test',
        'old task weights saw some new val source groups; folder negatives are proxies',
        'equal decoded pixels; upload size and EXIF validation excluded from this model comparison',
        'no per-box species truth; retaining original species model does not prove species improvement']))


if __name__=='__main__':
    main()
