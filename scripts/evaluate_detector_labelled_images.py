"""Image-label cascade checks after detector/policy selection, no box truth claim."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'web/backend'))
from app.services.detector import Detector
from app.services.classifier import SpeciesClassifier
from app.services.insect_classifier import InsectClassifier
from app.services.pipeline import Pipeline
from build_recognition_profile import digest


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),'utf-8'); temporary.replace(path)


def summarize(records,name,known):
    detected=correct_candidate=correct_confirmed=wrong_confirmed=confirmed=0
    by_class=defaultdict(lambda:dict(images=0,detected=0,correct_candidate=0,correct_confirmed=0))
    for record in records:
        ds=record[name]['detections']; label=record['label']
        accepted=[d for d in ds if d['status']=='confirmed']
        right_candidate=any(d['candidate_slug']==label for d in ds) if known else False
        right_confirmed=any(d['species_slug']==label for d in accepted) if known else False
        wrong=any(d['species_slug']!=label for d in accepted) if known else bool(accepted)
        detected+=bool(ds); confirmed+=bool(accepted); wrong_confirmed+=wrong
        correct_candidate+=right_candidate; correct_confirmed+=right_confirmed
        item=by_class[label]; item['images']+=1; item['detected']+=bool(ds)
        item['correct_candidate']+=right_candidate; item['correct_confirmed']+=right_confirmed
    result=dict(images=len(records),detected_images=detected,presence_recall=detected/len(records),
        confirmed_images=confirmed,images_with_wrong_known_species_confirmation=wrong_confirmed)
    if known:
        result.update(correct_candidate_images=correct_candidate,correct_confirmed_images=correct_confirmed,
            correct_candidate_coverage=correct_candidate/len(records),
            correct_confirmed_coverage=correct_confirmed/len(records),
            macro_correct_candidate_coverage=sum(v['correct_candidate']/v['images'] for v in by_class.values())/len(by_class),
            by_class=dict(by_class))
    return result


def summarize_binary(records,name):
    tp=fn=fp=tn=0
    for record in records:
        detected=bool(record[name]['detections']); positive=record['binary_label']==1
        tp+=detected and positive; fn+=not detected and positive
        fp+=detected and not positive; tn+=not detected and not positive
    return dict(tp=tp,fn=fn,fp=fp,tn=tn,presence_recall=tp/(tp+fn) if tp+fn else None,
        proxy_fpr=fp/(fp+tn) if fp+tn else None)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--species-split',type=Path)
    source.add_argument('--coverage-manifest',type=Path)
    source.add_argument('--binary-regression',type=Path)
    p.add_argument('--new-weights',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--proposal-weights',type=Path,help='Optional original-proposal ROI verification experiment')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(); torch.set_num_threads(4)
    known=bool(args.species_split); binary=bool(args.binary_regression)
    manifest=args.species_split or args.coverage_manifest or args.binary_regression
    data=json.loads(manifest.read_text('utf-8')); rows=data['val'] if known else data if binary else data['rows']
    if not rows or len(rows)!=len({r['path'] for r in rows}):
        raise ValueError('Require nonempty unique image rows')
    old_weights=ROOT/'output/det_v3/mask_rcnn_ticks_v3.pth'
    old_profile_path=ROOT/'configs/inference_optimization.json'
    original=json.loads(old_profile_path.read_text('utf-8')); profile=json.loads(args.profile.read_text('utf-8'))
    if profile.get('recognition'):
        raise ValueError('This evaluation is for the retrained detector with the retained legacy species model')
    if profile.get('detector',{}).get('model_sha256')!=digest(args.new_weights):
        raise ValueError('Profile detector identity mismatch')
    species_path=ROOT/'output/tick_species_clean_448/model/tick_species_best.pth'
    insect_path=ROOT/'output/insect_classifier/model/insect_best.pth'
    for config in (original,profile):
        if config['species']!=original['species'] or config['insect']!=original['insect']:
            raise ValueError('Retain original classifiers, views and calibration for this comparison')
    if original['species']['model_sha256']!=digest(species_path) or original['insect']['model_sha256']!=digest(insect_path):
        raise ValueError('Classifier artifact identity mismatch')
    metadata=dict(mode='known_species_val' if known else 'viewed_binary_regression' if binary else 'unsupported_positive_coverage',
        manifest_sha256=digest(manifest),old_weights_sha256=digest(old_weights),new_weights_sha256=digest(args.new_weights),
        old_profile_sha256=digest(old_profile_path),new_profile_sha256=digest(args.profile),
        evaluator_sha256=digest(Path(__file__)),pipeline_sha256=digest(ROOT/'web/backend/app/services/pipeline.py'),
        service_sources={name:digest(ROOT/'web/backend/app/services'/name)
            for name in ('detector.py','classifier.py','insect_classifier.py','classification_views.py')})
    if args.proposal_weights:
        metadata.update(proposal_weights_sha256=digest(args.proposal_weights),
            verifier_source_sha256=digest(ROOT/'web/backend/app/services/verified_detector.py'))
    args.out.mkdir(parents=True,exist_ok=True); metadata_path=args.out/'metadata.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text('utf-8'))!=metadata:
        raise ValueError('Changed inputs; preserve old evidence and use a new output directory')
    save(metadata_path,metadata)
    species=SpeciesClassifier(str(species_path),str(ROOT/'data/tick_species/split_clean.json'),448,'center_full')
    insect=InsectClassifier(str(insect_path),str(ROOT/'data/insects_split.json'),view_mode='center_full')
    pipelines={}
    for name,weights,config in (('old',old_weights,original),('new',args.new_weights,profile)):
        if name=='new' and args.proposal_weights:
            from app.services.verified_detector import VerifiedDetector
            detector=VerifiedDetector(str(args.proposal_weights),str(weights),tile=2)
        else:
            detector=Detector(str(weights),tile=2)
        detector.model.roi_heads.mask_roi_pool=None
        if name=='new':
            detector.CONFIDENCE=profile['detector']['score_threshold']
        pipelines[name]=Pipeline(insect,detector,species,str(ROOT/'output/tick_species_clean_448/thresholds.csv'),{},config)
    if known and not {r['label'] for r in rows}<=set(species.classes):
        raise ValueError('Known-species labels outside retained model classes')
    if not known and not binary and any(r['label'] in species.classes for r in rows):
        raise ValueError('Coverage manifest must contain unsupported species')
    if binary and any(r['binary_label'] not in (0,1) for r in rows):
        raise ValueError('Binary regression requires explicit positive/negative image labels')
    trace_path=args.out/'predictions.json'; records=json.loads(trace_path.read_text('utf-8')) if trace_path.exists() else []
    if len(records)>len(rows) or any(r['path']!=row['path'] for r,row in zip(records,rows)):
        raise ValueError('Trace order changed')
    for index,row in enumerate(rows):
        if index<len(records):
            continue
        if digest(ROOT/row['path'])!=row['sha256']:
            raise ValueError('Frozen input changed')
        with Image.open(ROOT/row['path']) as source_image:
            image=ImageOps.exif_transpose(source_image).convert('RGB')
        record=dict(path=row['path'],label=row['label'])
        if binary:
            record['binary_label']=row['binary_label']
        for name,pipeline in pipelines.items():
            with patch('app.services.pipeline.load_rgb_image',return_value=image):
                result=pipeline.analyze(b'')
            record[name]=dict(detections=result['detections'],insect=result['insect'])
        records.append(record)
        if (index+1)%10==0:
            save(trace_path,records); print(f'{index+1}/{len(rows)}',flush=True)
    save(trace_path,records)
    summarize_results=(lambda records,name:summarize_binary(records,name)) if binary else (lambda records,name:summarize(records,name,known))
    report=dict(metadata=metadata,old=summarize_results(records,'old'),new=summarize_results(records,'new'),
        limitations=['positive folder image labels, not independently adjudicated per-box or count truth',
            'known-species val is development evidence; old models saw some sources',
            'correct candidate and wrong confirmation may coexist in one image; these are coverage counts, not per-box accuracy',
            'unsupported species have no correct class among the 28 deployed classes',
            'identical EXIF-normalized pixels, upload validation bypassed equally',
            'unsupported coverage must not be used to choose checkpoints or thresholds'])
    if binary:
        report['limitations'].append('Previously viewed same-source binary regression, not independent test or species/count truth')
    save(args.out/'report.json',report)
    print(json.dumps({name:{k:v for k,v in report[name].items() if k!='by_class'} for name in ('old','new')}),flush=True)


if __name__=='__main__':
    main()
