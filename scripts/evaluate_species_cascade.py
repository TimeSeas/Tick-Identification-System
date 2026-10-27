"""Compare species weights inside the unchanged, actual balanced cascade.

Only specialist/calibration may select a species threshold. Other populations
are frozen-policy regressions, and image labels never become box ground truth.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image,ImageOps
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'web/backend'))
from app.services.detector import Detector
from app.services.classifier import SpeciesClassifier
from app.services.insect_classifier import InsectClassifier
from app.services.classification_views import probabilities,prediction
from app.services.pipeline import Pipeline
from evaluate_species_specialist import select_threshold,gate_metrics
from prepare_species_specialist import digest,save
from species_threshold_calibration import calibrate_class_thresholds
from evaluate_recognition_cascade import CachedDetector,summarize as summarize_boxes
from evaluate_detector_labelled_images import summarize,summarize_binary
from evaluate_optimized_cascade import CachedClassifier


class RecordingSpecies(SpeciesClassifier):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.cache={};self.records=[]

    def predict(self,rgb,whole_rgb=None,whole_weight=0.):
        hasher=hashlib.sha256(rgb.tobytes())
        hasher.update(str((rgb.shape,whole_weight)).encode())
        if whole_rgb is not None:
            hasher.update(str(whole_rgb.shape).encode());hasher.update(whole_rgb.tobytes())
        key=hasher.hexdigest()
        if key not in self.cache:
            scores=probabilities(self.model,rgb,self.input_size,self.device,self.view_mode)
            if whole_rgb is not None and whole_weight>0:
                scores=(1-whole_weight)*scores+whole_weight*probabilities(
                    getattr(self,'whole_model',None) or self.model,whole_rgb,
                    self.input_size,self.device,self.view_mode)
            self.cache[key]=(prediction(scores,self.classes),scores.tolist())
        result,scores=self.cache[key];self.records.append(scores)
        return copy.deepcopy(result)


def load_rows(data,kind,subset):
    source=data[subset] if kind in ('specialist','known','detector') else data if kind=='binary' else data['rows']
    if kind=='specialist':
        source=[dict(r,label=r['source_species']) for r in source if r['bbox'] is None]
    if not source or len(source)!=len({r['path'] for r in source}):
        raise ValueError('Require nonempty unique source images')
    return source


def calibration_rows(row,scores,classes):
    label=classes.index(row['label']) if row['label'] in classes else -1
    if not scores:
        return [dict(path=row['path'],label=label,detected=False,probabilities=[0.]*len(classes))]
    return [dict(path=row['path'],label=label,probabilities=p) for p in scores]


def validate_policy_binding(policy,profile_sha,model_sha,manifest_sha):
    for key,value in (('profile_sha256',profile_sha),('candidate_sha256',model_sha),('manifest_sha256',manifest_sha)):
        if policy.get(key)!=value:raise ValueError('Frozen policy mismatch '+key)


def validate_classifier_contract(profile,species_split_sha,insect_split_sha,classes,insect_classes):
    for name,expected,labels in (('species',species_split_sha,classes),('insect',insect_split_sha,insect_classes)):
        if profile[name]['split_sha256']!=expected:raise ValueError('Classifier split identity mismatch '+name)
        if set(profile[name]['thresholds'])!=set(labels):raise ValueError('Classifier threshold classes mismatch '+name)
    if set(profile['species']['region_thresholds'])!=set(classes):raise ValueError('Region threshold classes mismatch')


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--kind',choices=('specialist','known','detector','binary','unsupported'),required=True)
    p.add_argument('--subset',choices=('val','calibration','holdout'),default='val')
    p.add_argument('--weights',type=Path,required=True,help='Raw frozen candidate species state_dict')
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--calibrate',action='store_true')
    p.add_argument('--unknown-constraint',choices=('absolute5','relative_half'),default='absolute5')
    p.add_argument('--threshold-policy',choices=('global','classwise'),default='global')
    p.add_argument('--policy',type=Path,help='Frozen policy record mandatory for specialist holdout')
    p.add_argument('--detector-cache',type=Path,help='This evaluator cache directory, identities checked')
    args=p.parse_args();torch.set_num_threads(4)
    if args.out.exists():raise ValueError('Require a new output directory')
    if args.calibrate and (args.kind!='specialist' or args.subset!='calibration'):
        raise ValueError('Only specialist calibration population may select thresholds')
    baseline_path=ROOT/'configs/inference_detector_balanced.json'
    baseline=json.loads(baseline_path.read_text('utf-8'));profile=json.loads(args.profile.read_text('utf-8'))
    detector_path=ROOT/'output/optimization_20261015/rebuild_20261024/detector_v3_finetune_run01/frozen_detector.pth'
    old_path=ROOT/'output/tick_species_clean_448/model/tick_species_best.pth'
    insect_path=ROOT/'output/insect_classifier/model/insect_best.pth'
    species_split=ROOT/'data/tick_species/split_clean.json';insect_split=ROOT/'data/insects_split.json'
    classes=sorted({r['label'] for r in json.loads(species_split.read_text('utf-8'))['train']})
    insect_classes=sorted({r['label'] for r in json.loads(insect_split.read_text('utf-8'))['train']})
    for config in (baseline,profile):
        validate_classifier_contract(config,digest(species_split),digest(insect_split),classes,insect_classes)
    if baseline['detector']!=profile['detector'] or baseline['insect']!=profile['insect'] or baseline['rejection']!=profile['rejection']:
        raise ValueError('Freeze detector, insect and veto policy for species comparison')
    for field in ('input_size','view_mode','region_padding','single_image_weight'):
        if baseline['species'][field]!=profile['species'][field]:raise ValueError('Changed preprocessing '+field)
    for path,expected in ((detector_path,baseline['detector']['model_sha256']),
        (old_path,baseline['species']['model_sha256']),(insect_path,baseline['insect']['model_sha256']),
        (args.weights,profile['species']['model_sha256'])):
        if digest(path)!=expected:raise ValueError('Changed model '+str(path))
    reference=profile['species'].get('whole_reference')
    if reference and reference['model_sha256']!=digest(old_path):
        raise ValueError('Whole-image reference must be the frozen original species model')
    data=json.loads(args.manifest.read_text('utf-8'));rows=load_rows(data,args.kind,args.subset)
    if args.kind=='specialist' and args.subset=='holdout':
        if args.calibrate or not args.policy:
            raise ValueError('Holdout requires a frozen policy from the matching calibration manifest')
        frozen=json.loads(args.policy.read_text('utf-8'))
        validate_policy_binding(frozen,digest(args.profile),digest(args.weights),digest(args.manifest))
        for item in frozen['artifacts']:
            if digest(Path(item['path']))!=item['sha256']:raise ValueError('Frozen calibration evidence changed')
        if frozen['evaluator_sha256']!=digest(__file__):raise ValueError('Frozen evaluator source changed')
        if frozen['calibrator_sha256']!=digest(ROOT/'scripts/species_threshold_calibration.py'):
            raise ValueError('Frozen calibrator source changed')
        for name,expected in frozen['service_sources'].items():
            if digest(ROOT/'web/backend/app/services'/name)!=expected:raise ValueError('Frozen inference source changed')
    for row in rows:
        if digest(ROOT/row['path'])!=row['sha256']:raise ValueError('Changed image '+row['path'])
    cache_identity=dict(manifest_sha256=digest(args.manifest),kind=args.kind,subset=args.subset,
        detector_sha256=digest(detector_path),detector_source_sha256=digest(ROOT/'web/backend/app/services/detector.py'),
        score=baseline['detector']['score_threshold'],min_size=1024,max_size=1200,tile=2,dedup=.3,
        decoder='raw_coco_else_exif_rgb',evaluator_sha256=digest(__file__))
    args.out.mkdir(parents=True)
    save(args.out/'metadata.json',dict(cache_identity=cache_identity,candidate_sha256=digest(args.weights),
        profile_sha256=digest(args.profile),baseline_profile_sha256=digest(baseline_path),
        species_split_sha256=digest(species_split),insect_split_sha256=digest(insect_split),
        frozen_policy_sha256=digest(args.policy) if args.policy else None,
        calibrator_sha256=digest(ROOT/'scripts/species_threshold_calibration.py'),
        sources={name:digest(ROOT/'web/backend/app/services'/name) for name in
            ('pipeline.py','classifier.py','classification_views.py','insect_classifier.py')}))
    def read(row):
        with Image.open(ROOT/row['path']) as im:
            return (im if args.kind=='detector' and not row.get('negative_label') else ImageOps.exif_transpose(im)).convert('RGB')
    if args.detector_cache:
        previous=json.loads((args.detector_cache/'metadata.json').read_text('utf-8'))['cache_identity']
        if previous!=cache_identity:raise ValueError('Detector cache identity mismatch')
        cache=json.loads((args.detector_cache/'detections.json').read_text('utf-8'))
    else:
        cache={};detector=Detector(str(detector_path),tile=2)
        detector.model.roi_heads.mask_roi_pool=None;detector.CONFIDENCE=cache_identity['score']
        for i,row in enumerate(rows,1):
            rgb=np.array(read(row));full=detector.detect(rgb,tile=1)
            cache[row['path']]=dict(full=full,tiles=detector.detect(rgb,tile=2) if not full else [])
            if i%20==0:save(args.out/'detections.json',cache);print(f'detect {i}/{len(rows)}',flush=True)
        del detector;torch.cuda.empty_cache()
    if set(cache)!={r['path'] for r in rows}:raise ValueError('Incomplete detector cache')
    save(args.out/'detections.json',cache)
    cached=CachedDetector();cached.CONFIDENCE=cache_identity['score']
    insect=InsectClassifier(str(insect_path),str(ROOT/'data/insects_split.json'),view_mode='center_full')
    insect=CachedClassifier(insect,'insect',digest(insect_path),{})
    records=[dict(path=r['path'],label=r.get('label'),**({'binary_label':r['binary_label']} if args.kind=='binary' else {})) for r in rows]
    raw_new=[];chosen=None
    for name,path,config in (('old',old_path,baseline),('new',args.weights,profile)):
        species=RecordingSpecies(str(path),str(species_split),448,'center_full',
            whole_model_path=str(old_path) if name=='new' and reference else None)
        pipeline=Pipeline(insect,cached,species,str(ROOT/'output/tick_species_clean_448/thresholds.csv'),{},config)
        def analyze(row):
            cached.row=cache[row['path']];species.records=[]
            with patch('app.services.pipeline.load_rgb_image',return_value=read(row)):
                result=pipeline.analyze(b'')
            return dict(detections=result['detections'],insect=result['insect'])
        for i,(row,record) in enumerate(zip(rows,records),1):
            record[name]=analyze(row)
            if name=='new' and args.calibrate:
                raw_new.extend(calibration_rows(row,species.records,species.classes))
            if i%20==0:save(args.out/'predictions.json',records[:i]);print(f'{name} {i}/{len(rows)}',flush=True)
        if name=='new' and args.calibrate:
            limit=.05
            if args.unknown_constraint=='relative_half':
                old_unknown=[r for r in records if r['label'] not in species.classes]
                limit=summarize(old_unknown,'old',False)['confirmed_images']/len(old_unknown)*.5
            chosen=select_threshold(raw_new,max_unknown_acceptance=limit)
            if args.threshold_policy=='classwise':
                fallback=min(baseline['species']['thresholds'].values())/100
                chosen=calibrate_class_thresholds(raw_new,species.classes,fallback,
                    min_threshold=fallback,max_unknown_acceptance=limit)
            save(args.out/'calibration_records.json',raw_new)
            if chosen:
                table=chosen.get('thresholds',{c:chosen.get('threshold') for c in species.classes})
                profile['species']['thresholds']={c:t*100 for c,t in table.items()}
                profile['species']['region_thresholds']=dict(profile['species']['thresholds'])
                for field in ('calibration','region_calibration','calibration_population','region_calibration_population'):
                    profile['species'].pop(field,None)
                profile['species']['specialist_calibration']=dict(manifest_sha256=digest(args.manifest),
                    selection_population='pre-veto native proposals; folder species labels are proxies',
                    unknown_constraint=args.unknown_constraint,max_unknown_acceptance=limit,**chosen)
                save(args.out/'candidate_profile.json',profile)
                pipeline=Pipeline(insect,cached,species,str(ROOT/'output/tick_species_clean_448/thresholds.csv'),{},profile)
                for row,record in zip(rows,records):record['new']=analyze(row)
        del species,pipeline;torch.cuda.empty_cache()
    save(args.out/'predictions.json',records)
    reports={}
    for name in ('old','new'):
        if args.kind=='detector':reports[name]=summarize_boxes(rows,[dict(old=r[name]['detections']) for r in records],'old')
        elif args.kind=='binary':reports[name]=summarize_binary(records,name)
        elif args.kind=='specialist':
            reports[name]={}
            for known in (True,False):
                selected=[r for r in records if (r['label'] in baseline['species']['thresholds'])==known]
                reports[name]['known' if known else 'unknown']=summarize(selected,name,known) if selected else None
        else:reports[name]=summarize(records,name,args.kind=='known')
    save(args.out/'report.json',dict(results=reports,calibration=chosen,
        limitations=['Development/retrospective evidence; old model saw some sources',
            'Image-level folder labels are not per-box species truth',
            'Calibration includes pre-veto proposals to avoid filtering by the selected threshold',
            'Unknown no-detection images are in the false-confirmation denominator; report detection coverage separately']))
    if args.calibrate and chosen:
        frozen=dict(profile_sha256=digest(args.out/'candidate_profile.json'),candidate_sha256=digest(args.weights),
            manifest_sha256=digest(args.manifest),evaluator_sha256=digest(__file__),
            calibrator_sha256=digest(ROOT/'scripts/species_threshold_calibration.py'),
            service_sources={name:digest(ROOT/'web/backend/app/services'/name) for name in
                ('pipeline.py','classifier.py','classification_views.py','insect_classifier.py','detector.py')},
            artifacts=[dict(path=str(path.resolve()),sha256=digest(path))
                for path in (args.out/'report.json',args.out/'calibration_records.json',args.out/'metadata.json')])
        save(args.out/'frozen_policy.json',frozen)
    print(json.dumps(reports,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
