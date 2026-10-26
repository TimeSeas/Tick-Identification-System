"""One frozen same-source binary comparison; not an independent real-world test."""
import argparse
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'web/backend'))
from app.services.detector import Detector
from app.services.classifier import SpeciesClassifier
from app.services.insect_classifier import InsectClassifier
from app.services.recognition_classifier import RecognitionClassifier
from app.services.recognition_pipeline import RecognitionPipeline
from app.services.pipeline import Pipeline
from evaluate_recognition_cascade import CachedDetector
from build_recognition_profile import digest


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(); torch.set_num_threads(4)
    if args.out.exists():
        raise ValueError('Frozen same-source comparison cannot overwrite prior results')
    rows=json.loads(args.manifest.read_text('utf-8'))
    profile=json.loads(args.profile.read_text('utf-8'))
    if profile['recognition']['model_sha256']!=digest(args.checkpoint):
        raise ValueError('Profile/checkpoint mismatch')
    detector=Detector(str(ROOT/'output/det_v3/mask_rcnn_ticks_v3.pth'),tile=2)
    detector.model.roi_heads.mask_roi_pool=None
    species=SpeciesClassifier(str(ROOT/'output/tick_species_clean_448/model/tick_species_best.pth'),
        str(ROOT/'data/tick_species/split_clean.json'),448,'center_full')
    insect=InsectClassifier(str(ROOT/'output/insect_classifier/model/insect_best.pth'),
        str(ROOT/'data/insects_split.json'),view_mode='center_full')
    recognition=RecognitionClassifier(args.checkpoint)
    cached=CachedDetector()
    old_profile=json.loads((ROOT/'configs/inference_optimization.json').read_text('utf-8'))
    old=Pipeline(insect,cached,species,str(ROOT/'output/tick_species_clean_448/thresholds.csv'),{},old_profile)
    new=RecognitionPipeline(insect,cached,recognition,profile,{},species_classifier=species)
    records=[]
    for index,row in enumerate(rows):
        if digest(ROOT/row['path'])!=row['sha256']:
            raise ValueError('Frozen input changed')
        with Image.open(ROOT/row['path']) as source:
            image=ImageOps.exif_transpose(source).convert('RGB')
        rgb=np.array(image); buffer=io.BytesIO(); image.save(buffer,format='PNG',exif=b'')
        full=detector.detect(rgb,tile=1)
        tiles=detector.detect(rgb,tile=2) if not full else []
        cached.row=dict(full=full,tiles=tiles)
        start=time.perf_counter(); previous=old.analyze(buffer.getvalue()); old_seconds=time.perf_counter()-start
        start=time.perf_counter(); current=new.analyze(buffer.getvalue()); new_seconds=time.perf_counter()-start
        records.append(dict(path=row['path'],label=row['label'],binary_label=row['binary_label'],
            old_count=len(previous['detections']),new_count=len(current['detections']),
            old_insect=previous['insect'],new_insect=current['insect'],
            old_detections=previous['detections'],new_detections=current['detections'],
            old_classifier_seconds=old_seconds,new_classifier_seconds=new_seconds))
        if (index+1)%20==0:
            print(f'same source {index+1}/{len(rows)}',flush=True)
    metrics={}
    for name in ('old','new'):
        tp=sum(bool(r[name+'_count']) and r['binary_label']==1 for r in records)
        fn=sum(not r[name+'_count'] and r['binary_label']==1 for r in records)
        fp=sum(bool(r[name+'_count']) and r['binary_label']==0 for r in records)
        tn=sum(not r[name+'_count'] and r['binary_label']==0 for r in records)
        metrics[name]=dict(tp=tp,fn=fn,fp=fp,tn=tn,recall=tp/(tp+fn),proxy_fpr=fp/(fp+tn))
    args.out.mkdir(parents=True)
    (args.out/'predictions.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),'utf-8')
    report=dict(manifest_sha256=digest(args.manifest),profile_sha256=digest(args.profile),
        model_sha256=digest(args.checkpoint),metrics=metrics,
        limitations=['same-source retrospective, historical-source checks incomplete',
        'image binary labels only, no count or species ground truth',
        'cached detection, classifier timing is not end-to-end request latency',
        'frozen policy; these results must not tune thresholds'])
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
