"""Full localization-policy validation against the unchanged deployed cascade.

Uses grouped detector val/calibration. Old weights may have seen these sources;
folder negatives are proxies. Does not label detector boxes with image species.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'web/backend'))
sys.path.insert(0,str(ROOT/'src/train'))
from train_recognition_rebuild import sha256, save_json
from app.services.detector import Detector
from app.services.classifier import SpeciesClassifier
from app.services.insect_classifier import InsectClassifier
from app.services.recognition_classifier import RecognitionClassifier
from app.services.recognition_pipeline import score_proposals
from app.services.pipeline import Pipeline
from evaluate_optimized_cascade import match_boxes


class CachedDetector:
    tile = 2

    def detect(self,rgb,tile=None):
        return copy.deepcopy([d for d in self.row['full' if tile==1 else 'tiles'] if d['score']>=.5])


def read_image(row):
    with Image.open(ROOT/row['path']) as source:
        # Source COCO truth is raw; coordinate-free folder proxies use EXIF.
        return (ImageOps.exif_transpose(source) if row.get('negative_label') else source).convert('RGB')


def summarize(rows,traces,kind,cutoff=.5):
    tp=fp=fn=positives=proxy_count=proxy_positive=0
    error=0
    by_label={}
    for row,trace in zip(rows,traces):
        ds=trace['old'] if kind=='old' else [d for d in trace['new'] if d['tick_probability']>=cutoff and not d.get('coarse_rejected',False)]
        if row.get('negative_label'):
            proxy_count+=1; proxy_positive+=bool(ds)
            counts=by_label.setdefault(row['negative_label'],dict(images=0,positive_images=0))
            counts['images']+=1; counts['positive_images']+=bool(ds)
        elif row['annotations']:
            truth=[]
            for ann in row['annotations']:
                x,y,w,h=ann['bbox']; truth.append([x,y,x+w,y+h])
            a,b,c=match_boxes(ds,truth)
            tp+=a; fp+=b; fn+=c; positives+=1; error+=abs(len(ds)-len(truth))
    return dict(tp=tp,fp=fp,fn=fn,positive_images=positives,
        precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else 0,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,count_mae=error/positives if positives else None,
        proxy_images=proxy_count,proxy_positive_images=proxy_positive,
        proxy_positive_rate=proxy_positive/proxy_count if proxy_count else None,proxy_by_label=by_label)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--subset',choices=['val','calibration'],default='val')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--new-detection-threshold',type=float,default=.3)
    p.add_argument('--tick-threshold',type=float,default=.5)
    p.add_argument('--coarse-guard',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(4)
    if not (0<args.new_detection_threshold<=.5 and 0<=args.tick_threshold<=1):
        p.error('Require 0 < new detection threshold <= .5 and tick threshold in [0,1]')
    args.out.mkdir(parents=True,exist_ok=True)
    split=json.loads(args.split.read_text('utf-8'))
    rows=split[args.subset]
    detector_path=ROOT/'output/det_v3/mask_rcnn_ticks_v3.pth'
    metadata=dict(detector_sha256=sha256(detector_path),split_sha256=sha256(args.split),
        checkpoint_sha256=sha256(args.checkpoint),subset=args.subset,
        new_detection_threshold=args.new_detection_threshold,tick_threshold=args.tick_threshold,
        coarse_guard=args.coarse_guard,
        evaluator_sha256=sha256(Path(__file__)),
        model_sha256=sha256(ROOT/'web/backend/app/services/recognition_model.py'),
        classifier_sha256=sha256(ROOT/'web/backend/app/services/recognition_classifier.py'),
        candidate_policy_sha256=sha256(ROOT/'web/backend/app/services/recognition_pipeline.py'),
        old_pipeline_sha256=sha256(ROOT/'web/backend/app/services/pipeline.py'))
    cache_path=args.out/'detections.json'
    cache_metadata={k:metadata[k] for k in ('detector_sha256','split_sha256','subset')}
    cache=json.loads(cache_path.read_text('utf-8')) if cache_path.exists() else dict(metadata=cache_metadata,rows={})
    if cache['metadata']!=cache_metadata:
        raise ValueError('Detection cache identity changed')
    if any(row['path'] not in cache['rows'] for row in rows):
        detector=Detector(str(detector_path),tile=2)
        detector.model.roi_heads.mask_roi_pool=None
        detector.CONFIDENCE=.05
        for i,row in enumerate(rows):
            if row['path'] in cache['rows']:
                continue
            image=read_image(row); rgb=np.array(image)
            full=detector.detect(rgb,tile=1)
            tiles=detector.detect(rgb,tile=2) if not any(d['score']>=.5 for d in full) else []
            cache['rows'][row['path']]=dict(full=full,tiles=tiles)
            if (i+1)%20==0:
                save_json(cache_path,cache); print(f'detector cache {i+1}/{len(rows)}',flush=True)
        save_json(cache_path,cache)
        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    species_path=ROOT/'output/tick_species_clean_448/model/tick_species_best.pth'
    insect_path=ROOT/'output/insect_classifier/model/insect_best.pth'
    profile_path=ROOT/'configs/inference_optimization.json'
    species=SpeciesClassifier(str(species_path),str(ROOT/'data/tick_species/split_clean.json'),448,'center_full')
    insect=InsectClassifier(str(insect_path),str(ROOT/'data/insects_split.json'),view_mode='center_full')
    cached=CachedDetector()
    old=Pipeline(insect,cached,species,str(ROOT/'output/tick_species_clean_448/thresholds.csv'),{},
                 json.loads(profile_path.read_text('utf-8')))
    candidate=RecognitionClassifier(args.checkpoint)
    metadata.update(old_species_sha256=sha256(species_path),old_insect_sha256=sha256(insect_path),
                    old_profile_sha256=sha256(profile_path))
    trace_path=args.out/'traces.json'
    traces=json.loads(trace_path.read_text('utf-8')) if trace_path.exists() else dict(metadata=metadata,rows=[])
    if traces['metadata']!=metadata:
        raise ValueError('Comparison settings changed; use a new output directory')
    for i,row in enumerate(rows):
        if i<len(traces['rows']):
            if traces['rows'][i]['path']!=row['path']:
                raise ValueError('Trace ordering mismatch')
            continue
        image=read_image(row); rgb=np.array(image)
        cached.row=cache['rows'][row['path']]
        # Supply the same decoded pixels to both policies. Upload validation is
        # outside this model comparison; COCO coordinates must remain raw.
        with patch('app.services.pipeline.load_rgb_image',return_value=image):
            previous=old.analyze(b'')
        proposals=[d for d in cached.row['full'] if d['score']>=args.new_detection_threshold]
        if not proposals:
            proposals=[d for d in cached.row['tiles'] if d['score']>=args.new_detection_threshold]
        proposed=[]
        guard=dict(confidence=90,strong_presence=.95) if args.coarse_guard else None
        for detection,result,crop in score_proposals(image,proposals,candidate,insect,guard,old.insect_thresholds):
            proposed.append(dict(detection,tick_probability=result['tick_probability'],
                candidate_slug=result['slug'],species_confidence=result['confidence']))
        traces['rows'].append(dict(path=row['path'],old=previous['detections'],new=proposed))
        if (i+1)%20==0:
            save_json(trace_path,traces); print(f'cascade {i+1}/{len(rows)}',flush=True)
    save_json(trace_path,traces)
    old_result=summarize(rows,traces['rows'],'old')
    new_result=summarize(rows,traces['rows'],'new',args.tick_threshold)
    report=dict(metadata=metadata,old=old_result,new=new_result,
        gates=dict(recall_loss_at_most_2pp=new_result['recall']>=old_result['recall']-.02,
                   relative_proxy_fp_reduction_30pct=new_result['proxy_positive_rate']<=old_result['proxy_positive_rate']*.7),
        limitations=['val for selection; calibration not independent test','old weights may have seen these sources',
        'folder negative image labels are proxies','COCO empty annotations excluded from detection metrics',
        'upload size/decoder validation bypassed equally for model-policy comparison',
        'no per-box species truth in detector set','candidate bypasses old global/local coarse veto rules'])
    if args.subset=='val':
        report['validation_threshold_diagnostics']={str(t):summarize(rows,traces['rows'],'new',t)
            for t in (.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.98,.99,.995,.999)}
    save_json(args.out/'report.json',report)
    print(json.dumps(dict(old=old_result,new=new_result,gates=report['gates']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
