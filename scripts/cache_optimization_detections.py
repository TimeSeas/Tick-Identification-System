"""Cache validation detections for reproducible inference-policy experiments."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/optimization_20261015'
spec=importlib.util.spec_from_file_location('baseline_detector',OUT/'baseline_app/services/detector.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--subset',choices=['val','test'],default='val');parser.add_argument('--species-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4)
    detector=module.Detector(str(ROOT/'output/det_v3/mask_rcnn_ticks_v3.pth'),tile=2);detector.CONFIDENCE=.05
    records=[]
    if args.species_only:
        split=json.loads((ROOT/'data/tick_species/split_clean.json').read_text('utf-8'))
        records=[dict(x,task='species') for x in split[args.subset]]
        filename=f'species_{args.subset}_detections.json'
    else:
        det_subset='valid' if args.subset=='val' else 'test'
        annotation=json.loads((ROOT/f'ticks image detection.v5i.coco/{det_subset}/_annotations.coco.json').read_text('utf-8'))
        for image in annotation['images']:
            boxes=[]
            for a in annotation['annotations']:
                if a['image_id']==image['id']:
                    x,y,w,h=a['bbox'];boxes.append([x,y,x+w,y+h])
            records.append({'task':'detection','path':f'ticks image detection.v5i.coco/{det_subset}/{image["file_name"]}','boxes':boxes})
        split=json.loads((ROOT/'data/insects_split.json').read_text('utf-8'))
        classes=sorted({r['label'] for r in split['train']} - {'tick'})
        # Predetermined balanced negatives, not cherry-picked after observing predictions.
        for label in classes:
            selected=sorted([r for r in split[args.subset] if r['label']==label],key=lambda r:r['path'])[:5]
            records.extend(dict(r,task='negative',boxes=[]) for r in selected)
        filename=f'detection_{args.subset}_cache.json'
    path=OUT/filename
    cache=json.loads(path.read_text('utf-8')) if path.exists() else []
    seen={r['path'] for r in cache}
    for i,row in enumerate(records):
        if row['path'] in seen:continue
        with Image.open(ROOT/row['path']) as image:rgb=np.array(image.convert('RGB'))
        h,w=rgb.shape[:2]
        item=dict(row,size=[w,h],full=detector._detect(rgb),tiles=[])
        if not args.species_only:
            th,tw=int(np.ceil(h/2)),int(np.ceil(w/2))
            for y in [0,th]:
                for x in [0,tw]:
                    if y>=h or x>=w:continue
                    for d in detector._detect(rgb[y:min(y+th,h),x:min(x+tw,w)]):
                        b=d['bbox'];d['bbox']=[b[0]+x,b[1]+y,b[2]+x,b[3]+y];item['tiles'].append(d)
        cache.append(item)
        if i%20==0:
            path.write_text(json.dumps(cache,ensure_ascii=False),encoding='utf-8')
            print(filename,'processed',i+1,'/',len(records),flush=True)
    path.write_text(json.dumps(cache,ensure_ascii=False),encoding='utf-8')
    print('complete',filename,len(cache),flush=True)

if __name__=='__main__':main()
