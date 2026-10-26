"""CPU comparison of saved full-frame detector outputs on identical val pixels.

Does not compare cascades or add tiles. Old task weights saw some source groups;
folder negatives are proxies. Frozen gallery is rendered in full, including errors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/train'))
from train_detector_rebuild import match_detections


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows, predictions):
    tp=fp=fn=error=positive=negative=negative_positive=0
    by_label={}
    for row in rows:
        boxes,scores=predictions[row['path']]
        if row.get('negative_label'):
            negative+=1; negative_positive+=bool(boxes)
            item=by_label.setdefault(row['negative_label'],dict(images=0,positive_images=0))
            item['images']+=1; item['positive_images']+=bool(boxes)
        else:
            truth=[[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in row['annotations']]
            a,b,c=match_detections(boxes,scores,truth)
            tp+=a; fp+=b; fn+=c; positive+=1; error+=abs(len(boxes)-len(truth))
    return dict(tp=tp,fp=fp,fn=fn,recall=tp/(tp+fn) if tp+fn else 0,
        precision=tp/(tp+fp) if tp+fp else 0,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
        positive_images=positive,count_mae=error/positive if positive else None,
        proxy_images=negative,proxy_positive_images=negative_positive,
        proxy_positive_rate=negative_positive/negative if negative else None,proxy_by_label=by_label)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--baseline-cache',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise ValueError('Preserve prior comparisons; use a new output directory')
    manifest=json.loads(args.split.read_text('utf-8')); rows=manifest['val']
    baseline=json.loads(args.baseline_cache.read_text('utf-8'))
    if baseline['metadata']['subset']!='val':
        raise ValueError('Require val baseline')
    expected=manifest.get('metadata',{}).get('source_sha256',digest(args.split))
    if baseline['metadata']['split_sha256']!=expected:
        raise ValueError('Baseline source split differs from supervision manifest')
    saved=json.loads(args.predictions.read_text('utf-8'))
    if len(saved)!=len(rows) or {r['path'] for r in saved}!={r['path'] for r in rows}:
        raise ValueError('Prediction set must contain every supervision val row exactly once')
    new={r['path']:(r['boxes'],r['scores']) for r in saved}
    old={}
    for row in rows:
        ds=[d for d in baseline['rows'][row['path']]['full'] if d['score']>=.5]
        old[row['path']]=([d['bbox'] for d in ds],[d['score'] for d in ds])
    previous,current=metrics(rows,old),metrics(rows,new)
    cases=json.loads(args.cases.read_text('utf-8'))
    if cases['manifest_sha256']!=expected:
        raise ValueError('Frozen gallery source mismatch')
    if any(c['row']['path'] not in new for c in cases['cases']):
        raise ValueError('Every frozen case must be present')
    args.out.mkdir(parents=True)
    report=dict(metadata=dict(split_sha256=digest(args.split),predictions_sha256=digest(args.predictions),
        baseline_cache_sha256=digest(args.baseline_cache),cases_sha256=digest(args.cases),
        evaluator_sha256=digest(Path(__file__)),score_threshold=.5,matching_iou=.5,dedup_iou=.3),
        old_full_frame=previous,new_full_frame=current,
        gates=dict(recall_loss_at_most_2pp=current['recall']>=previous['recall']-.02,
            proxy_fp_reduction_at_least_30pct=current['proxy_positive_rate']<=previous['proxy_positive_rate']*.7,
            count_mae_not_worse=current['count_mae']<=previous['count_mae']),
        limitations=['full-frame detector only, not tiles or full service cascade',
            'old detector saw some new val source groups; diagnostic comparison',
            'folder negative labels are proxies, not human-confirmed absence'])
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
    for start in range(0,len(cases['cases']),4):
        selected=cases['cases'][start:start+4]
        sheet=Image.new('RGB',(1260,40+len(selected)*280),'white'); draw=ImageDraw.Draw(sheet)
        draw.text((10,8),'Full-frame only: annotated/proxy | old detector | retrained detector',fill='black')
        for offset,case in enumerate(selected):
            row=case['row']; y=35+offset*280
            with Image.open(ROOT/row['path']) as source:
                image=(ImageOps.exif_transpose(source) if row.get('negative_label') else source).convert('RGB')
            truth=[[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in row['annotations']]
            for column,(boxes,label,color) in enumerate(((truth,case['kind'],'#2563eb'),
                (old[row['path']][0],'Old','#b91c1c'),(new[row['path']][0],'New','#166534'))):
                preview=image.copy(); pen=ImageDraw.Draw(preview)
                for box in boxes:
                    pen.rectangle(tuple(box),outline=color,width=max(2,round(max(image.size)/250)))
                preview=ImageOps.contain(preview,(400,240)); x=column*420+10
                sheet.paste(preview,(x+(400-preview.width)//2,y+20))
                draw.text((x,y),f'{label}: {len(boxes)} boxes',fill=color)
        sheet.save(args.out/f'full_frame_{start//4+1:02d}.png')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
