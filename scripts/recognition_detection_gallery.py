"""Freeze full-image cases before inference, then render every frozen result."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT=Path(__file__).resolve().parents[1]


def freeze(rows):
    selected=[]; counts=Counter()
    for index,row in enumerate(rows):
        annotations=row['annotations']
        if row.get('negative_label'):
            kind='negative_'+row['negative_label']
            limit=1
        elif len(annotations)>1:
            kind='multiple'; limit=4
        elif annotations:
            area=annotations[0]['bbox'][2]*annotations[0]['bbox'][3]
            kind='small' if area<32**2 else 'single'; limit=4
        else:
            continue
        if counts[kind]<limit:
            selected.append(dict(index=index,row=row,kind=kind)); counts[kind]+=1
    return selected


def draw_boxes(image,boxes,color):
    image=image.copy(); draw=ImageDraw.Draw(image)
    width=max(2,round(max(image.size)/250))
    for d in boxes:
        draw.rectangle(tuple(d['bbox']),outline=color,width=width)
    return ImageOps.contain(image,(510,260))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--comparison',type=Path)
    args=p.parse_args()
    raw=args.split.read_bytes(); identity=hashlib.sha256(raw).hexdigest()
    split=json.loads(raw)
    if args.comparison is None:
        if args.cases.exists():
            raise ValueError('Frozen cases cannot be overwritten')
        selected=freeze(split['val'])
        args.cases.write_text(json.dumps(dict(manifest_sha256=identity,subset='val',
            selection='first four multi/small/single and first proxy per class in frozen ordering',
            cases=selected),ensure_ascii=False,indent=2),'utf-8')
        print(f'Frozen {len(selected)} full-image cases'); return
    cases=json.loads(args.cases.read_text('utf-8'))
    if cases['manifest_sha256']!=identity:
        raise ValueError('Case manifest mismatch')
    traces=json.loads((args.comparison/'traces.json').read_text('utf-8'))
    if traces['metadata']['split_sha256']!=identity or traces['metadata']['subset']!='val':
        raise ValueError('Trace split mismatch')
    threshold=traces['metadata']['tick_threshold']
    pages=[]
    for start in range(0,len(cases['cases']),4):
        selected=cases['cases'][start:start+4]
        sheet=Image.new('RGB',(1060,50+len(selected)*310),'white'); draw=ImageDraw.Draw(sheet)
        draw.text((12,8),'Frozen full-image localization comparison: old / new',fill='black')
        for offset,item in enumerate(selected):
            row=item['row']; trace=traces['rows'][item['index']]
            if trace['path']!=row['path']:
                raise ValueError('Trace order changed')
            with Image.open(ROOT/row['path']) as source:
                image=(ImageOps.exif_transpose(source) if row.get('negative_label') else source).convert('RGB')
            new=[d for d in trace['new'] if d['tick_probability']>=threshold and not d.get('coarse_rejected',False)]
            expected='proxy non-tick: '+row['negative_label'] if row.get('negative_label') else 'annotated ticks: '+str(len(row['annotations']))
            y=40+offset*310
            draw.text((12,y),f'{item["kind"]} | {expected} | val row {item["index"]}',fill='black')
            for x,boxes,label,color in ((12,trace['old'],'Old','#b91c1c'),(542,new,'New','#166534')):
                preview=draw_boxes(image,boxes,color)
                sheet.paste(preview,(x+(510-preview.width)//2,y+25))
                draw.text((x,y+286),f'{label}: {len(boxes)} tick proposals',fill=color)
        name=f'localization_{start//4+1:02d}.png'; sheet.save(args.comparison/name); pages.append(name)
    (args.comparison/'localization_gallery.json').write_text(json.dumps(dict(pages=pages,
        cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        note='All frozen cases shown; colors identify policy, not correctness'),indent=2),'utf-8')
    print(f'Rendered {len(pages)} pages')


if __name__=='__main__':
    main()
