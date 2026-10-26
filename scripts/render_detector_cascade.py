"""Render all pre-training frozen cases from actual detector-cascade traces."""
import argparse
import json
from pathlib import Path

from PIL import Image,ImageDraw,ImageOps

from build_recognition_profile import digest
from recognition_detection_gallery import draw_boxes

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--variant',choices=('drop_in','regional_only','detector_only'),required=True)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True)
    args=p.parse_args()
    metadata=json.loads((args.comparison/'metadata.json').read_text('utf-8'))
    manifest=json.loads(args.split.read_text('utf-8')); cases=json.loads(args.cases.read_text('utf-8'))
    if metadata['split_sha256']!=digest(args.split) or cases['manifest_sha256']!=manifest['metadata']['source_sha256']:
        raise ValueError('Comparison and frozen case source manifests differ')
    trace_path=args.comparison/(args.variant+'_traces.json')
    records=json.loads(trace_path.read_text('utf-8')); traces={r['path']:r for r in records}
    if len(traces)!=len(records) or set(traces)!={r['path'] for r in manifest['val']}:
        raise ValueError('Require complete unique val traces')
    out=args.comparison/(args.variant+'_gallery')
    if out.exists():
        raise ValueError('Preserve rendered evidence; choose a new comparison directory')
    out.mkdir(); pages=[]
    for start in range(0,len(cases['cases']),4):
        selected=cases['cases'][start:start+4]
        sheet=Image.new('RGB',(1060,50+len(selected)*310),'white'); draw=ImageDraw.Draw(sheet)
        draw.text((12,8),'Full service policy comparison: previous / candidate '+args.variant,fill='black')
        for offset,case in enumerate(selected):
            row=case['row']; trace=traces[row['path']]
            if digest(ROOT/row['path'])!=row['sha256']:
                raise ValueError('Frozen image changed')
            with Image.open(ROOT/row['path']) as source:
                image=(ImageOps.exif_transpose(source) if row.get('negative_label') else source).convert('RGB')
            expected=('proxy non-tick: '+row['negative_label']) if row.get('negative_label') else 'annotated ticks: '+str(len(row['annotations']))
            y=40+offset*310
            draw.text((12,y),case['kind']+' | '+expected,fill='black')
            for x,key,label,color in ((12,'old','Previous','#b91c1c'),(542,'new','Candidate','#166534')):
                boxes=trace[key]; preview=draw_boxes(image,boxes,color)
                sheet.paste(preview,(x+(510-preview.width)//2,y+25))
                draw.text((x,y+286),f'{label}: {len(boxes)} tick detections',fill=color)
        name=f'cascade_{start//4+1:02d}.png'; sheet.save(out/name); pages.append(name)
    (out/'gallery.json').write_text(json.dumps(dict(cases_sha256=digest(args.cases),traces_sha256=digest(trace_path),
        pages=pages,note='All frozen cases; policy colors do not indicate correctness. No per-box species truth.'),indent=2),'utf-8')
    print(f'Rendered {len(pages)} pages with every frozen case')


if __name__=='__main__':
    main()
