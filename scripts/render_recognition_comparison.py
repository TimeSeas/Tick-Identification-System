"""Render frozen same-input component cases, including failures and regressions."""
import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'web/backend'))
from app.services.recognition_model import prepare_image


def text_block(record,classes):
    tick = record['tick_probability']>=.5
    correct = tick == bool(record['tick_label'])
    lines = [f'Tick: {tick} ({record["tick_probability"]:.3f})']
    if record['species_label']>=0:
        lines += [classes[record['species_prediction']],f'species confidence {record["species_confidence"]:.3f}']
        correct = correct and record['species_prediction']==record['species_label']
    else:
        lines += ['Species truth: unavailable']
    return lines, '#166534' if correct else '#b91c1c'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True)
    args=p.parse_args()
    cases=json.loads(args.cases.read_text('utf-8'))
    import hashlib
    if cases['manifest_sha256']!=hashlib.sha256(args.split.read_bytes()).hexdigest():
        raise ValueError('Gallery manifest changed')
    split=json.loads(args.split.read_text('utf-8'))
    old={r['index']:r for r in json.loads((args.comparison/'old_predictions.json').read_text('utf-8'))}
    new={r['index']:r for r in json.loads((args.comparison/'new_predictions.json').read_text('utf-8'))}
    pages=[]
    for start in range(0,len(cases['cases']),12):
        items=cases['cases'][start:start+12]
        sheet=Image.new('RGB',(1100,75+len(items)*160),'white')
        draw=ImageDraw.Draw(sheet)
        draw.text((12,8),'Frozen validation component comparison / source labels may be proxies',fill='black')
        draw.text((12,30),'Source view',fill='black'); draw.text((300,30),'Expected',fill='black')
        draw.text((565,30),'Old component',fill='black'); draw.text((835,30),'New component',fill='black')
        for offset,item in enumerate(items):
            row,index=item['row'],item['index']; y=65+offset*160
            with Image.open(ROOT/row['path']) as source:
                image=prepare_image(source,row['bbox'],normalize_exif=row['coordinate_frame']!='raw')
            thumbnail=ImageOps.contain(image,(270,145))
            sheet.paste(thumbnail,(12+(270-thumbnail.width)//2,y+(145-thumbnail.height)//2))
            expected=row['species_label'] or ('tick (species unknown)' if row['tick_label'] else row['negative_label'])
            draw.text((300,y+30),expected,fill='black')
            draw.text((300,y+55),row['sample_type'],fill='black')
            draw.text((300,y+80),f'val row {index}',fill='black')
            for x,record in ((565,old[index]),(835,new[index])):
                lines,color=text_block(record,split['classes'])
                for line_index,line in enumerate(lines):
                    draw.text((x,y+30+line_index*22),line,fill=color)
            draw.line((0,y+154,1100,y+154),fill='#dddddd')
        name=f'comparison_{start//12+1:02d}.png'
        sheet.save(args.comparison/name); pages.append(name)
    (args.comparison/'gallery.json').write_text(json.dumps(dict(pages=pages,cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest()),indent=2),'utf-8')
    print(f'{len(pages)} pages rendered')


if __name__=='__main__':
    main()
