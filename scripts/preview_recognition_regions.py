"""Contact sheet of deterministic teacher regions for input quality review."""
import argparse
import json
import math
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'web/backend'))
from app.services.recognition_model import prepare_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--regions',type=Path,required=True)
    args = parser.parse_args()
    split = json.loads((args.data/'species_split.json').read_text('utf-8'))
    cache = json.loads((args.regions/'proposals.json').read_text('utf-8'))['boxes']
    selected, labels = [], set()
    for row in split['train']:
        boxes = cache.get(row['sha256'],[])
        if row['label'] not in labels and len(boxes)==1 and boxes[0]['score']>=.8:
            selected.append((row,boxes[0])); labels.add(row['label'])
        if len(selected)==16:
            break
    if not selected:
        raise ValueError('No cached qualifying proposals yet')
    sheet = Image.new('RGB',(960,math.ceil(len(selected)/4)*270),'white')
    draw = ImageDraw.Draw(sheet)
    for index,(row,proposal) in enumerate(selected):
        x,y = (index%4)*240,(index//4)*270
        with Image.open(ROOT/row['path']) as source:
            original = prepare_image(source)
            crop = prepare_image(source,proposal['bbox'])
        for offset,image in ((0,original),(120,crop)):
            thumb = ImageOps.contain(image,(116,225))
            sheet.paste(thumb,(x+offset+(116-thumb.width)//2,y+(225-thumb.height)//2))
        draw.text((x+2,y+229),row['label'][:31],fill='black')
        draw.text((x+2,y+245),f'whole / ROI  score={proposal["score"]}',fill='black')
    sheet.save(args.regions/'quality_preview.png')
    (args.regions/'quality_preview_cases.json').write_text(json.dumps(selected,indent=2),'utf-8')


if __name__=='__main__':
    main()
