"""Freeze ordered validation examples before looking at candidate predictions."""
import argparse
import json
import hashlib
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    raw = args.split.read_bytes()
    split = json.loads(raw)
    selected, identities = [], set()
    # Include one ROI (or whole if unavailable) per species and per negative label.
    for kind in ('species_roi','species_full','non_tick_hard_roi','non_tick_full'):
        for index,row in enumerate(split['val']):
            if row['sample_type'] != kind:
                continue
            key = ('species',row['species_label']) if row['species_label'] else ('negative',row['negative_label'])
            if key not in identities:
                identities.add(key); selected.append(dict(index=index,row=row))
    coco = [(i,r) for i,r in enumerate(split['val']) if r['sample_type']=='coco_tick_roi']
    # Include fixed small, large and multi-target-source examples, independent of predictions.
    coco.sort(key=lambda pair:(pair[1]['bbox'][2]-pair[1]['bbox'][0])*(pair[1]['bbox'][3]-pair[1]['bbox'][1]))
    for index,row in coco[:4]+coco[-4:]:
        selected.append(dict(index=index,row=row))
    if args.out.exists():
        raise ValueError('Frozen cases already exist; do not overwrite after evaluation')
    args.out.write_text(json.dumps(dict(manifest_sha256=hashlib.sha256(raw).hexdigest(),subset='val',
        selection='first qualifying row per species/negative class; four smallest and largest COCO boxes',
        cases=selected),ensure_ascii=False,indent=2),'utf-8')
    print(f'Frozen {len(selected)} cases')


if __name__=='__main__':
    main()
