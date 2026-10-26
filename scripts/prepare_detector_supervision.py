"""Exclude unverified empty COCO images without changing original manifests."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise ValueError('Preserve existing supervision manifest')
    raw=args.split.read_bytes(); source=json.loads(raw)
    result=dict(metadata=dict(source_sha256=hashlib.sha256(raw).hexdigest(),
        policy='COCO positives plus explicit folder_proxy_non_tick only',
        limitation='folder proxy negatives still not manually verified'))
    excluded=[]
    for subset in ('train','val','calibration'):
        result[subset]=[]
        for row in source[subset]:
            if row['annotations'] or row.get('negative_label'):
                result[subset].append(row)
            else:
                excluded.append(dict(subset=subset,path=row['path'],reason='unverified_empty_coco'))
    result['excluded']=excluded
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(dict(counts={s:len(result[s]) for s in ('train','val','calibration')},
        excluded=len(excluded),sha256=hashlib.sha256(args.out.read_bytes()).hexdigest())))


if __name__=='__main__':
    main()
