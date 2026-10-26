"""Exclude unverified empty COCO images without changing original manifests."""
import argparse
import hashlib
import json
from pathlib import Path

from pycocotools import mask as mask_utils


def invalid_masks(row):
    invalid=[]
    for ann in row['annotations']:
        try:
            segmentation=ann.get('segmentation')
            if not segmentation:
                raise ValueError('missing segmentation')
            if isinstance(segmentation,list):
                rle=mask_utils.merge(mask_utils.frPyObjects(segmentation,row['height'],row['width']))
            elif isinstance(segmentation['counts'],list):
                rle=mask_utils.frPyObjects(segmentation,row['height'],row['width'])
            else:
                rle=dict(segmentation)
                if isinstance(rle['counts'],str):
                    rle['counts']=rle['counts'].encode('ascii')
            if list(rle['size'])!=[row['height'],row['width']] or int(mask_utils.area(rle))<=0:
                raise ValueError('empty raster mask or wrong dimensions')
        except (ValueError,TypeError,KeyError,IndexError) as error:
            invalid.append(dict(annotation_id=ann.get('id'),reason=str(error)))
    return invalid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise ValueError('Preserve existing supervision manifest')
    raw=args.split.read_bytes(); source=json.loads(raw)
    result=dict(metadata=dict(source_sha256=hashlib.sha256(raw).hexdigest(),
        policy='COCO positives with nonempty raster masks plus explicit folder_proxy_non_tick only',
        limitation='folder proxy negatives still not manually verified'))
    excluded=[]
    for subset in ('train','val','calibration'):
        result[subset]=[]
        for row in source[subset]:
            errors=invalid_masks(row)
            if errors:
                # Exclude the entire training image; never silently label a
                # known target as background or invent a rectangle mask.
                excluded.append(dict(subset=subset,path=row['path'],reason='invalid_raster_mask',details=errors))
            elif row['annotations'] or row.get('negative_label'):
                result[subset].append(row)
            else:
                excluded.append(dict(subset=subset,path=row['path'],reason='unverified_empty_coco'))
    result['excluded']=excluded
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(dict(counts={s:len(result[s]) for s in ('train','val','calibration')},
        excluded=len(excluded),sha256=hashlib.sha256(args.out.read_bytes()).hexdigest())))


if __name__=='__main__':
    main()
