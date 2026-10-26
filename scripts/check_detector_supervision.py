"""CPU preflight every train/val sample through the actual detector dataset."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/train'))
from train_detector_rebuild import RebuildDataset,collate_fn,file_sha256,validate_split


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(); torch.set_num_threads(2)
    if args.out.exists():
        raise ValueError('Preserve prior preflight reports')
    split=json.loads(args.split.read_text('utf-8')); validate_split(split); counts={}
    for subset in ('train','val'):
        loader=DataLoader(RebuildDataset(split[subset],ROOT),batch_size=1,num_workers=2,collate_fn=collate_fn)
        count=0
        for images,targets in loader:
            count+=len(images)
            if count%250==0:
                print(f'preflight {subset} {count}/{len(loader.dataset)}',flush=True)
        counts[subset]=count
    report=dict(split_sha256=file_sha256(args.split),counts=counts,
        dataset_sha256=file_sha256(ROOT/'src/train/train_detector_rebuild.py'),
        note='CPU decode, coordinate, target and raster-mask validation; no model predictions; calibration not read')
    args.out.write_text(json.dumps(report,indent=2),'utf-8'); print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
