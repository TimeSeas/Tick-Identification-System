"""Freeze one complete local detector checkpoint as service weights plus provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    manifest=args.out.with_suffix('.json')
    if args.out.exists() or manifest.exists():
        raise ValueError('Preserve frozen artifacts; use a new output filename')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=checkpoint['config']
    if checkpoint['partial'] or config['class_names']!=['background','tick'] or (config['min_size'],config['max_size'])!=(1024,1200):
        raise ValueError('Require a complete compatible two-class detector checkpoint')
    if checkpoint['model']['roi_heads.box_predictor.cls_score.weight'].shape[0]!=2:
        raise ValueError('Detector head class count differs from metadata')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(checkpoint['model'],args.out)
    evidence=dict(weights_sha256=digest(args.out),source_checkpoint=str(args.checkpoint.resolve()),
        source_checkpoint_sha256=digest(args.checkpoint),epoch=checkpoint['epoch']+1,
        config=config,validation=checkpoint['metrics'],status='frozen_development_candidate_not_automatically_promoted')
    manifest.write_text(json.dumps(evidence,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(dict(weights_sha256=evidence['weights_sha256'],epoch=evidence['epoch'])))


if __name__=='__main__':
    main()
