"""Export a frozen inference-only candidate while preserving training evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise ValueError('Inference export exists; do not overwrite a frozen model')
    source_digest=digest(args.checkpoint)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint['partial']:
        raise ValueError('Cannot export a partial pilot for deployment')
    keys=('model_state','classes','architecture','input_size','preprocessing','metadata','partial','epoch','best_score')
    exported={k:checkpoint[k] for k in keys}
    exported['training_checkpoint_sha256']=source_digest
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(exported,args.out)
    metadata=dict(training_checkpoint=str(args.checkpoint.resolve()),training_checkpoint_sha256=source_digest,
        inference_checkpoint=str(args.out.resolve()),inference_checkpoint_sha256=digest(args.out),
        epoch=checkpoint['epoch'],architecture=checkpoint['architecture'],
        preprocessing=checkpoint['preprocessing'],input_size=checkpoint['input_size'],classes=checkpoint['classes'])
    args.out.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(metadata,ensure_ascii=False))


if __name__=='__main__':
    main()
