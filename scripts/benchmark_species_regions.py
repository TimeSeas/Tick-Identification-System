"""Validate tight regions, context padding and whole-image fusion on single detections."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score,f1_score
from torchvision import models,transforms as T
from torchvision.transforms import functional as F

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/optimization_20261015'
sys.path.insert(0,str(OUT/'baseline_app/services'))
from detector import _dedup

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--subset',choices=['val','test'],default='val');args=ap.parse_args();torch.set_num_threads(4)
    split=json.loads((ROOT/'data/tick_species/split_clean.json').read_text('utf-8'));classes=sorted({r['label'] for r in split['train']})
    detections={r['path']:r for r in json.loads((OUT/f'species_{args.subset}_detections.json').read_text('utf-8'))}
    cache=OUT/f'species_{args.subset}_regions.npz'
    items=[]
    for row in split[args.subset]:
        ds=_dedup([d for d in detections[row['path']]['full'] if d['score']>=.5])
        if len(ds)==1:items.append(dict(row,bbox=ds[0]['bbox']))
    if cache.exists():data=np.load(cache);p=data['probabilities'];labels=data['labels']
    else:
        device='cuda' if torch.cuda.is_available() else 'cpu';model=models.resnet50(weights=None);model.fc=torch.nn.Linear(2048,len(classes));model.load_state_dict(torch.load(ROOT/'output/tick_species_clean_448/model/tick_species_best.pth',map_location='cpu',weights_only=True));model.to(device).eval()
        norm=T.Compose([T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])]);all_p=[];labels=[]
        with torch.inference_mode():
            for i,row in enumerate(items):
                with Image.open(ROOT/row['path']) as source:im=source.convert('RGB')
                x1,y1,x2,y2=row['bbox'];regions=[]
                for padding in [0,.15,.35]:
                    dx=(x2-x1)*padding;dy=(y2-y1)*padding
                    regions.append(im.crop((max(0,int(x1-dx)),max(0,int(y1-dy)),min(im.width,int(x2+dx)),min(im.height,int(y2+dy)))))
                regions.append(im);tensors=[]
                for region in regions:tensors.extend([norm(F.center_crop(F.resize(region,510),448)),norm(F.resize(region,[448,448]))])
                all_p.append(model(torch.stack(tensors).to(device)).softmax(1).cpu().numpy());labels.append(classes.index(row['label']))
                if i%40==0:print(args.subset,'single detection',i+1,'/',len(items),flush=True)
        p=np.stack(all_p);labels=np.array(labels);np.savez_compressed(cache,probabilities=p,labels=labels)
    candidates={'tight_center':p[:,0], 'tight_center_full':p[:,0:2].mean(1),'context15':p[:,2:4].mean(1),'context35':p[:,4:6].mean(1),'whole':p[:,6:8].mean(1)}
    candidates['blend25']=.75*candidates['context15']+.25*candidates['whole'];candidates['blend50']=.5*candidates['context15']+.5*candidates['whole']
    report={name:{'n':len(labels),'accuracy':float(accuracy_score(labels,a.argmax(1))),'macro_f1':float(f1_score(labels,a.argmax(1),average='macro'))} for name,a in candidates.items()}
    (OUT/f'species_{args.subset}_regions.json').write_text(json.dumps({'subset':args.subset,'note':'Single detected region per image; labels inherited from image, not independently annotated region labels','items':items,'classes':classes,'results':report},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
