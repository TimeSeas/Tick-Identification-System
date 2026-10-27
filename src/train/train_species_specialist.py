"""Single-head species adaptation with frozen teacher and unknown-tick exposure."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.models import resnet50

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'web/backend'))
from prepare_species_specialist import validate_specialist_manifest,digest,save
from app.services.recognition_model import prepare_image,MEAN,STD
from training_safety import optimizer_step,validate_resume_directory

WARM_SHA='e5301d269a45e0925d21f4573a503b8cc112834374e077db360595163859ab56'


def validate_frozen_images(manifest,root=ROOT):
    identities={}
    for subset in ('train','val'):
        for row in manifest[subset]:
            path=row['path'];expected=row['sha256']
            if path in identities and identities[path]!=expected:raise ValueError('Conflicting frozen image '+path)
            identities[path]=expected
    for path,expected in identities.items():
        if digest(Path(root)/path)!=expected:raise ValueError('Frozen image content changed '+path)


def validate_resume_epoch(out,epoch):
    journal=Path(out)/'epochs.jsonl'
    epochs=[json.loads(line)['epoch'] for line in journal.read_text('utf-8').splitlines() if line]
    if not epochs or epochs!=list(range(1,epochs[-1]+1)) or epoch+1!=epochs[-1]:
        raise ValueError('Resume requires the latest complete journalled epoch')
    for path in Path(out).iterdir():
        match=re.fullmatch(r'(?:checkpoint|val)_epoch_(\d+)\.(?:pth|json)',path.name)
        if match and int(match.group(1))>epoch+1:
            raise ValueError('Later artifacts exist; preserve this run and start a separate experiment')


def specialist_loss(logits,teacher_logits,labels,oe_weight=.2,distill_weight=.5):
    known=labels>=0;unknown=~known;loss=logits.sum()*0
    if known.any():
        loss=loss+F.cross_entropy(logits[known],labels[known],label_smoothing=.05)
        temperature=2.
        loss=loss+distill_weight*temperature**2*F.kl_div(
            F.log_softmax(logits[known]/temperature,dim=1),
            F.softmax(teacher_logits[known].detach()/temperature,dim=1),reduction='batchmean')
    if unknown.any():
        loss=loss-oe_weight*F.log_softmax(logits[unknown],dim=1).mean()
    return loss


class SpeciesViews(Dataset):
    def __init__(self,rows,classes,size=448,train=False):
        self.rows=rows;self.index={c:i for i,c in enumerate(classes)};self.size=size;self.train=train
        self.augment=T.Compose([T.RandomHorizontalFlip(),T.RandomVerticalFlip(),
            T.RandomRotation(10,fill=tuple(round(v*255) for v in MEAN)),
            T.ColorJitter(brightness=.1,contrast=.1,saturation=.1)])

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(ROOT/row['path']) as source:
            image=prepare_image(source,row['bbox'],context=.15,
                normalize_exif=row.get('coordinate_frame')!='raw')
        center=TF.center_crop(TF.resize(image,int(self.size*1.14)),self.size)
        full=TF.resize(image,[self.size,self.size])
        views=[self.augment(random.choice((center,full)))] if self.train else [center,full]
        tensor=torch.stack([TF.normalize(TF.to_tensor(view),MEAN,STD) for view in views])
        return tensor,self.index.get(row['species_label'],-1),index


def sampling_weights(rows):
    known_counts=Counter(r['species_label'] for r in rows if r['species_label'] is not None)
    unknown_counts=Counter(r['source_species'] for r in rows if r['species_label'] is None)
    raw=[1/math.sqrt((known_counts if r['species_label'] else unknown_counts)[r['species_label'] or r['source_species']])
         for r in rows]
    totals={known:sum(w for r,w in zip(rows,raw) if (r['species_label'] is not None)==known) for known in (True,False)}
    if not all(totals.values()):raise ValueError('Need known and unknown training rows')
    return [w*(.85 if r['species_label'] else .15)/totals[r['species_label'] is not None] for r,w in zip(rows,raw)]


def compute_metrics(records,classes):
    matrix=np.zeros((len(classes),len(classes)),dtype=np.int64);unknown=[]
    for row in records:
        probabilities=np.asarray(row['probabilities'])
        if row['label']<0:unknown.append(float(probabilities.max()))
        else:matrix[row['label'],int(probabilities.argmax())]+=1
    denom=matrix.sum(0)+matrix.sum(1)
    f1=np.divide(2*matrix.diagonal(),denom,out=np.zeros(len(classes)),where=denom>0)
    support=matrix.sum(1)>0
    return dict(known_images=int(matrix.sum()),known_top1=float(matrix.trace()/matrix.sum()) if matrix.sum() else None,
        known_macro_f1=float(f1[support].mean()) if support.any() else None,
        unknown_images=len(unknown),unknown_maxprob_mean=float(np.mean(unknown)) if unknown else None,
        unknown_over_05=int(sum(p>=.5 for p in unknown)),
        by_class={c:dict(images=int(matrix[i].sum()),correct=int(matrix[i,i]),f1=float(f1[i])) for i,c in enumerate(classes)})


@torch.inference_mode()
def evaluate(model,loader,device,classes):
    model.eval();records=[]
    for images,labels,indices in loader:
        batch,views,channels,h,w=images.shape
        probs=model(images.reshape(-1,channels,h,w).to(device)).softmax(-1).reshape(batch,views,-1).mean(1).cpu()
        records.extend(dict(index=index,label=label,probabilities=p.tolist())
            for index,label,p in zip(indices.tolist(),labels.tolist(),probs))
    return compute_metrics(records,classes),records


def train_epoch(model,teacher,loader,optimizer,scaler,device,accumulate,max_steps,oe_weight):
    model.train();teacher.eval()
    for module in model.modules():
        if isinstance(module,nn.BatchNorm2d):module.eval()
    total=count=skipped=consecutive=0;limit=min(len(loader),max_steps) if max_steps else len(loader)
    optimizer.zero_grad(set_to_none=True)
    for step,(images,labels,_) in enumerate(loader):
        if step>=limit:break
        images=images[:,0].to(device);labels=labels.to(device)
        with torch.autocast(device_type=device.type,enabled=scaler.is_enabled()):
            with torch.no_grad():teacher_logits=teacher(images)
            loss=specialist_loss(model(images),teacher_logits,labels,oe_weight=oe_weight)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss at '+str(step))
        window=min(accumulate,limit-(step//accumulate)*accumulate)
        scaler.scale(loss/window).backward();total+=loss.item();count+=1
        if count%accumulate==0 or count==limit:
            finite=optimizer_step(model.parameters(),optimizer,scaler)
            skipped+=not finite;consecutive=0 if finite else consecutive+1
            if consecutive>=5:raise FloatingPointError('Five consecutive gradient overflows')
        if count%50==0:print(f'batch={count}/{limit} loss={total/count:.5f} skipped={skipped}',flush=True)
    return total/count,count,skipped


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--warm-start',type=Path,default=ROOT/'output/tick_species_clean_448/model/tick_species_best.pth')
    p.add_argument('--epochs',type=int,default=6);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--accumulate',type=int,default=2);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--lr',type=float,default=2e-5);p.add_argument('--oe-weight',type=float,default=.2)
    p.add_argument('--size',type=int,default=448);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--patience',type=int,default=3);p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--resume',type=Path)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    args=p.parse_args()
    if args.size!=448:p.error('This specialist experiment requires the deployed 448 input size')
    if min(args.epochs,args.batch_size,args.accumulate,args.size,args.patience)<1 or min(args.workers,args.max_steps)<0 or args.lr<=0 or args.oe_weight<0:
        p.error('Invalid training arguments')
    torch.set_num_threads(4);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    device=torch.device(args.device);manifest=json.loads(args.manifest.read_text('utf-8'))
    classes=validate_specialist_manifest(manifest)
    validate_frozen_images(manifest)
    original=json.loads((ROOT/'data/tick_species/split_clean.json').read_text('utf-8'))
    if classes!=sorted({r['label'] for r in original['train']}):raise ValueError('Original class order mismatch')
    if digest(args.warm_start)!=WARM_SHA:raise ValueError('Unexpected warm-start artifact')
    config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(manifest_sha256=digest(args.manifest),warm_start_sha256=WARM_SHA,classes=classes,
        trainer_sha256=digest(__file__),data_builder_sha256=digest(ROOT/'scripts/prepare_species_specialist.py'),
        crop_source_sha256=digest(ROOT/'web/backend/app/services/recognition_model.py'),
        safety_sha256=digest(Path(__file__).with_name('training_safety.py')),
        preprocessing='exif_context015_center_or_full448',inference_views='center_full',
        limitation='task warm-start saw some known val sources; development comparison')
    args.out.mkdir(parents=True,exist_ok=True)
    if not args.resume and any(args.out.iterdir()):raise ValueError('Preserve prior run; require empty out')
    state=torch.load(args.warm_start,map_location='cpu',weights_only=True)
    model=resnet50(weights=None);model.fc=nn.Linear(model.fc.in_features,len(classes));model.load_state_dict(state,strict=True)
    teacher=resnet50(weights=None);teacher.fc=nn.Linear(teacher.fc.in_features,len(classes));teacher.load_state_dict(state,strict=True)
    for name,parameter in model.named_parameters():parameter.requires_grad=name.startswith(('layer3.','layer4.','fc.'))
    teacher.requires_grad_(False);model.to(device);teacher.to(device).eval()
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=args.lr,weight_decay=.0001)
    scaler=torch.amp.GradScaler('cuda',enabled=args.amp and device.type=='cuda',init_scale=1024)
    generator=torch.Generator().manual_seed(args.seed)
    train_rows=manifest['train'];sampler=WeightedRandomSampler(sampling_weights(train_rows),len(train_rows),generator=generator)
    train=DataLoader(SpeciesViews(train_rows,classes,args.size,True),batch_size=args.batch_size,
        sampler=sampler,num_workers=args.workers)
    val=DataLoader(SpeciesViews(manifest['val'],classes,args.size),batch_size=args.batch_size,
        shuffle=False,num_workers=args.workers)
    start=0;best=-math.inf;stale=0;baseline=None
    if args.resume:
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=False)
        validate_resume_directory(checkpoint['config'],args.out,args.resume)
        validate_resume_epoch(args.out,checkpoint['epoch'])
        for k,v in config.items():
            if k!='resume' and checkpoint['config'][k]!=v:raise ValueError('Resume mismatch '+k)
        if checkpoint['partial']:raise ValueError('Cannot resume partial pilot')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);scaler.load_state_dict(checkpoint['scaler'])
        start=checkpoint['epoch']+1;best=checkpoint['best_score'];stale=checkpoint['stale'];baseline=checkpoint['baseline']
        generator.set_state(checkpoint['loader_rng']);torch.set_rng_state(checkpoint['torch_rng'])
        random.setstate(checkpoint['python_rng']);np.random.set_state(checkpoint['numpy_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    save(args.out/'config.json',config)
    try:
        if baseline is None:
            baseline,records=evaluate(teacher,val,device,classes)
            save(args.out/'baseline_val.json',dict(metrics=baseline,predictions=records))
            print('baseline '+json.dumps({k:v for k,v in baseline.items() if k!='by_class'}),flush=True)
        for epoch in range(start,args.epochs):
            loss,batches,skipped=train_epoch(model,teacher,train,optimizer,scaler,device,args.accumulate,args.max_steps,args.oe_weight)
            metrics,records=evaluate(model,val,device,classes)
            eligible=(metrics['known_top1']>=baseline['known_top1']-.01 and metrics['known_macro_f1']>=baseline['known_macro_f1']-.015)
            score=(metrics['known_top1']+metrics['known_macro_f1'])/2-.05*metrics['unknown_maxprob_mean']
            improved=eligible and score>best
            if improved:best=score;stale=0
            else:stale+=1
            metrics.update(loss=loss,batches=batches,amp_skipped=skipped,eligible=eligible,selection_score=score)
            save(args.out/f'val_epoch_{epoch+1:03}.json',dict(metrics=metrics,predictions=records))
            checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                config=config,epoch=epoch,best_score=best,stale=stale,baseline=baseline,metrics=metrics,
                partial=bool(args.max_steps),loader_rng=generator.get_state(),torch_rng=torch.get_rng_state(),
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
            torch.save(checkpoint,args.out/'last_checkpoint.pth')
            torch.save(checkpoint,args.out/f'checkpoint_epoch_{epoch+1:03}.pth')
            if improved:torch.save(checkpoint,args.out/'best_checkpoint.pth')
            with (args.out/'epochs.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(epoch=epoch+1,**metrics))+'\n')
            print('epoch '+json.dumps(dict(epoch=epoch+1,**{k:v for k,v in metrics.items() if k!='by_class'})),flush=True)
            if args.max_steps or stale>=args.patience:break
    except BaseException as error:
        save(args.out/'failure.json',dict(type=type(error).__name__,error=str(error),config=config));raise


if __name__=='__main__':main()
