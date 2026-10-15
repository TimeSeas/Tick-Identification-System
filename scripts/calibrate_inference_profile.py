"""Fit deployment thresholds on validation predictions grouped by predicted class.

The precision target is empirical validation performance, never a test/deployment guarantee.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def accepted_summary(probabilities, labels, thresholds):
    predicted=probabilities.argmax(1); confidence=probabilities.max(1)*100
    accepted=confidence>=np.array([thresholds[int(c)] for c in predicted])
    correct=predicted==labels; n=int(accepted.sum()); good=int((accepted&correct).sum())
    return {'samples':len(labels),'accepted':n,'correct_accepted':good,'accepted_accuracy':good/n if n else 0,
            'coverage':n/len(labels),'useful_correct_fraction':good/len(labels)}

def fit(probabilities,labels,target=.95):
    predicted=probabilities.argmax(1);confidence=probabilities.max(1)*100;correct=predicted==labels
    def threshold(mask,minimum):
        valid=[]
        for t in np.unique(confidence[mask]):
            keep=mask&(confidence>=t)
            if t>=50 and keep.sum()>=minimum and correct[keep].mean()>=target:valid.append(float(t))
        return min(valid) if valid else 100.0001
    global_threshold=threshold(np.ones(len(labels),dtype=bool),20)
    global_table={i:global_threshold for i in range(probabilities.shape[1])}
    adaptive={}
    for i in global_table:
        mask=predicted==i
        adaptive[i]=threshold(mask,5) if mask.sum()>=20 else global_threshold
    global_summary=accepted_summary(probabilities,labels,global_table);adaptive_summary=accepted_summary(probabilities,labels,adaptive)
    # Deploy per-class fitting only if it beats the global baseline in useful correct outputs
    # while satisfying the same aggregate precision target. Sparse classes use global fallback.
    if adaptive_summary['accepted_accuracy']>=target and adaptive_summary['correct_accepted']>global_summary['correct_accepted']:
        selected,method=adaptive,'predicted_class'
    else:selected,method=global_table,'global_fallback'
    return selected,{'method':method,'target_validation_precision':target,'global':global_summary,'predicted_class':adaptive_summary,'selected':accepted_summary(probabilities,labels,selected)}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=ROOT/'output/optimization_20261015/inference_optimization_candidate.json');args=ap.parse_args()
    base=ROOT/'output/optimization_20261015'
    data=np.load(base/'species_val_regions.npz');p=data['probabilities'];labels=data['labels']
    crop=p[:,2:4].mean(1);whole=p[:,6:8].mean(1);scores=.5*crop+.5*whole
    meta=json.loads((base/'species_val_regions.json').read_text('utf-8'))
    thresholds,metrics=fit(scores,labels)
    species={'view_mode':'center_full','input_size':448,'region_padding':.15,'single_image_weight':.5,
             'thresholds':{c:thresholds[i] for i,c in enumerate(meta['classes'])},'calibration':metrics,
             'calibration_population':'Validation images with exactly one baseline detection; image-level species labels inherited by the detected region'}
    region_thresholds,region_metrics=fit(crop,labels)
    species.update(region_thresholds={c:region_thresholds[i] for i,c in enumerate(meta['classes'])},
                   region_calibration=region_metrics,
                   region_calibration_population='Context-only crops from single-detection validation images; mixed-species multi-object scenes still need independent annotation')
    data=np.load(base/'insect_val_views.npz');p=data['probabilities'][:,[4,10]].mean(1);labels=data['labels']
    meta=json.loads((base/'insect_val_comparison.json').read_text('utf-8'))
    thresholds,metrics=fit(p,labels)
    insect={'view_mode':'center_full','input_size':224,'thresholds':{c:thresholds[i] for i,c in enumerate(meta['classes'])},'calibration':metrics}
    for name,section in [('species',species),('insect',insect)]:
        provenance=json.loads((base/f'{name}_val_comparison.json').read_text('utf-8'))
        section.update(model_sha256=provenance['weights_sha256'],split_sha256=provenance['split_sha256'])
    result={'schema_version':1,'baseline_commit':'25427928','selection_split':'val','species':species,'insect':insect,
            'limitations':['Validation precision is empirical, not a guarantee','Single-object region labels are inherited from image annotations','Test results must not feed back into this profile']}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'species':species['calibration'],'insect':insect['calibration']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
