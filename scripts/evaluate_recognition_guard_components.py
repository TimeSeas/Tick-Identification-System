"""Apply the prespecified regional guard to cached component probabilities."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/train'))
from train_recognition_rebuild import metrics_from_predictions,sha256,save_json
from app.services.recognition_pipeline import coarse_guard_rejects


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    report=json.loads((args.comparison/'report.json').read_text('utf-8'))
    old=json.loads((args.comparison/'old_predictions.json').read_text('utf-8'))
    new=json.loads((args.comparison/'new_predictions.json').read_text('utf-8'))
    calibration=json.loads(args.calibration.read_text('utf-8'))
    if calibration['checkpoint_sha256']!=report['checkpoint_sha256']:
        raise ValueError('Model calibration mismatch')
    profile=json.loads((ROOT/'configs/inference_optimization.json').read_text('utf-8'))
    threshold=calibration['binary']['threshold']
    processed=[]
    for baseline,candidate in zip(old,new):
        if baseline['index']!=candidate['index']:
            raise ValueError('Cached view ordering mismatch')
        rejected=coarse_guard_rejects(dict(slug=baseline['insect_slug'],confidence=baseline['insect_confidence']),
            candidate['tick_probability'],dict(confidence=90,strong_presence=.95),profile['insect']['thresholds'])
        row=copy.deepcopy(candidate)
        row['tick_probability']=float(candidate['tick_probability']>=threshold and not rejected)
        processed.append(row)
    metrics=metrics_from_predictions([],processed,28)
    metrics['by_sample_type']={kind:metrics_from_predictions([],[r for r in processed if r['sample_type']==kind],28)
        for kind in sorted({r['sample_type'] for r in processed})}
    save_json(args.out,dict(subset=report['subset'],source_report_sha256=sha256(args.comparison/'report.json'),
        calibration_sha256=sha256(args.calibration),guard=dict(confidence=90,strong_presence=.95),
        tick_threshold=threshold,metrics=metrics,
        note='Cached same input views; confidence checks per model, correlated views and source proxy limits remain'))
    print(json.dumps(dict(subset=report['subset'],by_sample_type={k:dict(recall=v['tick_recall'],fpr=v['non_tick_fpr'])
        for k,v in metrics['by_sample_type'].items()}),indent=2))


if __name__=='__main__':
    main()
