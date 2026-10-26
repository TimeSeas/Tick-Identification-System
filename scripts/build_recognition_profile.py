"""Create a reversible profile only for a candidate passing frozen comparisons."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_profile(baseline,validation,calibration,model_sha,manifest_sha,baseline_sha):
    for report,subset in ((validation,'val'),(calibration,'calibration')):
        if report['metadata']['subset']!=subset or not all(report['gates'].values()):
            raise ValueError('Candidate must pass val and frozen calibration gates')
        if report['metadata']['checkpoint_sha256']!=model_sha:
            raise ValueError('Comparison checkpoint mismatch')
    for key in ('new_detection_threshold','tick_threshold','detector_sha256','old_species_sha256','old_insect_sha256',
                'split_sha256','old_profile_sha256','coarse_guard'):
        left=validation['metadata'].get(key,False) if key=='coarse_guard' else validation['metadata'][key]
        right=calibration['metadata'].get(key,False) if key=='coarse_guard' else calibration['metadata'][key]
        if left!=right:
            raise ValueError(f'Frozen policy changed: {key}')
    if validation['metadata']['old_profile_sha256']!=baseline_sha:
        raise ValueError('Current baseline profile was not used in these comparisons')
    for name in ('species','insect'):
        if baseline[name]['model_sha256']!=validation['metadata']['old_'+name+'_sha256']:
            raise ValueError(f'Baseline {name} model differs from evaluated model')
    profile=copy.deepcopy(baseline)
    profile['recognition']=dict(model_sha256=model_sha,input_size=448,
        architecture='resnet50_tick2_species_v1',preprocessing='exif_then_xyxy_context015_letterbox_imagenet_v1',
        training_manifest_sha256=manifest_sha,
        tick_threshold=validation['metadata']['tick_threshold'],
        detector_threshold=validation['metadata']['new_detection_threshold'],
        detector_sha256=validation['metadata']['detector_sha256'],species_mode='legacy',
        selection_split='grouped_val',confirmation_split='grouped_calibration',
        probability_note='threshold is a decision rule, not a calibrated correctness guarantee',
        status='research_candidate_requires_source_validation',
        limitations=['folder negatives are proxies','old task models saw some grouped holdout sources',
                     'new species head rejected; original species model retained'])
    if validation['metadata'].get('coarse_guard'):
        profile['recognition']['coarse_guard']={'confidence':90,'strong_presence':.95}
    return profile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise ValueError('Do not overwrite a published inference profile')
    baseline_path=ROOT/'configs/inference_optimization.json'
    baseline=json.loads(baseline_path.read_text('utf-8'))
    validation=json.loads(args.validation.read_text('utf-8'))
    calibration=json.loads(args.calibration.read_text('utf-8'))
    profile=make_profile(baseline,validation,calibration,digest(args.model),digest(args.manifest),digest(baseline_path))
    profile['recognition'].update(validation_report_sha256=digest(args.validation),
        confirmation_report_sha256=digest(args.calibration))
    args.out.write_text(json.dumps(profile,ensure_ascii=False,indent=2),'utf-8')
    print(f'Created {args.out}')


if __name__=='__main__':
    main()
