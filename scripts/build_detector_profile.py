"""Bind an evaluated detector policy to exact weights without changing baseline."""
import argparse
import copy
import json
from pathlib import Path

from build_recognition_profile import digest

ROOT=Path(__file__).resolve().parents[1]


def build_profile(comparison,variant,weights,baseline_path):
    report_path=comparison/'report.json'
    report=json.loads(report_path.read_text('utf-8'))
    metadata=json.loads((comparison/'metadata.json').read_text('utf-8'))
    if metadata.get('proposal_weights_sha256'):
        raise ValueError('Proposal verification needs a two-model deployment profile; not a drop-in artifact')
    selected=report['variants'][variant]
    if metadata.get('score_threshold',.5)!=.5 and not metadata.get('cache_helper_sha256'):
        raise ValueError('Non-default threshold comparison needs the threshold-aware cache helper identity')
    if report['metadata']!=metadata or not selected['gates'] or not all(selected['gates'].values()):
        raise ValueError('Require complete passing frozen comparison evidence')
    if metadata['old_profile_sha256']!=digest(baseline_path) or metadata['weights_sha256']!=digest(weights):
        raise ValueError('Current baseline or detector differs from evaluated artifacts')
    evaluated_path=comparison/(variant+'_profile.json')
    if selected['profile_sha256']!=digest(evaluated_path):
        raise ValueError('Evaluated policy artifact has changed')
    baseline=json.loads(baseline_path.read_text('utf-8'))
    for name in ('species','insect'):
        if metadata[name+'_sha256']!=baseline[name]['model_sha256']:
            raise ValueError('Evaluated classifier differs from baseline: '+name)
    profile=json.loads(evaluated_path.read_text('utf-8'))
    if (profile['species']!=baseline['species'] or profile['insect']!=baseline['insect'] or
        profile.get('recognition')):
        raise ValueError('This experiment must retain the original classifiers')
    profile=copy.deepcopy(profile)
    profile['detector']=dict(model_sha256=metadata['weights_sha256'],bbox_only=True,
        input_min_size=1024,input_max_size=1200,score_threshold=metadata.get('score_threshold',.5),dedup_iou=.3,
        cascade_variant=variant,validation_report_sha256=digest(report_path),
        status='validated_development_candidate_requires_source_checks',
        limitations=['old v3 initialization saw some grouped val sources',
            'folder negative labels are proxies','retained species model is not an open-set species classifier'])
    return profile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--variant',choices=('drop_in','regional_only','detector_only'),required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise ValueError('Do not overwrite a published inference profile')
    profile=build_profile(args.comparison,args.variant,args.weights,ROOT/'configs/inference_optimization.json')
    args.out.write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(f'Created frozen candidate profile {args.out}')


if __name__=='__main__':
    main()
