"""Same-input component comparison; val is selection, calibration is not test.

The old models may have seen these new validation sources. This comparison is
therefore a diagnostic, not an independent test of old-model generalization.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/train'))
from train_recognition_rebuild import sha256, save_json, metrics_from_predictions, validate_manifest
from app.services.recognition_model import RecognitionModel, prepare_image, letterbox_tensor, PREPROCESSING
from app.services.classifier import SpeciesClassifier
from app.services.insect_classifier import InsectClassifier


def summarize(rows, records, n):
    result = metrics_from_predictions(rows,records,n)
    result['by_sample_type'] = {kind:metrics_from_predictions(rows,
        [r for r in records if r['sample_type']==kind],n) for kind in sorted({r['sample_type'] for r in records})}
    return result


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--subset',choices=['val','calibration'],default='val')
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    split = json.loads(args.split.read_text('utf-8'))
    classes = validate_manifest(split)
    checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint['partial']:
        raise ValueError('Pilot weights cannot be used as a formal candidate')
    if classes != checkpoint['classes'] or checkpoint['metadata']['split_sha256'] != sha256(args.split):
        raise ValueError('Candidate classes or input manifest mismatch')
    if checkpoint['preprocessing'] != PREPROCESSING:
        raise ValueError('Unsupported candidate preprocessing')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RecognitionModel(len(classes)).to(device)
    model.load_state_dict(checkpoint['model_state']); model.eval()
    old_species_path = ROOT/'output/tick_species_clean_448/model/tick_species_best.pth'
    old_insect_path = ROOT/'output/insect_classifier/model/insect_best.pth'
    old_species = SpeciesClassifier(str(old_species_path),str(ROOT/'data/tick_species/split_clean.json'),448,'center_full')
    old_insect = InsectClassifier(str(old_insect_path),str(ROOT/'data/insects_split.json'),view_mode='center_full')
    if old_species.classes != classes:
        raise ValueError('Old/new species class order differs')
    args.out.mkdir(parents=True,exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError('Use a new comparison directory')
    old_records, new_records, all_species_probabilities = [], [], []
    for index,row in enumerate(split[args.subset]):
        with Image.open(ROOT/row['path']) as source:
            image = prepare_image(source,row['bbox'],normalize_exif=row['coordinate_frame']!='raw')
            whole = prepare_image(source,normalize_exif=row['coordinate_frame']!='raw')
        rgb = np.array(image)
        species_label = classes.index(row['species_label']) if row['species_label'] is not None else -1
        shared = dict(index=index,path=row['path'],sample_type=row['sample_type'],tick_label=row['tick_label'],
                      species_label=species_label,group_id=row['group_id'])
        output = model(letterbox_tensor(image,checkpoint['input_size']).unsqueeze(0).to(device))
        probabilities = output['species_logits'].softmax(1)[0].cpu().numpy()
        all_species_probabilities.append(probabilities)
        new_records.append(dict(shared,tick_probability=float(output['tick_logits'].softmax(1)[0,1]),
            species_prediction=int(probabilities.argmax()),species_confidence=float(probabilities.max())))
        insect = old_insect.predict(rgb)
        # Reproduce deployed single-image fusion for species teacher ROI rows.
        species = old_species.predict(rgb,whole_rgb=np.array(whole) if row['sample_type']=='species_roi' else None,
                                      whole_weight=.5 if row['sample_type']=='species_roi' else 0)
        old_records.append(dict(shared,tick_probability=1.0 if insect['slug']=='tick' else 0.0,
            insect_slug=insect['slug'],insect_confidence=insect['confidence'],
            species_prediction=classes.index(species['slug']),species_confidence=species['confidence']/100))
        if (index+1)%100==0:
            print(f'{args.subset}: {index+1}/{len(split[args.subset])}',flush=True)
    save_json(args.out/'old_predictions.json',old_records)
    save_json(args.out/'new_predictions.json',new_records)
    np.savez_compressed(args.out/'new_species_probabilities.npz',probabilities=np.stack(all_species_probabilities))
    report = dict(subset=args.subset,checkpoint_sha256=sha256(args.checkpoint),manifest_sha256=sha256(args.split),
        model_source_sha256=sha256(ROOT/'web/backend/app/services/recognition_model.py'),
        evaluator_sha256=sha256(Path(__file__)),old_species_sha256=sha256(old_species_path),
        old_insect_sha256=sha256(old_insect_path),old=summarize(split[args.subset],old_records,len(classes)),
        new=summarize(split[args.subset],new_records,len(classes)),
        limitations=['component comparison, not complete cascade', 'multiple views share source groups',
        'old task weights saw some new validation sources','folder negatives are proxies',
        'binary baseline uses old coarse top1, not old cascade veto rules',
        'species ROI old baseline includes deployed whole-image fusion'])
    save_json(args.out/'report.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
