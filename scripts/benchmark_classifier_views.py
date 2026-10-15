"""Compare fixed inference views on validation; freeze selection before testing.

Run from repository root in tick_detection. Results never overwrite training artifacts.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T
from torchvision.transforms import functional as F

ROOT = Path(__file__).resolve().parents[1]
VIEW_NAMES = ['top_left', 'top_right', 'bottom_left', 'bottom_right', 'center',
              'top_left_flip', 'top_right_flip', 'bottom_left_flip', 'bottom_right_flip', 'center_flip',
              'full', 'full_flip']
MODES = {'center': [4], 'flip': [4, 9], 'five_crop': list(range(5)),
         'ten_crop': list(range(10)), 'full': [10], 'full_flip': [10, 11],
         'center_full': [4, 10], 'center_full_flip': [4, 9, 10, 11]}
SPECS = {
    'species': ('data/tick_species/split_clean.json', 'output/tick_species_clean_448/model/tick_species_best.pth', 448, 4),
    'insect': ('data/insects_split.json', 'output/insect_classifier/model/insect_best.pth', 224, 12),
}

class ViewsDataset(Dataset):
    def __init__(self, items, classes, size):
        self.items, self.classes, self.size = items, classes, size
        self.normalize = T.Compose([T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        with Image.open(ROOT / item['path']) as source:
            image = source.convert('RGB')
            resized = F.resize(image, int(self.size * 1.14))
            crops = list(F.five_crop(resized, self.size))
            crops += [F.hflip(crop) for crop in crops]
            full = F.resize(image, [self.size, self.size])
            crops += [full, F.hflip(full)]
            tensors = torch.stack([self.normalize(crop) for crop in crops])
        return tensors, self.classes.index(item['label'])

def run(task, subset, out):
    split_name, weight_name, size, batch_size = SPECS[task]
    split = json.loads((ROOT / split_name).read_text('utf-8'))
    classes = sorted({x['label'] for x in split['train']})
    cache = out / f'{task}_{subset}_views.npz'
    if cache.exists():
        loaded = np.load(cache); probs, labels = loaded['probabilities'], loaded['labels']
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
        model.load_state_dict(torch.load(ROOT / weight_name, map_location='cpu', weights_only=True))
        model.to(device).eval()
        loader = DataLoader(ViewsDataset(split[subset], classes, size), batch_size=batch_size, shuffle=False, num_workers=0)
        predictions, truths = [], []
        started = time.perf_counter()
        with torch.inference_mode():
            for i, (images, truth) in enumerate(loader):
                n, views, c, h, w = images.shape
                p = model(images.flatten(0, 1).to(device)).softmax(1).reshape(n, views, -1)
                predictions.append(p.cpu().numpy()); truths.append(truth.numpy())
                if i % 20 == 0: print(task, subset, 'processed', sum(len(x) for x in truths), '/', len(split[subset]), flush=True)
        probs, labels = np.concatenate(predictions), np.concatenate(truths)
        np.savez_compressed(cache, probabilities=probs, labels=labels)
        print('seconds', time.perf_counter() - started, flush=True)
        del model
        if device.type == 'cuda': torch.cuda.empty_cache()
    report = {}
    for mode, ids in MODES.items():
        mean = probs[:, ids].mean(1); pred = mean.argmax(1)
        report[mode] = {'accuracy': float(accuracy_score(labels, pred)),
                        'macro_f1': float(f1_score(labels, pred, average='macro')),
                        'n': len(labels), 'views': len(ids)}
        with open(out / f'{task}_{subset}_{mode}.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f);writer.writerow(['path', 'true', 'pred', 'prob', 'margin'])
            for item, y, yhat, scores in zip(split[subset], labels, pred, mean):
                ordered = np.sort(scores)
                writer.writerow([item['path'], classes[y], classes[yhat], float(scores[yhat])*100,
                                 float(ordered[-1]-ordered[-2])*100])
    meta = {'task': task, 'subset': subset, 'classes': classes, 'view_names': VIEW_NAMES,
            'weights': weight_name, 'weights_sha256': hashlib.sha256((ROOT/weight_name).read_bytes()).hexdigest(),
            'split_sha256': hashlib.sha256((ROOT/split_name).read_bytes()).hexdigest(), 'results': report}
    (out / f'{task}_{subset}_comparison.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({task: report}, ensure_ascii=False, indent=2), flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subset', choices=['val','test'], default='val')
    parser.add_argument('--task', choices=['species','insect','all'], default='all')
    parser.add_argument('--out', type=Path, default=ROOT/'output/optimization_20261015')
    args=parser.parse_args();args.out.mkdir(parents=True, exist_ok=True);torch.set_num_threads(4)
    for task in SPECS if args.task=='all' else [args.task]:run(task,args.subset,args.out)

if __name__=='__main__':main()
