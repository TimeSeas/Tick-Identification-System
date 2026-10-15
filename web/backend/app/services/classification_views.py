"""Deterministic inference views shared by both ResNet classifiers."""
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F

NORMALIZE = T.Compose([T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])

def probabilities(model, rgb, size, device, view_mode='center'):
    image = Image.fromarray(np.ascontiguousarray(rgb))
    center = F.center_crop(F.resize(image, int(size * 1.14)), size)
    views = [center]
    if view_mode == 'center_full':
        views.append(F.resize(image, [size, size]))
    elif view_mode == 'flip':
        views.append(F.hflip(center))
    elif view_mode != 'center':
        raise ValueError(f'Unsupported classification view mode: {view_mode}')
    tensor = torch.stack([NORMALIZE(view) for view in views]).to(device)
    with torch.inference_mode():
        return model(tensor).softmax(1).mean(0).cpu().numpy()

def prediction(scores, classes):
    indices = np.argsort(-scores)[:3]
    return {'slug': classes[int(indices[0])], 'confidence': float(scores[indices[0]]) * 100,
            'candidates': [{'slug': classes[int(i)], 'confidence': round(float(scores[i]) * 100, 2)} for i in indices]}
