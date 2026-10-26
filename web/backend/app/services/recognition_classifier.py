"""Optional rebuild candidate loader; not registered as the default classifier."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from app.services.recognition_model import RecognitionModel, letterbox_tensor, PREPROCESSING
from app.services.classification_views import prediction


class RecognitionClassifier:
    def __init__(self, checkpoint_path, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        checkpoint = torch.load(Path(checkpoint_path),map_location='cpu',weights_only=False)
        if checkpoint['partial']:
            raise ValueError('Partial training checkpoint cannot be deployed')
        if checkpoint['architecture'] != 'resnet50_tick2_species_v1' or checkpoint['preprocessing'] != PREPROCESSING:
            raise ValueError('Unsupported recognition architecture or preprocessing')
        self.classes = checkpoint['classes']
        self.input_size = checkpoint['input_size']
        self.metadata = checkpoint['metadata']
        self.model = RecognitionModel(len(self.classes)).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'],strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, crop_rgb, whole_rgb=None, whole_weight=0.0):
        if whole_weight:
            raise ValueError('Rebuild candidate uses the trained region view without whole-image score fusion')
        image = Image.fromarray(np.ascontiguousarray(crop_rgb)).convert('RGB')
        output = self.model(letterbox_tensor(image,self.input_size).unsqueeze(0).to(self.device))
        result = prediction(output['species_logits'].softmax(1)[0].cpu().numpy(),self.classes)
        result['tick_probability'] = float(output['tick_logits'].softmax(1)[0,1])
        return result

    def classify(self, crop_rgb):
        result = self.predict(crop_rgb)
        return result['slug'],round(result['confidence'],2)
