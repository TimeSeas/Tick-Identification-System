"""
昆虫粗分类器（28 类）：检测后的局部复核与未定位时的整图兜底
契约：classify(img_rgb: np.ndarray) -> (slug, 置信度 0-100)
整图分类不否决其他位置的蜱虫；低置信结果保留为待确认候选。
"""
import json

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from app.services.classification_views import probabilities, prediction


class InsectClassifier:
    def __init__(self, model_path: str, split_json: str, device: str = None,
                 view_mode: str = 'center'):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = 224
        self.view_mode = view_mode
        with open(split_json, encoding="utf-8") as f:
            split = json.load(f)
        self.classes = sorted({d["label"] for d in split["train"]})
        self.num_classes = len(self.classes)

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        resize_size = int(self.input_size * 1.14)
        self.eval_tf = T.Compose([T.Resize(resize_size), T.CenterCrop(self.input_size),
                                  T.ToTensor(), normalize])
        self.model = self._build_model(model_path)
        print(f"[昆虫分类] 模型加载完成，设备: {self.device}，类别数: {self.num_classes}")

    def _build_model(self, model_path: str):
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        return model.to(self.device).eval()

    def classify(self, img_rgb: np.ndarray) -> tuple:
        """返回 (slug, 置信度百分数)"""
        result = self.predict(img_rgb)
        return result['slug'], round(result['confidence'], 2)

    def predict(self, img_rgb: np.ndarray) -> dict:
        scores = probabilities(self.model, img_rgb, self.input_size, self.device, self.view_mode)
        return prediction(scores, self.classes)
