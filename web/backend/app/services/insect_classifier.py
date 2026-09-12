"""
昆虫粗分类器（28 类）：Web 级联的第一级——"这是什么昆虫"
契约：classify(img_rgb: np.ndarray) -> (slug, 置信度 0-100)
判为 tick 时进入检测+蜱种分支；其他类别直接返回"这是 XX（非蜱虫）"
"""
import json

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torchvision.models import ResNet50_Weights


class InsectClassifier:
    def __init__(self, model_path: str, split_json: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = 224
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
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        return model.to(self.device).eval()

    def classify(self, img_rgb: np.ndarray) -> tuple:
        """返回 (slug, 置信度百分数)"""
        img = Image.fromarray(img_rgb)
        tensor = self.eval_tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.model(tensor).softmax(dim=1)[0]
            pred_idx = int(probs.argmax())
        return self.classes[pred_idx], round(float(probs[pred_idx]) * 100, 2)
