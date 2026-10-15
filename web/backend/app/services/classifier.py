
"""
蜱种分类推理，蜱虫种类

"""
import json
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
from app.services.classification_views import probabilities, prediction
class SpeciesClassifier:
    """蜱种分类器"""
    def __init__(self, model_path: str, split_json: str, input_size: int = 448,
                 view_mode: str = 'center'):
        self.device =('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size=input_size
        self.view_mode = view_mode
        with open(split_json,encoding='utf-8') as f:
            split=json.load(f)
        self.classes=sorted({d["label"] for d in split["train"]})
        self.num_classes=len(self.classes)
    
         # 预处理变换
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        resize_size = int(self.input_size * 1.14)
        self.eval_tf = T.Compose([
            T.Resize(resize_size),
            T.CenterCrop(self.input_size),
            T.ToTensor(),
            normalize
        ])
        self.model=self._build_model(model_path)
        print(f"[蜱虫分类] 模型加载完成，设备: {self.device}，类别数: {self.num_classes}")

    def _build_model(self, model_path:str):
           """
           # 构建多蜱虫分类模型
           """
           model = models.resnet50(weights=None)
           model.fc = nn.Linear(model.fc.in_features, self.num_classes)
           state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
           model.load_state_dict(state_dict)
           model.to(self.device).eval()
           return model     
    def classify(self,crop_rgb:np.ndarray)->tuple[str,float]:
        """
        对裁剪后的蜱虫区域进行分类,输入图像RGB，输出类别和置信值
        :param crop_rgb: shape (H, W, 3)，RGB 格式，像素值 0~255
        :return: (slug, confidence)，置信度为0至100的百分数
        """

        result = self.predict(crop_rgb)
        return result['slug'], round(result['confidence'], 2)

    def predict(self, crop_rgb: np.ndarray, whole_rgb=None, whole_weight=0.0) -> dict:
        scores = probabilities(self.model, crop_rgb, self.input_size, self.device, self.view_mode)
        if whole_rgb is not None and whole_weight > 0:
            whole = probabilities(self.model, whole_rgb, self.input_size, self.device, self.view_mode)
            scores = (1 - whole_weight) * scores + whole_weight * whole
        return prediction(scores, self.classes)
        



