
"""
蜱种分类推理，蜱虫种类

"""
import json
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
class SpeciesClassifier:
    """蜱种分类器"""
    def __init__(self, model_path: str, split_json: str):
        self.device =('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size=224
        with open(split_json,encoding='utf-8') as f:
            split=json.load(f)
        self.classes=sorted({d["label"] for d in split["train"]})
        self.num_classes=len(self.classes)
    
         # 预处理变换
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        resize_size = int(self.input_size * 1.14)  # 255
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
           model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
           model.fc = nn.Linear(model.fc.in_features, self.num_classes)
           state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
           model.load_state_dict(state_dict)
           model.to(self.device).eval()
           return model     
    def classify(self,crop_rgb:np.ndarray)->tuple[str,float]:
        """
        对裁剪后的蜱虫区域进行分类,输入图像RGB，输出类别和置信值
        :param crop_rgb: shape (H, W, 3)，RGB 格式，像素值 0~255
        :return: (slug, confidence) 例如 ("ixodes_ricinus", 0.9523)
        """

        img=Image.fromarray(crop_rgb)
        tensor=self.eval_tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs=self.model(tensor).softmax(dim=1)[0]
            pred_idx=int(probs.argmax())
            confidence=float(probs[pred_idx])
        slug = self.classes[pred_idx]
        return slug, round(confidence*100, 2)#转为百分数        
        



