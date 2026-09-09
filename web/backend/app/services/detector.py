
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np
class Detector:
    """蜱虫检测，模型推理，只检测是否有蜱虫，有几个"""
    CONFIDENCE = 0.5
    def __init__(self,model_path:str):
        """加载预训练模型"""
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=self._build_model(model_path)
        print(f"[蜱虫检测] 模型加载完成，设备：{self.device}")
        
    def _build_model(self, model_path:str):
        """构建Mask R-CNN二类模型"""
        model = maskrcnn_resnet50_fpn(pretrained=False)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes=2)
        #加载权重
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.to(self.device).eval()
        return model

    def detect(self,img_rgb:np.ndarray)->list[dict]:
        #读取图片RGB转成张量
        #预处理：转 tensor -> 归一化 -> 加 batch 维 -> 送设备
        img_tensor = torch.tensor(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

            # 检测
        with torch.no_grad():
            outputs = self.model(img_tensor)

        output = outputs[0]
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        results=[]

        for i in range(len(scores)):
            if scores[i]<self.CONFIDENCE:
                continue
            box = boxes[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            results.append({
                "bbox":[x1,y1,x2,y2],
                "score":round(float(scores[i]),4)
            })
        results.sort(key=lambda d:d["score"],reverse=True)
        return results



