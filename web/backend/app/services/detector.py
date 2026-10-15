
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np
DEDUP_IOU = 0.3   # 重叠去重阈值：大像素蜱虫重复检出框实测 IoU 0.476（0.5 拦不住）；
                  # 相邻两只真蜱虫的框 IoU 通常 <0.1，不会被误合并
class Detector:
    """蜱虫检测，模型推理，只检测是否有蜱虫，有几个
    tile>1 时启用切块推理（小目标检测路线 A）：
    大图切成 tile×tile 块分别推理再合并坐标，蜱虫在块内占比放大，小目标可检出；
    相邻块可能重复检出同一只蜱，用 IoU 去重
    """
    CONFIDENCE = 0.5
    def __init__(self,model_path:str,tile:int=1):
        """加载预训练模型"""
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tile = tile
        self.model=self._build_model(model_path)
        print(f"[蜱虫检测] 模型加载完成，设备：{self.device}，切块：{tile}x{tile}")
        
    def _build_model(self, model_path:str):
        """构建Mask R-CNN二类模型"""
        model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes=2)
        #加载权重
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.transform.min_size = (1024,)   # ← 加：恢复训练分辨率，否则默认 800（train-serve skew）
        model.transform.max_size = 1200      # ← 加
        model.to(self.device).eval()
        return model

    def detect(self,img_rgb:np.ndarray,tile:int=None)->list[dict]:
        """主入口：tile=1 全图推理（大目标绝不能切碎）；tile>1 切块推理合并（小目标补偿）
        tile 为 None 时用构造时的默认切分数
        两个分支都做重叠去重：同一只大蜱虫会被模型重复检出多个嵌套框"""
        if tile is None:
            tile = self.tile
        if tile <= 1:
            return _dedup(self._detect(img_rgb))
        h, w = img_rgb.shape[:2]
        th, tw = int(np.ceil(h / tile)), int(np.ceil(w / tile))
        merged = []
        for i in range(tile):
            for j in range(tile):
                y1, x1 = i * th, j * tw
                y2, x2 = min(y1 + th, h), min(x1 + tw, w)
                for d in self._detect(img_rgb[y1:y2, x1:x2]):
                    b = d["bbox"]
                    d["bbox"] = [b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1]  # 坐标平移回原图
                    merged.append(d)
        # 跨块边界同一只蜱会被相邻块检出两次：与全图路径用同一个去重
        return _dedup(merged)

    def _detect(self,img_rgb:np.ndarray)->list[dict]:
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


def _dedup(dets:list[dict])->list[dict]:
    """重叠框去重：按分数从高到低贪心，与已保留框 IoU>DEDUP_IOU 的视为同一只蜱的重复检出"""
    deduped = []
    for d in sorted(dets, key=lambda x: -x["score"]):
        if all(_iou(d["bbox"], k["bbox"]) <= DEDUP_IOU for k in deduped):
            deduped.append(d)
    return deduped


def _iou(box_a, box_b):
    """两框交并比（去重用）"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)



