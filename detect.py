"""简单易用的蜱虫检测推理脚本"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import numpy as np
import cv2

MODEL_PATH = "output/mask_rcnn_ticks.pth"
CONFIDENCE = 0.5

def get_model(num_classes=2):
    model = maskrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

def load_model(model_path, device):
    print(f"加载模型: {model_path}")
    model = get_model(2)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()
    print(f"模型加载成功! 使用设备: {device}")
    return model

def detect(model, image_path, device, conf_thresh=CONFIDENCE):
    # 读取图片
    img = Image.open(image_path).convert('RGB')
    img_rgb = np.array(img)
    orig_h, orig_w = img_rgb.shape[:2]

    # 转换为tensor
    img_tensor = torch.tensor(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # 检测
    with torch.no_grad():
        outputs = model(img_tensor)

    output = outputs[0]
    boxes = output['boxes'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    masks = output['masks'].cpu().numpy()

    return boxes, scores, masks, img_rgb

def draw_results(img, boxes, scores, masks, conf_thresh=CONFIDENCE):
    result_img = img.copy()
    detections = []

    for i in range(len(boxes)):
        if scores[i] < conf_thresh:
            continue

        box = boxes[i]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # 绘制边界框
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制标签
        label = f"tick: {scores[i]:.2f}"
        cv2.putText(result_img, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 绘制mask（绿色半透明）
        mask = masks[i, 0] > 0.5
        result_img[mask] = np.clip(result_img[mask] * 0.5 + np.array([0, 255, 0]) * 0.5, 0, 255).astype(np.uint8)

        detections.append({
            'box': box,
            'score': float(scores[i])
        })

    return result_img, detections

def process_image(image_path, output_path=None, show=True, conf_thresh=CONFIDENCE):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = load_model(MODEL_PATH, device)

    print(f"\n处理图片: {image_path}")
    boxes, scores, masks, img = detect(model, image_path, device, conf_thresh)

    # 绘制结果
    result_img, detections = draw_results(img, boxes, scores, masks, conf_thresh)

    print(f"\n检测结果:")
    print(f"  检测到 {len(detections)} 个蜱虫")
    for i, det in enumerate(detections):
        box = det['box']
        print(f"  蜱虫{i+1}: 置信度={det['score']:.2f}, 位置=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f})")

    # 保存结果
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        print(f"\n结果已保存: {output_path}")

    if show:
        # 显示图片（如果支持）
        try:
            cv2.imshow("Tick Detection", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("(cv2.imshow 在此环境不可用)")

    return detections

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="蜱虫检测推理")
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument("-o", "--output", default=None, help="输出图片路径")
    parser.add_argument("-c", "--conf", type=float, default=CONFIDENCE, help=f"置信度阈值 (默认={CONFIDENCE})")
    parser.add_argument("--no-show", action="store_true", help="不显示图片")

    args = parser.parse_args()

    output_path = args.output or f"output/result_{os.path.basename(args.image)}"
    process_image(args.image, output_path, show=not args.no_show, conf_thresh=args.conf)
