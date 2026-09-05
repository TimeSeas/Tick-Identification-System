"""蜱虫检测推理 - 处理文件夹中的所有图片"""

import os
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# ============ 配置区 ============
MODEL_PATH = "output/mask_rcnn_ticks.pth"
INPUT_FOLDER = "archive3/farm_insects/Aphids"
OUTPUT_FOLDER = "output/detection_results"  # 检测结果输出文件夹
CONFIDENCE = 0.5
# ================================

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
    print(f"模型加载成功! 设备: {device}")
    return model

def detect(model, image_path, device):
    img = Image.open(image_path).convert('RGB')
    img_rgb = np.array(img)

    img_tensor = torch.tensor(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    output = outputs[0]
    return output['boxes'].cpu().numpy(), output['scores'].cpu().numpy(), output['masks'].cpu().numpy(), img_rgb

def draw_results(img, boxes, scores, masks, conf_thresh=CONFIDENCE):
    result_img = img.copy()
    count = 0

    for i in range(len(boxes)):
        if scores[i] < conf_thresh:
            continue

        box = boxes[i]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # 绿色边界框
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result_img, f"tick: {scores[i]:.2f}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 绿色mask
        mask = masks[i, 0] > 0.5
        result_img[mask] = np.clip(result_img[mask] * 0.5 + np.array([0, 255, 0]) * 0.5, 0, 255).astype(np.uint8)

        count += 1

    return result_img, count

def process_folder(model, input_folder, output_folder, device):
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有图片文件
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(extensions)]

    if not image_files:
        print(f"文件夹中没有找到图片: {input_folder}")
        return

    print(f"\n找到 {len(image_files)} 张图片")
    print(f"输出文件夹: {output_folder}")
    print(f"置信度阈值: {CONFIDENCE}")
    print("-" * 50)

    total_ticks = 0
    results = []

    for img_file in tqdm(image_files, desc="检测中"):
        img_path = os.path.join(input_folder, img_file)
        output_path = os.path.join(output_folder, f"result_{img_file}")

        try:
            boxes, scores, masks, img = detect(model, img_path, device)
            result_img, tick_count = draw_results(img, boxes, scores, masks, CONFIDENCE)

            cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

            results.append({
                'file': img_file,
                'ticks': tick_count,
                'total_score': sum(scores[scores >= CONFIDENCE])
            })
            total_ticks += tick_count

        except Exception as e:
            print(f"\n错误: {img_file} - {e}")

    # 打印结果
    print("\n" + "=" * 50)
    print("检测结果汇总")
    print("=" * 50)
    print(f"处理图片数: {len(image_files)}")
    print(f"检测到蜱虫数: {total_ticks}")
    print(f"\n结果保存在: {output_folder}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 50)
    print("蜱虫检测系统")
    print("=" * 50)

    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在 - {MODEL_PATH}")
        print("请先运行训练脚本!")
        return

    if not os.path.exists(INPUT_FOLDER):
        print(f"错误: 输入文件夹不存在 - {INPUT_FOLDER}")
        print("请修改 INPUT_FOLDER 为你的图片文件夹路径")
        return

    model = load_model(MODEL_PATH, device)
    process_folder(model, INPUT_FOLDER, OUTPUT_FOLDER, device)

if __name__ == "__main__":
    main()
