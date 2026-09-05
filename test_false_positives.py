"""
Test current model false positive rate on archive3 categories.
Run this script after setting up the environment with:
    conda env create -f environment.yml
    conda activate tick_detection
"""

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
    return model

def detect(model, image_path, device):
    img = Image.open(image_path).convert('RGB')
    img_rgb = np.array(img)
    img_tensor = torch.tensor(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
    output = outputs[0]
    return output['boxes'].cpu().numpy(), output['scores'].cpu().numpy()

def test_category(model, folder_path, device, conf_thresh=CONFIDENCE):
    """Test on a single category folder and return false positive stats."""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]

    false_positives = 0
    total_images = len(image_files)

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        try:
            boxes, scores = detect(model, img_path, device)
            # Count detections above threshold
            detections = sum(1 for s in scores if s >= conf_thresh)
            false_positives += detections
        except Exception as e:
            print(f"Error: {img_file} - {e}")

    return total_images, false_positives

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(MODEL_PATH, device)

    # Categories to test from archive3
    categories = [
        "archive3/farm_insects/Aphids",
        "archive3/farm_insects/Spider Mites",
        "archive3/farm_insects/Thrips",
        "archive3/farm_insects/Cabbage Loopers",
    ]

    print("\n" + "=" * 60)
    print("Baseline False Positive Rate Test")
    print("=" * 60)
    print(f"Confidence threshold: {CONFIDENCE}")
    print("-" * 60)

    results = []
    for cat_path in categories:
        if not os.path.exists(cat_path):
            print(f"Skipping (not found): {cat_path}")
            continue

        cat_name = os.path.basename(cat_path)
        total, fp = test_category(model, cat_path, device, CONFIDENCE)
        fp_rate = (fp / total * 100) if total > 0 else 0
        results.append({
            'category': cat_name,
            'images': total,
            'false_positives': fp,
            'fp_rate': fp_rate
        })
        print(f"{cat_name:30s} | Images: {total:4d} | False Positives: {fp:4d} | FP Rate: {fp_rate:5.2f}%")

    print("-" * 60)
    avg_fp_rate = sum(r['fp_rate'] for r in results) / len(results) if results else 0
    print(f"Average False Positive Rate: {avg_fp_rate:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()