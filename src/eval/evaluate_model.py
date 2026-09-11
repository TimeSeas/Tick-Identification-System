"""Evaluation script for tick detection model using COCO metrics."""

import os
import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import json
from tqdm import tqdm

# Configuration
MODEL_PATH = "output/mask_rcnn_ticks.pth"
TEST_ANN = "ticks image detection.v5i.coco/test/_annotations.coco.json"
TEST_IMG = "ticks image detection.v5i.coco/test"
CONFIDENCE_THRESHOLD = 0.05  # Low threshold to get all detections for evaluation

def get_model(num_classes=2):
    """Create Mask R-CNN model with custom head."""
    model = maskrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def load_model(model_path, device):
    """Load trained model weights."""
    print(f"Loading model from {model_path}...")
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model

def coco_box_to_xyxy(box):
    """Convert COCO box format [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = box
    return [x, y, x + w, y + h]

def evaluate_model(model, coco_gt, image_dir, device, max_images=None):
    """Run evaluation on test set using COCO metrics."""
    model.eval()

    img_ids = list(coco_gt.imgs.keys())
    if max_images:
        img_ids = img_ids[:max_images]

    results = []
    print(f"Evaluating {len(img_ids)} images...")

    for img_id in tqdm(img_ids):
        # Get image info
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            continue

        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(img_tensor)

        output = outputs[0]

        # Convert detections to COCO format
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()

        # Filter by confidence
        keep = scores >= CONFIDENCE_THRESHOLD
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # Convert to COCO format
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            results.append({
                'image_id': img_id,
                'category_id': int(label),
                'bbox': [float(x1), float(y1), float(w), float(h)],
                'score': float(score),
                'area': float(w * h),
                'iscrowd': 0
            })

    return results

def compute_metrics(coco_gt, results):
    """Compute COCO metrics."""
    # Save results to temporary file
    import tempfile
    import datetime

    temp_dir = tempfile.gettempdir()
    result_file = os.path.join(temp_dir, f"coco_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    with open(result_file, 'w') as f:
        json.dump(results, f)

    # Load results
    coco_dt = COCO.loadRes(coco_gt, result_file)

    # Run evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Clean up
    os.remove(result_file)

    return coco_eval

def main():
    """Main evaluation function."""
    print("=" * 60)
    print("Tick Detection Model Evaluation")
    print("=" * 60)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please run training first: python src/train/train_torchvision.py")
        return

    model = load_model(MODEL_PATH, device)

    # Load ground truth
    print(f"\nLoading ground truth from: {TEST_ANN}")
    coco_gt = COCO(TEST_ANN)
    print(f"Test images: {len(coco_gt.imgs)}")
    print(f"Test annotations: {len(coco_gt.anns)}")

    # Run evaluation
    print("\nRunning inference on test set...")
    results = evaluate_model(model, coco_gt, TEST_IMG, device)

    print(f"\nTotal detections: {len(results)}")

    if len(results) == 0:
        print("No detections above threshold. Try lowering CONFIDENCE_THRESHOLD")
        return

    # Compute metrics
    print("\n" + "=" * 60)
    print("COCO METRICS (BBox)")
    print("=" * 60)
    metrics = compute_metrics(coco_gt, results)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    # COCOeval.stats 顺序：0=mAP@[.50:.95], 1=mAP@.50, 2=mAP@.75（此前标反，已修正）
    print(f"mAP@[0.50:0.95]: {metrics.stats[0]:.4f}")
    print(f"mAP@0.50: {metrics.stats[1]:.4f}")
    print(f"mAP@0.75: {metrics.stats[2]:.4f}")

if __name__ == "__main__":
    main()
