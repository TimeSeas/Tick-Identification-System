"""Demo inference script for tick detection."""

import cv2
import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import numpy as np
import os

def run_inference(model_path, image_path, cfg_path="configs/train_mask_rcnn.yaml", output_dir="output/demo"):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = model_path

    model = build_model(cfg)
    model.eval()

    # Load and preprocess image
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to RGB for visualization
    image_rgb = image[:, :, ::-1]

    # Normalize and convert to tensor
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    image_tensor = (image_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model([{"image": image_tensor, "height": image.shape[0], "width": image.shape[1]}])

    # Get predictions
    instances = outputs[0]["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    labels = instances.pred_classes.numpy()
    masks = instances.pred_masks.numpy()

    print(f"Detected {len(boxes)} tick(s)")
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        print(f"  Tick {i+1}: box={box}, score={score:.3f}")

    # Visualize
    v = Visualizer(image_rgb, MetadataCatalog.get("tick_train"), scale=1.0)
    out = v.draw_instance_predictions(instances)

    # Save result
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
    print(f"Saved visualization to {output_path}")

    return {
        'boxes': boxes,
        'scores': scores,
        'labels': labels,
        'masks': masks
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python demo_inference.py <model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    run_inference(model_path, image_path)