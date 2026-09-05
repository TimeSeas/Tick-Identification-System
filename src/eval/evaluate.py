"""Evaluate tick detection model."""

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from src.data.register_datasets import register_tick_datasets

def evaluate(model_path, dataset_name="tick_test", cfg_path="configs/train_mask_rcnn.yaml"):
    # Register datasets
    register_tick_datasets()

    # Setup config
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TEST = (dataset_name,)

    # Create predictor
    predictor = DefaultPredictor(cfg)

    # Evaluate
    evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir="output/eval")
    val_loader = build_detection_test_loader(cfg, dataset_name)
    results = inference_on_dataset(predictor.model, val_loader, evaluator)

    # Print results
    print("\n=== Evaluation Results ===")
    if 'bbox' in results:
        print(f"mAP@50: {results['bbox'].get('AP50', 0):.3f}")
        print(f"mAP@75: {results['bbox'].get('AP75', 0):.3f}")
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "output/mask_rcnn/model_final.pth"

    results = evaluate(model_path)