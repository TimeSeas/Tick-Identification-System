"""Export Detectron2 model to ONNX for TensorRT deployment."""

import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.export import TracingAdapter
import os

def export_to_onnx(model_path, output_path="output/tick_detector.onnx", cfg_path="configs/train_mask_rcnn.yaml"):
    # Load model
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = "cpu"

    print("Building model...")
    model = build_model(cfg)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 640, 640)

    print("Tracing model...")
    traced_model = TracingAdapter(model, [dummy_input])

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        traced_model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["boxes", "scores", "labels", "masks"],
        dynamic_axes={
            "input": {0: "batch"},
            "boxes": {0: "num_detections"},
            "scores": {0: "num_detections"},
            "labels": {0: "num_detections"},
            "masks": {0: "num_detections"}
        },
        opset_version=11
    )
    print(f"Exported successfully to {output_path}")
    return output_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "output/mask_rcnn/model_final.pth"

    export_to_onnx(model_path)