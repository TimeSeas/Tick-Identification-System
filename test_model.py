"""Test if Mask R-CNN model works for inference."""

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn

print("Testing Mask R-CNN model...")

# Load model with pretrained weights
print("Loading pretrained model...")
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

print("Model loaded successfully!")

# Test with dummy input
print("Testing inference...")
dummy_input = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    output = model(dummy_input)

print(f"Output keys: {output[0].keys()}")
print(f"Number of detections: {len(output[0]['boxes'])}")

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Load our trained weights
print("\nLoading trained weights...")
trained_model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=2)
trained_model.load_state_dict(torch.load('output/mask_rcnn_ticks.pth', map_location=device))
trained_model.eval()
print("Trained weights loaded!")

print("\n✓ Mask R-CNN model is ready to use!")