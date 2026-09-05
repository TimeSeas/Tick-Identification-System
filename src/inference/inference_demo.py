"""Inference demo for tick detection using trained Mask R-CNN model."""

import os
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import numpy as np
import cv2
from datetime import datetime

# Configuration
MODEL_PATH = "output/mask_rcnn_ticks.pth"
#加载模型
TEST_IMAGE_DIR = "ticks image detection.v5i.coco/test"
#加载测试数据集
OUTPUT_DIR = "output/inference_results"
#输出路径，推理结果
CONFIDENCE_THRESHOLD = 0.5
NUM_CLASSES=2
def get_model(num_classes=NUM_CLASSES):
#预训练模型的输出层是 硬编码为 81 类（80 类 COCO + 背景），直接使用会导致：
# 预测维度与标注数据（标签编号范围）不匹配，报错。
# 无法正确训练或推理，因为类别数不对。
# 替换预测头后，模型的其他部分（骨干网络、RPN、ROI 池化等）完全保留，既能继承 COCO 学到的通用特征（若加载预训练权重），又能输出你需要的类别数。
   
    #构建与训练时完全相同的Mask R-CNN网络结构
    #需要与训练脚本train/train_torchvision.py中使用的num_classes一致
    """Create Mask R-CNN model with custom head."""
    model = maskrcnn_resnet50_fpn(pretrained=False)#pretrained=false表示不自动加载预训练权重
#创建完整的Mask R-CNN网络，骨干网络是ResNet-50+FPN

    # Replace box predictor
    #替换边界框预测器
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    #model.roi_heads.box_predictor是负责 目标分类 + 边界框回归 的网络头
# 在原始模型中，它包含两个线性层（cls_score 和 bbox_pred），其中：
# cls_score 的输入特征维度（in_features）来自 ROI Pooling 后的特征向量（通常为 1024）。
# 输出维度是 80+1 = 81（COCO 的 80 个类别 + 背景）
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#重新生成预测头，输出维度是num_classes
    
    # Replace mask predictor
    #替换掩码预测器
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    #model.roi_heads.mask_predictor负责生成实例分割掩码
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def load_model(model_path, device):
    """Load trained model weights."""
    #加载模型
    print(f"Loading model from {model_path}...")
    model = get_model(num_classes=NUM_CLASSES)
    #调用get_model函数
    model.load_state_dict(torch.load(model_path, map_location=device))
    #使用torch。load加载权重，map_location保证CPU/GPU 兼容
    # （底层原理：保存的模型.pth 文件，PyTorch 保存的不仅是数值，还包括张量的设备元数据,
    # 如果直接 torch.load(model_path) 而不加 map_location，PyTorch 会严格遵守元数据，试图将张量加载到保存时所在的设备上。）

    #（device在主函数定义了，加载设备在cpu还是GPU）
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model

def detect_ticks(model, image_path, device):
    """Run inference on a single image."""
# 硬盘图片 (jpg/png)
#     ↓ PIL.Image.open
# PIL对象 [H, W, C] (0~255)
#     ↓ np.array + torch.tensor
# 张量 [H, W, C] (uint8)
#     ↓ .permute(2,0,1)
# 张量 [C, H, W] (uint8)
#     ↓ .float() / 255.0
# 张量 [C, H, W] (float32, 范围 0~1)
#     ↓ .unsqueeze(0)
# 张量 [1, C, H, W] (模型标准输入)
#     ↓ model(img_tensor)
# Python 列表: [ 字典 ]  (长度为 1)
#     ↓ outputs[0]
# 字典: {'boxes': [N,4], 'labels': [N], 'scores': [N], 'masks': [N,1,H,W]}

    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    #读取图片，convert('RGB') 强制转为 3 通道（防止灰度图导致通道数不匹配）。数据以 PIL 格式存储
    
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0#归一化到[0,1]
    #np.array(img)将 PIL 对象转为 NumPy 数组。此时数值范围是 0 ~ 255，形状为 高(H) × 宽(W) × 通道(C)
    #torch.tensor(...)将 NumPy 数组复制到 PyTorch 张量中
    #.permute(2, 0, 1) PyTorch 的卷积神经网络（CNN）要求输入形状为 (Batch, Channels, Height, Width)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    #img_tensor.unsqueeze(0)从索引0处增加 Batch 维度
    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
        #调用模型
    #输出是一个列表（batch_size=1），取出第一个元素（字典）
    return outputs[0], img

def draw_predictions(image, boxes, labels, masks, scores, threshold=0.5):
    #将模型输出的原始预测（边界框、类别、掩码、置信度）绘制到原始图像上，生成带标注的结果图
# image：PIL Image 对象，原始 RGB 图像。
# boxes：NumPy 数组，形状 (N, 4)，每行 [x1, y1, x2, y2]（浮点数）。
# labels：NumPy 数组，形状 (N,)，整数类别 ID（目前仅支持 1 表示 tick）。
# masks：NumPy 数组，形状 (N, H, W) 或 (N, C, H, W)（此处通常为前者）。
# scores：NumPy 数组，形状 (N,)，置信度分数（0~1）。
# threshold：分数阈值，低于该值的检测将被过滤。

    """Draw bounding boxes, labels and masks on image."""
    image = np.array(image).copy()

    # Create color map for different labels
    color = (0, 255, 0)  # Green for ticks

    result_masks = []
    for i in range(len(boxes)):
        score = float(scores[i])
        if score < threshold:
            continue

        box = boxes[i]
        label = int(labels[i])
        mask = masks[i]

        # Handle mask - it could be 2D or have extra dimensions
        # if len(mask.shape) == 2:
        #     mask_bool = mask > 0.5
        # else:
        #     mask_bool = mask[0] > 0.5

        # # # Draw mask with transparency
        # # if mask_bool.any():
        # #     image[mask_bool] = color

        # Draw bounding box
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label_text = f"tick: {score:.2f}"
        cv2.putText(image, label_text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        result_masks.append({
            'box': box,
            'label': label,
            'score': score
        })

    return image, result_masks

def process_directory(model, image_dir, output_dir, device, max_images=10):
    """Process all images in a directory."""
#     model：加载好权重的 Mask R‑CNN 模型。

# image_dir：待处理图像所在的文件夹。

# output_dir：结果图像保存目录（自动创建）。

# device：推理设备（'cpu' 或 'cuda'）。

# max_images：最多处理的图像数量（默认 10，避免演示时处理过多）。
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Found {len(image_files)} images to process")

    results = []
    for i, img_file in enumerate(image_files[:max_images]):
        img_path = os.path.join(image_dir, img_file)
        print(f"\n[{i+1}/{min(len(image_files), max_images)}] Processing: {img_file}")

        try:
            outputs, original_img = detect_ticks(model, img_path, device)
#detect_ticks 返回模型输出字典和原始 PIL 图像。然后将各个张量移到 CPU 并转为 NumPy（便于绘图）
            boxes = outputs['boxes'].cpu().numpy()
            labels = outputs['labels'].cpu().numpy()
            masks = outputs['masks'].cpu().numpy()
            scores = outputs['scores'].cpu().numpy()

            print(f"  Detections: {len(boxes)}")

            # Draw and save result画图
            result_img, detection_results = draw_predictions(
                original_img, boxes, labels, masks, scores, CONFIDENCE_THRESHOLD
            )

            # Save result image
            output_path = os.path.join(output_dir, f"result_{img_file}")
            cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {output_path}")

            results.append({
                'file': img_file,
                'detections': len(boxes),
                'detection_results': detection_results
            })

        except Exception as e:
            print(f"  Error processing {img_file}: {e}")

    return results

def main():
    """Main inference function."""
    print("=" * 60)
    print("Tick Detection Inference Demo")
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

    # Check test images
    if not os.path.exists(TEST_IMAGE_DIR):
        print(f"ERROR: Test image directory not found: {TEST_IMAGE_DIR}")
        return

    # Process test images
    print(f"\nProcessing images from: {TEST_IMAGE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

    results = process_directory(model, TEST_IMAGE_DIR, OUTPUT_DIR, device, max_images=10)

    # Summary
    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {len(results)}")
    total_detections = sum(r['detections'] for r in results)
    print(f"Total tick detections: {total_detections}")
    print(f"\nResults saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
