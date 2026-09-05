"""
Train Mask R-CNN for tick detection using torchvision.
This script provides a complete training pipeline with COCO dataset support.

Supports multi-class detection when --multi-class is specified.
For multi-class mode, use --num-classes 4 (background + tick + mite + other)
"""

import os
import torch
import torch.utils.data
import argparse
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.datasets.coco import CocoDetection
from torchvision import transforms
from PIL import Image
import numpy as np
from pycocotools.coco import COCO#使用pycocotools来解析coco标注json

from pycocotools import mask as coco_mask#使用 pycocotools 内置方法将多边形转为精密掩码



# Default Configuration
DATA_ROOT = "ticks image detection.v5i.coco"
TRAIN_ANN = f"{DATA_ROOT}/train/_annotations.coco.json"
TRAIN_IMG = f"{DATA_ROOT}/train"
VAL_ANN = f"{DATA_ROOT}/valid/_annotations.coco.json"
VAL_IMG = f"{DATA_ROOT}/valid"
OUTPUT_DIR = "output"
MODEL_PATH = f"{OUTPUT_DIR}/mask_rcnn_ticks.pth"

# Default Hyperparameters
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_WORKERS = 2

# Multi-class settings
#默认数据集路径、超参数（批量大小、学习率等）以全局变量定义。

NUM_CLASSES = 2  # default: background + tick，默认传入的图片只有蜱虫+背景图，后续需要修改！！！！
CLASS_NAMES = ['__background__', 'tick']  # COCO format requires background class first

def parse_args():# 提供灵活的命令行接口，允许用户覆盖默认路径、超参数、切换多类模式。
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Mask R-CNN for insect detection")
    parser.add_argument("--data-root", default=DATA_ROOT, help="Root directory for dataset")
    parser.add_argument("--train-ann", help="Training annotation JSON path")
    parser.add_argument("--train-img", help="Training images folder path")
    parser.add_argument("--val-ann", help="Validation annotation JSON path")
    parser.add_argument("--val-img", help="Validation images folder path")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes (including background)")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Output directory for model checkpoints")
    parser.add_argument("--model-path", default=MODEL_PATH,
                        help="Path to save final model")
    parser.add_argument("--multi-class", action="store_true",
                        help="Enable multi-class mode with classes: tick, mite, ant, bee, beetle")
    return parser.parse_args()
#action="store_true" 表示若出现该选项则 args.multi_class 为 True。

def setup_multi_class_config(args):
    """Configure for multi-class detection when annotation data is available."""
    # This is called when user provides custom annotation paths
    # The class mapping should match the COCO format categories
    pass

def collate_fn(batch):
    """Custom collate function for detection datasets."""
    return tuple(zip(*batch))

class TickDetectionDataset(torch.utils.data.Dataset):#自定义数据集类
    """Custom dataset for tick detection using COCO format."""

    def __init__(self, ann_file, img_folder, transforms=None):
        self.coco = COCO(ann_file)#解析 COCO 标注文件
#COCO(ann_file) —— 解析标注 JSON
# ann_file 是一个 COCO 格式的 JSON 文件路径。
# from pycocotools.coco import COCO 提供的 COCO 类会读取并解析该 JSON，构建内部索引结构。
# 解析后，self.coco 对象包含：
# self.coco.imgs：一个字典，键为图像 ID，值为图像信息（如 file_name, width, height 等）。
# self.coco.anns：注解字典，键为注解 ID，值为标注信息（包括边界框、类别、分割多边形等）。
# self.coco.cats：类别信息字典。
# 通过 self.coco.getAnnIds()、self.coco.loadAnns()、self.coco.loadImgs() 等方法可以方便地查询标注和图像信息。


        self.img_folder = img_folder#存储图像文件夹路径,这个路径将在 __getitem__ 中与图像文件名拼接，用于读取实际图像文件。

        self.transforms = transforms
        self.ids = list(self.coco.imgs.keys())# 获取所有图像 ID 列表,
        #self.coco.imgs 是一个字典，键是所有图像的唯一 ID。通过 keys() 提取所有 ID，并转为列表。
        #这个列表用于实现 __len__（返回列表长度）和 __getitem__（通过索引获取对应的图像 ID）
    def __getitem__(self, index):
        img_id = self.ids[index]# 根据索引获取图像 ID
        ann_ids = self.coco.getAnnIds(imgIds=img_id) #查询该图像的所有注解 ID
        target_list = self.coco.loadAnns(ann_ids)# 加载注解数据

        img_info = self.coco.loadImgs(img_id)[0]# 获取图像信息（包含文件名）
        path = img_info['file_name']# 获取图像文件名
        img = Image.open(os.path.join(self.img_folder, path)).convert('RGB')# 实际读取图像

        # Get image dimensions
        width = img_info['width']
        height = img_info['height']

        # Create boxes, labels, and masks
        boxes = []
        labels = []
        masks = []

        for ann in target_list:
            x, y, w, h = ann['bbox']
            # Skip invalid bboxes
            if w <= 0 or h <= 0:
                continue

            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            mask = coco_mask.decode(ann['segmentation'])
            masks.append(mask)

        if len(boxes) == 0:
            # Return dummy data for empty images
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([img_id])
        }
        #将 boxes, labels, masks 转为 torch.Tensor，并组装成字典 target。

        # Apply transforms
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

def get_transform(train=True):
    """Get transforms for training or validation."""
    if train:
        return transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
        ])

def get_model(num_classes=2):
    """Create Mask R-CNN model with custom head for tick detection."""
    # Load pretrained model
    model = maskrcnn_resnet50_fpn(pretrained=True)
#加载预训练 ResNet50-FPN 骨干网络。
    # Replace box predictor for custom number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
#替换分类头（box predictor）和掩码头（mask predictor）以适配自定义类别数（包含背景）。
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train for one epoch."""
    #遍历训练数据集一次，计算损失，反向传播，更新模型权重，返回该周期的平均损失值
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #将每个目标字典里的所有张量放进对应设备
        # Forward pass
        try:
            loss_dict = model(images, targets)
# 在 torchvision 的检测模型中，当传入 targets 时，前向传播不返回预测结果，而是直接返回一个损失字典。
# 该字典包含 loss_classifier、loss_box_reg、loss_mask、loss_rpn_box、loss_rpn_cls 等多个分量。
            losses = sum(loss for loss in loss_dict.values())#将所有分量损失求和，得到标量总损失

            # Backward pass
            optimizer.zero_grad()#清空上一批次的梯度（防止累积）
            losses.backward()#计算总损失对模型所有可训练参数的梯度。
            optimizer.step()#根据梯度更新权重。

            total_loss += losses.item()
            num_batches += 1
            #累加总损失并计数，以便在函数末尾计算该 Epoch 的平均损失。
            if batch_idx % 10 == 0:#每 10 个批次打印一次当前批次损失。
                print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(data_loader)}] Loss: {losses.item():.4f}")

        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    #计算一个完整训练周期平均损失
    print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
    return avg_loss

def validate(model, data_loader, device):
    """Validate the model."""
    #简单跑一个 batch 的推理
    model.eval()
    # Simple validation - just run inference
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            break  # Just validate one batch for now
    return 0.0

def main():
    """Main training function."""
    args = parse_args()

    # Override defaults with command line arguments
    train_ann = args.train_ann or TRAIN_ANN
    train_img = args.train_img or TRAIN_IMG
    val_ann = args.val_ann or VAL_ANN
    val_img = args.val_img or VAL_IMG
    num_classes = args.num_classes
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    output_dir = args.output_dir
    model_path = args.model_path

    print("=" * 50)
    print("Insect Detection Training - Mask R-CNN")
    if args.multi_class:
        print("Mode: Multi-class Detection")
    else:
        print("Mode: Single-class (Tick) Detection")
    print("=" * 50)
    print(f"Number of classes: {num_classes}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}")
    print(f"Learning rate: {lr}")

    # Set device，打印设备信息
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get datasets
    print(f"\nLoading datasets...")
    print(f"  Train: {train_ann}")
    print(f"  Val: {val_ann}")


    train_dataset = TickDetectionDataset(
        train_ann, train_img,
        transforms=get_transform(train=True)
    )#数据集实例化，传入参数


    val_dataset = TickDetectionDataset(
        val_ann, val_img,
        transforms=get_transform(train=False)
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    # Get model
    print(f"\nCreating model with {num_classes} classes...")
    model = get_model(num_classes=num_classes)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training loop
    print("\nStarting training...")
    print("-" * 50)

    for epoch in range(num_epochs):#迭代训练每个epoch
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch+1)
        #训练一个epoch调用函数获得这个周期的平均损失
        lr_scheduler.step()

        # Save checkpoint
        if (epoch + 1) % 5 == 0:#每 5 个 epoch 保存检查点（包含模型和优化器状态）。
            checkpoint_path = f"{output_dir}/mask_rcnn_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    torch.save(model.state_dict(), model_path)#保存最终模型权重
    print(f"\nTraining complete! Model saved to: {model_path}")

    # Evaluate on test set (if annotation exists)
    test_ann = val_ann.replace("valid", "test")  # Assume test folder mirrors valid
    test_img = val_img.replace("valid", "test")

    if os.path.exists(test_ann) and os.path.exists(test_img):#若测试集存在，则加载并对测试集进行推理
        print("\nEvaluating on test set...")
        test_dataset = TickDetectionDataset(test_ann, test_img, transforms=get_transform(False))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        model.eval()
        detections = []
        with torch.no_grad():
            for i, (images, targets) in enumerate(test_loader):
                images = [img.to(device) for img in images]
                outputs = model(images)
                detections.append(outputs)
                if i % 50 == 0:
                    print(f"Processed {i}/{len(test_loader)} test images")

        print(f"\nTest set inference complete! Processed {len(test_dataset)} images.")
    else:
        print(f"\nTest set not found at {test_ann}, skipping evaluation.")

if __name__ == "__main__":
    main()