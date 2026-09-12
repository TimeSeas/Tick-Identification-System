"""
昆虫粗分类训练（28 类，文件夹标签零标注成本）
定位：Web 级联的第一级——"这是什么昆虫"。蜱虫分支再走检测器+蜱种细分类。
设计取舍：粗类别视觉差异大，用 224 输入足够（448 留给细粒度蜱种）；加权采样对抗 archive(800+/类) 与 archive3(~100/类) 的长尾。
"""
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as T
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import f1_score, confusion_matrix
from torchvision.models import ResNet50_Weights

# ==================== 配置区 ====================
SPLIT_JSON = "data/insects_split.json"
OUTPUT_DIR = "output/insect_classifier"
INPUT_SIZE = 224
WEIGHTED = True
LABEL_SMOOTHING = 0.1
BATCH_SIZE, LR, EPOCHS, SEED, NUM_WORKERS = 32, 1e-4, 20, 42, 0
# ================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)


class SplitDataset(torch.utils.data.Dataset):
    """读取划分 JSON 的图像数据集，坏图自动跳过"""
    def __init__(self, split_name, class_index, transform=None):
        data = json.load(open(SPLIT_JSON, encoding="utf-8"))[split_name]
        self.items = [(Path(d["path"]), class_index[d["label"]]) for d in data]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        for _ in range(len(self)):
            path, label = self.items[index]
            try:
                img = Image.open(path).convert("RGB")
                return self.transform(img), label
            except Exception:
                index = (index + 1) % len(self)
        raise RuntimeError("没有可打开的图片")


def get_model(num_classes):
    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward(); optimizer.step()
        total_loss += loss.item() * len(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(imgs)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    correct, n = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item(); n += len(imgs)
        all_preds += preds.cpu().tolist()
        all_labels += labels.cpu().tolist()
        all_probs.append(logits.softmax(1).cpu())
    probs = torch.cat(all_probs).max(1).values.tolist()
    return correct / n, f1_score(all_labels, all_preds, average="macro"), all_preds, all_labels, probs


def save_val_csv(dataset, preds, labels, probs, classes):
    with open(f"{OUTPUT_DIR}/val_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["fname", "true", "pred", "prob"])
        for (path, _), t, p, prob in zip(dataset.items, labels, preds, probs):
            w.writerow([Path(path).name, classes[t], classes[p], round(prob * 100, 2)])


def plot_confusion_matrix(preds, labels, classes):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(max(8, len(classes) * 0.5), max(8, len(classes) * 0.45)))
    plt.imshow(cm, cmap="Blues")
    if len(classes) <= 10:
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=7)
    plt.yticks(range(len(classes)), classes, fontsize=7)
    plt.xlabel("预测"); plt.ylabel("真实"); plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")


def main():
    print(device, "| weighted:", WEIGHTED, "| size:", INPUT_SIZE)
    os.makedirs(f"{OUTPUT_DIR}/model", exist_ok=True)
    model_path = f"{OUTPUT_DIR}/model/insect_best.pth"

    split = json.load(open(SPLIT_JSON, encoding="utf-8"))
    classes = sorted({d["label"] for d in split["train"]})
    class_index = {c: i for i, c in enumerate(classes)}
    print(f"{len(classes)} 类")

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    size = INPUT_SIZE
    train_tf = T.Compose([
        T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.75, 1.3333)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(), normalize])
    eval_tf = T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size), T.ToTensor(), normalize])

    train_ds = SplitDataset("train", class_index, train_tf)
    val_ds = SplitDataset("val", class_index, eval_tf)
    test_ds = SplitDataset("test", class_index, eval_tf)

    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    if WEIGHTED:
        counts = np.bincount([i for _, i in train_ds.items])
        weights = torch.tensor([1.0 / counts[i] for _, i in train_ds.items], dtype=torch.float)
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
        train_loader = torch.utils.data.DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = get_model(len(classes))
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = 0
    for epoch in range(EPOCHS):
        loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_acc, val_f1, _, _, _ = evaluate(model, val_loader)
        scheduler.step()
        print(f"epoch {epoch + 1}/{EPOCHS} | loss {loss:.4f} | train_acc {train_acc:.4f} | val_acc {val_acc:.4f} | val_f1 {val_f1:.4f}")
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/model/epoch_{epoch + 1}.pth")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_path)

    model.load_state_dict(torch.load(model_path, weights_only=True))
    val_acc, val_f1, v_p, v_l, v_prob = evaluate(model, val_loader)
    save_val_csv(val_ds, v_p, v_l, v_prob, classes)
    test_acc, test_f1, t_p, t_l, _ = evaluate(model, test_loader)
    plot_confusion_matrix(t_p, t_l, classes)
    print(f"best val_f1: {best_f1:.4f} | test_acc: {test_acc:.4f} | test_f1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
