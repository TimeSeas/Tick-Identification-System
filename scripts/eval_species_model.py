"""
蜱种分类模型评估：在指定测试集上输出整体与逐类指标（论文对比实验用）

用法：
    python scripts/eval_species_model.py \
        --model output/tick_species_31_weighted/model/tick_species_best.pth \
        --classes-split data/tick_species/split_31cls.json \
        --test-split data/tick_species/split_clean.json \
        --out output/基线对比/old31_on_clean_test.csv

说明：
- --classes-split 用于还原模型训练时的类别表（31 类模型用 split_31cls）
- --test-split 是评估数据集（新旧模型都用清洗后 split_clean，才公平可比）
- 逐类指标：accuracy；整体：acc / macro-F1 / weighted-F1
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import f1_score
from torchvision.models import ResNet50_Weights

INPUT_SIZE = 224  # 旧 31 类模型是 224 训练；新 448 模型评估时需对应改为 448


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, items, class_index, transform):
        self.items = items
        self.class_index = class_index
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--classes-split", required=True)
    ap.add_argument("--test-split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    # 模型训练时的类别表
    train_split = json.load(open(args.classes_split, encoding="utf-8"))
    classes = sorted({d["label"] for d in train_split["train"]})
    class_index = {c: i for i, c in enumerate(classes)}

    # 评估测试集（新的清洗后划分）
    test_split = json.load(open(args.test_split, encoding="utf-8"))
    items = [(Path(d["path"]), class_index[d["label"]]) for d in test_split["test"]]

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    size = args.input_size
    eval_tf = T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size), T.ToTensor(), normalize])

    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    loader = torch.utils.data.DataLoader(EvalDataset(items, class_index, eval_tf),
                                         batch_size=args.batch_size, shuffle=False,
                                         num_workers=0)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs).argmax(1)
            all_preds += preds.tolist()
            all_labels += labels.tolist()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 整体指标
    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    # 逐类指标
    rows = []
    for cls in classes:
        i = class_index[cls]
        mask = all_labels == i
        if mask.sum() == 0:
            continue
        cls_acc = (all_preds[mask] == i).mean()
        rows.append({"类别": cls, "样本数": int(mask.sum()),
                     "正确数": int((all_preds[mask] == i).sum()),
                     "准确率%": round(cls_acc * 100, 2)})
    rows.sort(key=lambda r: r["准确率%"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["类别", "样本数", "正确数", "准确率%"])
        w.writeheader()
        w.writerows(rows)

    print(f"整体: acc {acc * 100:.2f}% | macro-F1 {macro_f1:.4f} | weighted-F1 {weighted_f1:.4f}")
    print(f"逐类明细已保存 -> {args.out}")
    bad = [r for r in rows if r["准确率%"] < 60]
    print(f"准确率 <60% 的类: {len(bad)} 个" + (f"，最差: {bad[0]}" if bad else ""))


if __name__ == "__main__":
    main()
