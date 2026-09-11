"""
训练集清洗（检测模型辅助）：用现成蜱虫检测模型批量筛查蜱种图片
按检测结果分三桶，人工复核后决定删除，**本脚本不删除任何文件**

用法：
    python scripts/clean_species_data.py --min-images 50

输出（output/species_cleaning/）：
    clean_report.csv      全部图片的检测结果清单（含分桶）
    review/no_tick/       零检出图片副本（疑似非蜱/坏图 → 复核）
    review/multi_tick/    多只蜱同框副本（标签可能混杂 → 复核）
    review/small_box/     检出框过小副本（模糊/远景 → 复核）
断点续跑：已记录在 CSV 里的图片自动跳过
"""
import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

MODEL_PATH = "output/mask_rcnn_ticks.pth"
OUT_DIR = Path("output/species_cleaning")
CONFIDENCE = 0.5        # 与 detect.py 同口径
SMALL_BOX_RATIO = 0.005 # 检测框面积 < 图像 0.5% 视为"目标过小"（疑似模糊/远景）
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model():
    """与 detect.py 一致的 2 类 Mask R-CNN"""
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    return model.to(device).eval()


def detect(model, img_rgb):
    """返回 [(x1,y1,x2,y2,score)]，按置信度过滤"""
    tensor = torch.tensor(img_rgb).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        outputs = model(tensor.unsqueeze(0).to(device))[0]
    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    keep = [(b[0], b[1], b[2], b[3], s) for b, s in zip(boxes, scores)
            if s >= CONFIDENCE]
    return sorted(keep, key=lambda x: -x[4])


def bucket_of(dets, h, w):
    """分桶规则：零检出=疑似非蜱；多检出=标签可能混杂；小框=疑似模糊"""
    if not dets:
        return "no_tick"
    if len(dets) > 1:
        return "multi_tick"
    x1, y1, x2, y2, _ = dets[0]
    ratio = ((x2 - x1) * (y2 - y1)) / (h * w)
    return "small_box" if ratio < SMALL_BOX_RATIO else "keep"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/tick_species")
    ap.add_argument("--min-images", type=int, default=50, help="只筛>=N 张的类，0=全部")
    args = ap.parse_args()

    root = Path(args.root)
    for bucket in ("no_tick", "multi_tick", "small_box"):
        (OUT_DIR / "review" / bucket).mkdir(parents=True, exist_ok=True)

    report_path = OUT_DIR / "clean_report.csv"
    done = set()
    if report_path.exists():
        with open(report_path, encoding="utf-8-sig") as f:
            done = {row["path"] for row in csv.DictReader(f)}

    model = get_model()
    rows = []
    targets = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name == "unknown":
            continue
        imgs = [f for f in d.glob("*") if f.suffix.lower() in IMG_EXTS]
        if args.min_images and len(imgs) < args.min_images:
            continue
        for f in imgs:
            if f.as_posix() not in done:
                targets.append((d.name, f))

    print(f"待筛查: {len(targets)} 张")
    for i, (slug, f) in enumerate(targets):
        try:
            img = Image.open(f).convert("RGB")
        except Exception:
            print(f"  打开失败: {f}")
            continue
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        dets = detect(model, img_np)
        b = bucket_of(dets, h, w)
        row = {
            "species_slug": slug,
            "filename": f.name,
            "path": f.as_posix(),
            "bucket": b,
            "num_detections": len(dets),
            "max_score": round(dets[0][4], 4) if dets else 0,
        }
        rows.append(row)
        if b != "keep":
            shutil.copy2(f, OUT_DIR / "review" / b / f"{slug}_{f.name}")
        if (i + 1) % 200 == 0:
            print(f"  已筛 {i + 1}/{len(targets)}")
            _append_csv(report_path, rows, first=not report_path.exists())
            rows = []

    _append_csv(report_path, rows, first=not report_path.exists())

    from collections import Counter
    print("完成。分桶统计:")
    with open(report_path, encoding="utf-8-sig") as f:
        cnt = Counter(r["bucket"] for r in csv.DictReader(f))
    for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v} 张")
    print(f"复核目录: {OUT_DIR / 'review'}（人工确认后手动删除原图）")


def _append_csv(report_path, rows, first=False):
    with open(report_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["species_slug", "filename", "path",
                                          "bucket", "num_detections", "max_score"])
        if first:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
