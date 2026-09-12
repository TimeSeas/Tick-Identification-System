"""
小目标补标数据合并进主数据集 train split（检测 v3 训练前）

- 用户标注在 data/tick_small_targets_raw/（X-AnyLabeling JSON，rectangle 形状）
- 矩形 → 4 点多边形（mask 监督）；标签统一为 tick（单类检测，蜱种信息在文件名）
- 原标注文件先备份 _annotations.coco.pre_small_target.json

用法：
    python scripts/merge_small_targets.py
"""
import json
import shutil
from pathlib import Path

RAW_DIR = Path("data/tick_small_targets_raw")
TRAIN_DIR = Path("ticks image detection.v5i.coco/train")
ANN_PATH = TRAIN_DIR / "_annotations.coco.json"


def main():
    # 备份原标注
    backup = TRAIN_DIR / "_annotations.coco.pre_small_target.json"
    if not backup.exists():
        shutil.copy2(ANN_PATH, backup)
        print(f"已备份原标注 -> {backup.name}")

    data = json.load(open(ANN_PATH, encoding="utf-8"))
    next_img_id = max(im["id"] for im in data["images"]) + 1
    next_ann_id = max(a["id"] for a in data["annotations"]) + 1
    added_img = added_ann = 0

    for jf in sorted(RAW_DIR.glob("*.json")):
        img_file = jf.with_suffix(".jpg")
        if not img_file.exists():
            print(f"  跳过（无图）: {jf.name}")
            continue
        label = json.load(open(jf, encoding="utf-8"))
        shapes = label.get("shapes", [])
        if not shapes:
            continue
        h = label.get("imageHeight")
        w = label.get("imageWidth")
        # 目标文件名加 small_ 前缀避免与现有图片撞名
        dest_name = f"small_{jf.stem}.jpg"
        shutil.copy2(img_file, TRAIN_DIR / dest_name)

        data["images"].append({
            "id": next_img_id,
            "file_name": dest_name,
            "height": h, "width": w,
        })
        for s in shapes:
            pts = s.get("points") or []
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            data["annotations"].append({
                "id": next_ann_id,
                "image_id": next_img_id,
                "category_id": 1,          # tick 单类（蜱种信息保留在文件名）
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
                "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]],  # 矩形→4点多边形
            })
            next_ann_id += 1
            added_ann += 1
        next_img_id += 1
        added_img += 1

    json.dump(data, open(ANN_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"合并完成：+{added_img} 图 / +{added_ann} 标注")
    print(f"train 现状：{len(data['images'])} 图 / {len(data['annotations'])} 标注")


if __name__ == "__main__":
    main()
