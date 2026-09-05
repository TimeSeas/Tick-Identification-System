"""Verify COCO dataset format for tick detection."""

import json
import os
from pycocotools.coco import COCO

DATA_ROOT = "ticks image detection.v5i.coco"
splits = ["train", "valid", "test"]

print("=== COCO Dataset Verification ===\n")

for split in splits:
    ann_file = f"{DATA_ROOT}/{split}/_annotations.coco.json"
    if not os.path.exists(ann_file):
        print(f"{split}: No annotation file found")
        continue

    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    # Get annotation statistics
    total_annotations = 0
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        total_annotations += len(ann_ids)

    avg_ann = total_annotations / len(img_ids) if img_ids else 0

    # Get categories
    cats = coco.loadCats(coco.getCatIds())
    cat_names = [c['name'] for c in cats]

    print(f"[{split}]")
    print(f"  Images: {len(img_ids)}")
    print(f"  Annotations: {total_annotations}")
    print(f"  Avg annotations/image: {avg_ann:.2f}")
    print(f"  Categories: {cat_names}")
    print()