"""
SAM-based dataset enhancement.

Modes:
  fill - 用现有 bbox 标注作 SAM box 提示生成分割多边形，写回 COCO JSON
         （原文件备份为 _annotations.coco.bbox_only.json，categories 清理为单条 ticks id=1）
  auto - 对"文件夹即类别"的未标注图片跑 SAM 自动掩码生成（AMG），输出 COCO JSON、
         X-AnyLabeling 复核副本（review/）和掩码预览图（previews/）

Usage:
  python scripts/sam_auto_label.py --mode fill \
      --coco "ticks image detection.v5i.coco/train/_annotations.coco.json" \
             "ticks image detection.v5i.coco/valid/_annotations.coco.json" \
             "ticks image detection.v5i.coco/test/_annotations.coco.json"

  python scripts/sam_auto_label.py --mode auto \
      --input-dirs archive2/dataset/train archive3/farm_insects \
      --classes tick mite aphid colorado_potato_beetle thrips corn_earworm fruit_fly \
      --per-class 30 --out-dir output/sam_pilot
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 文件夹名 -> 28 类 slug（与 models/classes.txt 一致）
CLASS_MAPPING = {
    # archive2
    "tick": "tick",
    "mite": "mite",
    # archive（11 类）
    "Ant": "ant", "Bee": "bee", "Beetle": "beetle", "Butterfly": "butterfly",
    "Dragonfly": "dragonfly", "Fly": "fly", "Grasshopper": "grasshopper",
    "Ladybug": "ladybug", "Mosquito": "mosquito", "Spider": "spider", "Wasp": "wasp",
    # archive3（15 类显示名）
    "Africanized Honey Bees (Killer Bees)": "africanized_honey_bee",
    "Aphids": "aphid",
    "Armyworms": "armyworm",
    "Brown Marmorated Stink Bugs": "brown_marmorated_stink_bug",
    "Cabbage Loopers": "cabbage_looper",
    "Citrus Canker": "citrus_canker",
    "Colorado Potato Beetles": "colorado_potato_beetle",
    "Corn Borers": "corn_borer",
    "Corn Earworms": "corn_earworm",
    "Fall Armyworms": "fall_armyworm",
    "Fruit Flies": "fruit_fly",
    "Spider Mites": "spider_mite",
    "Thrips": "thrips",
    "Tomato Hornworms": "tomato_hornworm",
    "Western Corn Rootworms": "western_corn_rootworm",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

AMG_KWARGS = dict(
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.90,
    min_mask_region_area=64,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
)

POLYGON_EPSILON = 0.5


def build_sam(checkpoint, device):
    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
    return sam.to(device)


def mask_to_polygons(mask):
    """bool 掩码 -> COCO 多边形列表（扁平 [x,y,...] 格式，忽略 <1% 掩码面积的小轮廓）"""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_area = max(int(mask.sum()), 1)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < 0.01 * mask_area:
            continue
        c = cv2.approxPolyDP(c, POLYGON_EPSILON, closed=True)
        if len(c) >= 3:
            polys.append(c.reshape(-1).astype(float).tolist())
    return polys


def mask_to_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()),
            float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]


def polys_to_bbox_area(polys):
    """由保留的多边形推导 bbox 与面积，保证与 segmentation 严格一致"""
    xs = [p[i] for p in polys for i in range(0, len(p), 2)]
    ys = [p[i] for p in polys for i in range(1, len(p), 2)]
    bbox = [min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1]
    area = sum(abs(cv2.contourArea(np.array(p, dtype=np.float32).reshape(-1, 2)))
               for p in polys)
    return bbox, area


def fill_masks(coco_paths, checkpoint, device):
    """模式 fill：bbox 提示 -> 掩码 -> 多边形写回原 COCO JSON"""
    from segment_anything import SamPredictor
    predictor = SamPredictor(build_sam(checkpoint, device))

    for coco_path in coco_paths:
        coco_path = Path(coco_path)
        data = json.load(open(coco_path, encoding="utf-8"))
        backup = coco_path.with_name("_annotations.coco.bbox_only.json")
        if not backup.exists():
            shutil.copy2(coco_path, backup)

        img_dir = coco_path.parent
        anns_by_img = {}
        for a in data["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)

        missing_imgs = empty_masks = 0
        for idx, img in enumerate(tqdm(data["images"], desc=coco_path.name, mininterval=30)):
            anns = [a for a in anns_by_img.get(img["id"], []) if not a.get("segmentation")]
            if not anns:
                continue
            img_path = img_dir / img["file_name"]
            if not img_path.exists():
                missing_imgs += 1
                continue
            im = np.array(Image.open(img_path).convert("RGB"))
            predictor.set_image(im)
            h, w = im.shape[:2]
            for a in anns:
                x, y, bw, bh = a["bbox"]
                bw, bh = max(bw, 2.0), max(bh, 2.0)  # 过小框至少撑到 2px 供提示
                box = [max(x, 0), max(y, 0), min(x + bw, w - 1), min(y + bh, h - 1)]
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue  # 框完全出界，保留原空 segmentation
                masks, scores, _ = predictor.predict(
                    box=np.array(box, dtype=np.float32)[None, :], multimask_output=True)
                mask = masks[scores.argmax()].astype(bool)
                polys = mask_to_polygons(mask)
                if not polys:
                    empty_masks += 1
                    continue
                a["segmentation"] = polys
                a["bbox"], a["area"] = polys_to_bbox_area(polys)

            # 断点续跑：每 50 张增量写盘，被杀后重启自动跳过已完成标注
            if (idx + 1) % 50 == 0:
                json.dump(data, open(coco_path, "w", encoding="utf-8"), ensure_ascii=False)

        data["categories"] = [{"id": 1, "name": "ticks", "supercategory": "ticks"}]
        json.dump(data, open(coco_path, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"{coco_path.name}: {missing_imgs} 缺失图片, {empty_masks} 掩码为空")


def run_auto(input_dirs, classes, per_class, seed, checkpoint, out_dir, device):
    """模式 auto：AMG 自动标注，类别取文件夹名，输出 COCO + 复核副本 + 预览"""
    from segment_anything import SamAutomaticMaskGenerator
    generator = SamAutomaticMaskGenerator(build_sam(checkpoint, device), **AMG_KWARGS)

    out_dir = Path(out_dir)
    preview_dir = out_dir / "previews"
    review_dir = out_dir / "review"
    preview_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    # 断点续跑：加载已有产物，跳过已完成图片
    out_json = out_dir / "annotations.json"
    if out_json.exists():
        coco = json.load(open(out_json, encoding="utf-8"))
        cat_ids = {c["name"]: c["id"] for c in coco["categories"]}
        done = {im["file_name"] for im in coco["images"]}
        img_id = max((im["id"] for im in coco["images"]), default=0) + 1
        ann_id = max((a["id"] for a in coco["annotations"]), default=0) + 1
    else:
        coco = {"images": [], "annotations": [], "categories": []}
        cat_ids, done = {}, set()
        img_id = ann_id = 1
    report, rng = {}, random.Random(seed)

    reverse = {slug: folder for folder, slug in CLASS_MAPPING.items()}
    for cls in classes:
        folder_name = reverse.get(cls, cls)  # 接受 slug 或文件夹名
        folder = next((Path(d) / folder_name for d in input_dirs
                       if (Path(d) / folder_name).is_dir()), None)
        if folder is None:
            print(f"skip {cls}: 目录不存在")
            continue
        slug = CLASS_MAPPING[folder_name]
        imgs = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in IMG_EXTS]
        chosen = rng.sample(imgs, min(per_class, len(imgs)))
        kept = empty = 0
        for f in tqdm(chosen, desc=slug, mininterval=30):
            if f.as_posix() in done:
                continue
            im = np.array(Image.open(f).convert("RGB"))
            h, w = im.shape[:2]
            cands = []
            for m in generator.generate(im):
                if m["area"] < 0.001 * h * w:
                    continue
                seg = np.asarray(m["segmentation"], dtype=bool)
                if seg[0].any() or seg[-1].any() or seg[:, 0].any() or seg[:, -1].any():
                    continue  # 贴边多为背景
                cands.append(m)
            cands.sort(key=lambda m: -m["area"])
            cands = cands[:3]
            if not cands:
                empty += 1
                continue

            coco["images"].append(
                {"id": img_id, "file_name": f.as_posix(), "width": w, "height": h})
            shapes = []
            for m in cands:
                seg = np.asarray(m["segmentation"], dtype=bool)
                polys = mask_to_polygons(seg)
                if slug not in cat_ids:
                    cat_ids[slug] = len(coco["categories"]) + 1
                    coco["categories"].append(
                        {"id": cat_ids[slug], "name": slug, "supercategory": "insect"})
                for poly in polys:
                    bbox, area = polys_to_bbox_area([poly])
                    coco["annotations"].append({
                        "id": ann_id, "image_id": img_id, "category_id": cat_ids[slug],
                        "bbox": bbox, "area": area,
                        "iscrowd": 0, "segmentation": [poly]})
                    ann_id += 1
                    shapes.append({
                        "label": slug,
                        "points": [[poly[i], poly[i + 1]] for i in range(0, len(poly), 2)],
                        "group_id": None, "description": "", "difficult": False,
                        "shape_type": "polygon", "flags": {}, "attributes": {},
                        "score": round(float(m["predicted_iou"]), 4)})
                kept += 1

            # X-AnyLabeling 复核副本（与 archive2 现有 per-image JSON 同构）
            cls_review = review_dir / slug
            cls_review.mkdir(exist_ok=True)
            shutil.copy2(f, cls_review / f.name)
            json.dump({"version": "2.3.6", "flags": {}, "shapes": shapes},
                      open(cls_review / (f.stem + ".json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)

            # 预览图
            prev = im.copy()
            for m in cands:
                seg = np.asarray(m["segmentation"], dtype=bool)
                for poly in mask_to_polygons(seg):
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.polylines(prev, [pts], True, (0, 0, 255), 2)
                bx = mask_to_bbox(seg)
                if bx:
                    cv2.rectangle(prev, (int(bx[0]), int(bx[1])),
                                  (int(bx[0] + bx[2]), int(bx[1] + bx[3])), (0, 255, 0), 1)
            cv2.putText(prev, slug, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.imwrite(str(preview_dir / f"{slug}_{f.stem}.png"),
                        cv2.cvtColor(prev, cv2.COLOR_RGB2BGR))
            img_id += 1

            # 断点续跑：逐张写盘
            json.dump(coco, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        report[slug] = {"images": len(chosen), "masks_kept": kept, "empty": empty}

    print("\n报告（slug | 图片 | 保留掩码 | 空结果）:")
    for slug, r in report.items():
        print(f"  {slug:24s} {r['images']:4d} {r['masks_kept']:6d} {r['empty']:4d}")
    print(f"COCO 输出: {out_json}")


def preview_coco(coco_path, out_dir, n, seed, ref_path=None):
    """模式 preview：从 COCO JSON 抽样图片，叠加多边形(红)与 bbox(绿)，
    ref_path 给原标注 JSON 时同时画原 bbox(黄) 做对比"""
    from pycocotools.coco import COCO
    coco = COCO(coco_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = {}
    if ref_path:
        for a in json.load(open(ref_path, encoding="utf-8"))["annotations"]:
            ref[a["id"]] = a["bbox"]

    by_img = {}
    for a in coco.loadAnns(coco.getAnnIds()):
        by_img.setdefault(a["image_id"], []).append(a)
    imgs_with_anns = [im for im in coco.dataset["images"] if im["id"] in by_img]
    rng = random.Random(seed)

    for im in rng.sample(imgs_with_anns, min(n, len(imgs_with_anns))):
        base = coco_path.parent
        img_path = base / im["file_name"]
        if not img_path.exists():
            img_path = Path(im["file_name"])
        img = np.array(Image.open(img_path).convert("RGB"))
        vis = img.copy()
        for a in by_img[im["id"]]:
            for poly in a["segmentation"]:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
            bx, by_, bw, bh = a["bbox"]
            cv2.rectangle(vis, (int(bx), int(by_)), (int(bx + bw), int(by_ + bh)),
                          (0, 255, 0), 1)
            if a["id"] in ref:  # 原 bbox 对比（黄色虚线）
                ox, oy, ow, oh = ref[a["id"]]
                cv2.rectangle(vis, (int(ox), int(oy)), (int(ox + ow), int(oy + oh)),
                              (0, 255, 255), 1)
        out = out_dir / f"{coco_path.parent.name}_{im['file_name']}"
        cv2.imwrite(str(out), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"已生成 {min(n, len(imgs_with_anns))} 张预览 -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["auto", "fill", "preview"], required=True)
    ap.add_argument("--checkpoint", default="models/sam_vit_b_01ec64.pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--coco", nargs="+", help="fill 模式：COCO JSON 路径（可多个 split）")
    ap.add_argument("--input-dirs", nargs="+", help="auto 模式：类文件夹所在的根目录（可多个）")
    ap.add_argument("--classes", nargs="+", help="auto 模式：文件夹名（CLASS_MAPPING 的 key）")
    ap.add_argument("--per-class", type=int, default=30, help="auto 模式：每类抽样数，0=全量")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="output/sam_pilot")
    ap.add_argument("--n", type=int, default=20, help="preview 模式：抽样图片数")
    ap.add_argument("--ref-json", help="preview 模式：原标注 JSON（画黄色对比框）")
    args = ap.parse_args()

    if args.mode == "preview":
        if not args.coco:
            sys.exit("preview 模式需要 --coco（一个 JSON）")
        preview_coco(Path(args.coco[0]), args.out_dir, args.n, args.seed, args.ref_json)
        return

    if not Path(args.checkpoint).exists():
        sys.exit(f"checkpoint 不存在: {args.checkpoint}")

    if args.mode == "fill":
        if not args.coco:
            sys.exit("fill 模式需要 --coco")
        fill_masks(args.coco, args.checkpoint, args.device)
    else:
        if not args.input_dirs or not args.classes:
            sys.exit("auto 模式需要 --input-dirs 和 --classes")
        run_auto(args.input_dirs, args.classes, args.per_class, args.seed,
                 args.checkpoint, args.out_dir, args.device)


if __name__ == "__main__":
    main()
