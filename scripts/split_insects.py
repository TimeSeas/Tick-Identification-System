"""
昆虫粗分类数据划分：扫三份文件夹分类数据（archive2/archive/archive3），
按 CLASS_MAPPING 映射成 28 类 slug，8:1:1 分层划分。

用法：
    python scripts/split_insects.py --out data/insects_split.json
"""
import argparse
import json
import random
from pathlib import Path

from sam_auto_label import CLASS_MAPPING  # 与 SAM 标注共用同一套类别映射

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
SOURCE_DIRS = ["archive2/dataset/train", "archive", "archive3/farm_insects"]


def collect():
    """返回 {slug: [图片路径]}"""
    per_class = {}
    for d in SOURCE_DIRS:
        for sub in sorted(Path(d).iterdir()):
            if not sub.is_dir() or sub.name == "annotations":
                continue
            slug = CLASS_MAPPING.get(sub.name)
            if slug is None:
                print(f"  跳过未映射文件夹: {sub}")
                continue
            imgs = sorted(f for f in sub.iterdir() if f.suffix.lower() in IMG_EXTS)
            per_class.setdefault(slug, []).extend(imgs)
    return per_class


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/insects_split.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    per_class = collect()
    splits = {"train": [], "val": [], "test": []}
    for slug in sorted(per_class):
        imgs = sorted(per_class[slug])
        rng.shuffle(imgs)
        n = len(imgs)
        n_test = max(1, round(n * 0.1))
        n_val = max(1, round(n * 0.1))
        n_train = n - n_test - n_val
        for p in imgs[:n_train]:
            splits["train"].append({"path": p.as_posix(), "label": slug})
        for p in imgs[n_train:n_train + n_val]:
            splits["val"].append({"path": p.as_posix(), "label": slug})
        for p in imgs[n_train + n_val:]:
            splits["test"].append({"path": p.as_posix(), "label": slug})
    rng.shuffle(splits["train"])

    json.dump(splits, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(per_class)} 类 | train {len(splits['train'])} / val {len(splits['val'])} / test {len(splits['test'])}")
    for slug in sorted(per_class):
        print(f"  {slug}: {len(per_class[slug])}")


if __name__ == "__main__":
    main()
