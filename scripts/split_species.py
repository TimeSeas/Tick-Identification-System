"""
按最小样本数筛选蜱种并生成分层划分 JSON（8:1:1，seed 可复现）。

用法：
    python scripts/split_species.py --min-per-class 50  --out data/tick_species/split_31cls.json
    python scripts/split_species.py --min-per-class 200 --out data/tick_species/split_11cls.json

产出格式与 data/tick_species/split_3cls.json 一致：
    {"train": [{"path": ..., "label": slug}, ...], "val": [...], "test": [...]}
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/tick_species")
    ap.add_argument("--min-per-class", type=int, default=50, help="每类最少图片数")
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="每类上限（0=不限），用于压平极端长尾的消融实验")
    ap.add_argument("--out", default="data/tick_species/split_species.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    rng = random.Random(args.seed)
    per_class = defaultdict(list)
    for d in sorted(root.iterdir()):
        if d.name == "unknown" or not d.is_dir():
            continue
        imgs = sorted(d.glob("*.jpg"))
        if len(imgs) < args.min_per_class:
            continue
        if args.max_per_class and len(imgs) > args.max_per_class:
            imgs = rng.sample(imgs, args.max_per_class)
        per_class[d.name] = imgs

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
    rng.shuffle(splits["train"])  # 训练集跨类打乱，DataLoader 不再需要 shuffle

    json.dump(splits, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(per_class)} 类 | train {len(splits['train'])} / val {len(splits['val'])} / test {len(splits['test'])}")
    for s in sorted(per_class):
        print(f"  {s}: {len(per_class[s])} 张")


if __name__ == "__main__":
    main()
