"""
从 HF fine-grained-challenges 的 lance 数据中提取 Ixodidae（硬蜱科）组图片，
按物种归档到 data/tick_species/<物种slug>/，并写全局 manifest。

前置：已下载 lance 分片到 data/tick_species_raw/fgc/challenges.lance/
用法：
    python scripts/extract_fgc_ixodidae.py --lance data/tick_species_raw/fgc/challenges.lance
"""

import argparse
import csv
import re
from pathlib import Path

import lance
from PIL import Image


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lance", default="data/tick_species_raw/fgc/challenges.lance")
    ap.add_argument("--out-dir", default="data/tick_species")
    ap.add_argument("--group", default="ixodidae")
    args = ap.parse_args()

    ds = lance.dataset(args.lance)
    print("全部列:", ds.schema.names)

    # 图片字节列固定为 image；不用 lance 索引（_indices 未下载），
    # 顺序扫描全部批次后在客户端按 challenge_group 过滤
    keep = ["uuid", "challenge_group", "species", "license_name",
            "source_url", "image"]

    out_root = Path(args.out_dir)
    manifest_path = out_root / "fgc_ixodidae_manifest.csv"
    existing = set()
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            existing = {row[0] for row in csv.reader(f) if row}

    done = skipped = bad = scanned = 0
    import io
    with open(manifest_path, "a", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        if manifest_path.stat().st_size == 0:
            writer.writerow(["uuid", "species", "license", "source_url", "local_path"])
        for batch in ds.to_batches(columns=keep, batch_size=64):
            t = batch.to_pandas()
            t = t[t["challenge_group"].str.lower() == args.group.lower()]
            scanned += len(t)
            for row in t.itertuples(index=False):
                uuid = row.uuid
                if uuid in existing:
                    skipped += 1
                    continue
                species = row.species or "unknown"
                lic = row.license_name or ""
                src = row.source_url or ""
                data = row.image
                if not data:
                    bad += 1
                    continue
                slug = slugify(species)
                out_dir = out_root / slug
                out_dir.mkdir(parents=True, exist_ok=True)
                try:
                    img = Image.open(io.BytesIO(data))
                    img.load()
                    # 统一存为 JPEG 方便后续训练管线
                    img.convert("RGB").save(out_dir / f"{uuid}.jpg", quality=95)
                    writer.writerow([uuid, species, lic, src,
                                     str((out_dir / f"{uuid}.jpg").as_posix())])
                    existing.add(uuid)
                    done += 1
                except Exception:
                    bad += 1
                    continue
                if done % 200 == 0:
                    print(f"  已解出 {done} 张（已扫描 {scanned} 条 ixodidae）...")
    print(f"完成：ixodidae 记录 {scanned} 条，新增 {done} 张，跳过 {skipped}，失败 {bad}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
