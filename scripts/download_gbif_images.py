"""
从 GBIF 达尔文核心归档（README 的 3 个 DOI 已下载到 data/tick_species_raw/）中
提取 iNaturalist 图片并下载（1024px large 版本，走官方 Open Data S3 桶，
static.inaturalist.org 在本机网络被 403）。

用法：
    python scripts/download_gbif_images.py --max-per-species 400 --workers 8

- 按 gbifID 去重（同一观测记录只取第一张）
- 断点续跑：已下载的 gbifID 自动跳过
- 产出：data/tick_species/<slug>/<gbifID>.jpg + manifest.csv（gbifID/photo_id/license/url）
"""

import argparse
import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

SPECIES_DIRS = [
    "amblyomma_americanum",
    "dermacentor_variabilis",
    "ixodes_scapularis",
]
HEADERS = {"User-Agent": "tick-species-research/1.0 (academic graduation project)"}


def collect_pending(mm_path, existing, max_per_species):
    """返回待下载列表 [(gbif_id, s3_url, license, photo_id)]"""
    pending, seen = [], set(existing)
    with open(mm_path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        cols = {name: i for i, name in enumerate(header)}
        for line in f:
            if max_per_species and len(seen) >= max_per_species:
                break
            parts = line.rstrip("\n").split("\t")
            get = lambda c: parts[cols[c]] if cols[c] < len(parts) else ""
            gbif_id, mtype, url = get("gbifID"), get("type"), get("identifier")
            if mtype != "StillImage" or gbif_id in seen:
                continue
            m = re.search(r"/photos/(\d+)/", url)
            if not m:
                continue
            seen.add(gbif_id)
            pending.append((gbif_id,
                            f"https://inaturalist-open-data.s3.amazonaws.com/photos/{m.group(1)}/large.jpg",
                            get("license"), m.group(1)))
    return pending


def fetch_one(item, out_dir):
    gbif_id, url, license_code, photo_id = item
    try:
        img = requests.get(url, headers=HEADERS, timeout=120)
    except requests.RequestException as e:
        return None, f"{gbif_id}: {e}"
    if img.status_code != 200:
        return None, f"{gbif_id}: HTTP {img.status_code}"
    (out_dir / f"{gbif_id}.jpg").write_bytes(img.content)
    return [gbif_id, photo_id, license_code, url], None


def download_species(slug, raw_dir, out_root, max_per_species, workers):
    mm = raw_dir / slug / "multimedia.txt"
    if not mm.exists():
        print(f"skip {slug}: 无 multimedia.txt")
        return 0
    out_dir = out_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {f.stem for f in out_dir.glob("*.jpg")}
    pending = collect_pending(mm, existing, max_per_species)
    if not pending:
        print(f"{slug}: 无待下载（已有 {len(existing)} 张）")
        return len(existing)

    t0 = time.time()
    rows, ok, fail = [], 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, item, out_dir): item for item in pending}
        for fut in as_completed(futures):
            row, err = fut.result()
            if err:
                fail += 1
                print(err)
            else:
                rows.append(row)
                ok += 1
            if (ok + fail) % 100 == 0:
                print(f"  {slug}: {ok + fail}/{len(pending)}（成功 {ok}，失败 {fail}）")

    manifest_path = out_dir / "manifest.csv"
    mode = "a" if manifest_path.exists() else "w"
    with open(manifest_path, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(["gbifID", "photo_id", "license", "url"])
        w.writerows(rows)
    print(f"{slug}: 本轮成功 {ok}，失败 {fail}，共 {len(existing) + ok} 张，"
          f"用时 {(time.time() - t0) / 60:.1f} 分钟")
    return len(existing) + ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="data/tick_species_raw")
    ap.add_argument("--out-dir", default="data/tick_species")
    ap.add_argument("--max-per-species", type=int, default=0, help="每物种上限，0=全部")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    for slug in SPECIES_DIRS:
        print(f"== {slug}")
        download_species(slug, raw_dir, Path(args.out_dir),
                         args.max_per_species, args.workers)


if __name__ == "__main__":
    main()
