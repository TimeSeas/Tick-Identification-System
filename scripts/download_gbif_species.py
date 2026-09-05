"""
按物种名从 GBIF API（免登录）下载蜱虫图片，支持任意蜱种（含中国蜱种）。

原理：species/match 拿 usageKey → occurrence/search 拿带 media 的记录
→ 取 iNaturalist photo id → 走官方 Open Data S3 桶下载 large(1024px) 图。

用法：
    python scripts/download_gbif_species.py --species "Ixodes persulcatus" "Hyalomma asiaticum" ...
    python scripts/download_gbif_species.py --list scripts/../species_list.txt

- 每物种目标 200 张（--max-per-species 调整），按 photo id 去重
- 断点续跑：已下载的 photo id 自动跳过
- 产出：data/tick_species/<slug>/<photo_id>.jpg + manifest.csv（photo_id/物种/许可/URL）
"""

import argparse
import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

SPECIES_API = "https://api.gbif.org/v1/species/match"
OCC_API = "https://api.gbif.org/v1/occurrence/search"
HEADERS = {"User-Agent": "tick-species-research/1.0 (academic graduation project)"}


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def api_get(url, params, tries=5):
    """带重试的 GET，应对 GBIF API 偶发断连"""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError("unreachable")


def get_usage_key(name):
    r = api_get(SPECIES_API, {"name": name})
    d = r.json()
    if d.get("matchType") not in ("EXACT", "FUZZY") or d.get("rank") != "SPECIES":
        print(f"  {name}: 无法精确匹配物种")
        return None
    return d["usageKey"]


def collect_pending(name, existing, max_per_species):
    """返回待下载 [(photo_id, s3_url, license, obs_url)]"""
    key = get_usage_key(name)
    if key is None:
        return []
    pending, page = [], 1
    while len(existing) + len(pending) < (max_per_species or 10**9) and page <= 10:
        params = {"taxonKey": key, "limit": 300, "offset": (page - 1) * 300}
        r = api_get(OCC_API, params)
        results = r.json().get("results", [])
        if not results:
            break
        for rec in results:
            for m in rec.get("media") or []:
                if m.get("type") != "StillImage":
                    continue
                mid = m.get("identifier", "")
                mm = re.search(r"/photos/(\d+)/", mid)
                if not mm or mm.group(1) in existing:
                    continue
                existing.add(mm.group(1))
                pending.append((mm.group(1),
                                f"https://inaturalist-open-data.s3.amazonaws.com/photos/{mm.group(1)}/large.jpg",
                                m.get("license", ""), rec.get("references", "")))
                if len(existing) >= (max_per_species or 10**9):
                    break
            if len(existing) >= (max_per_species or 10**9):
                break
        page += 1
        time.sleep(0.5)
    return pending


def fetch_one(item, out_dir):
    pid, url, license_code, ref = item
    img = None
    for i in range(3):
        try:
            img = requests.get(url, headers=HEADERS, timeout=120)
            break
        except requests.RequestException:
            if i == 2:
                return None, f"{pid}: 连接失败（重试 3 次）"
            time.sleep(2)
    if img.status_code != 200:
        return None, f"{pid}: HTTP {img.status_code}"
    (out_dir / f"{pid}.jpg").write_bytes(img.content)
    return [pid, license_code, url, ref], None


def download_species(name, out_root, max_per_species, workers):
    slug = slugify(name)
    out_dir = out_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {f.stem for f in out_dir.glob("*.jpg")}
    print(f"== {name} ({slug})，已有 {len(existing)} 张")
    pending = collect_pending(name, existing, max_per_species)
    if not pending:
        print(f"  无新增可下载")
        return len(existing)

    t0 = time.time()
    rows, ok, fail = [], 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, item, out_dir): item for item in pending}
        for fut in as_completed(futures):
            row, err = fut.result()
            if err:
                fail += 1
                print(" ", err)
            else:
                rows.append(row)
                ok += 1
            if (ok + fail) % 50 == 0:
                print(f"  {ok + fail}/{len(pending)}（成功 {ok}，失败 {fail}）")

    manifest_path = out_dir / "manifest.csv"
    mode = "a" if manifest_path.exists() else "w"
    with open(manifest_path, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(["photo_id", "license", "url", "reference"])
        w.writerows(rows)
    print(f"{name}: 本轮 +{ok}（失败 {fail}），共 {len(existing)} 张，"
          f"用时 {(time.time() - t0) / 60:.1f} 分钟")
    return len(existing)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", nargs="+", help="物种学名列表")
    ap.add_argument("--list", help="物种名单文件（每行一个学名，# 开头为注释）")
    ap.add_argument("--out-dir", default="data/tick_species")
    ap.add_argument("--max-per-species", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    names = args.species or []
    if args.list:
        for line in open(args.list, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    if not names:
        ap.error("需要 --species 或 --list")

    for name in names:
        for attempt in range(3):
            try:
                download_species(name, Path(args.out_dir),
                                 args.max_per_species, args.workers)
                break
            except Exception as e:
                print(f"  {name}: 第 {attempt + 1} 次失败（{type(e).__name__}），等待 30s 重试")
                time.sleep(30)
        else:
            print(f"  {name}: 重试 3 次仍失败，跳过（续跑时会再试）")


if __name__ == "__main__":
    main()
