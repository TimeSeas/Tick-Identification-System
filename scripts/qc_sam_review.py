"""
SAM 试点标注自动质检：给每张图按疑点规则打分，只让人复核"机器最没把握"的子集
（主动学习思路：复核工作量最小化）

用法：
    python scripts/qc_sam_review.py --dir output/sam_pilot/review --out output/sam_pilot/qc_report.csv

规则（触发任一条即列入待复核清单）：
    - 标注数为 0                    （SAM 漏检）
    - 任一掩码置信度 < 0.85         （predicted_iou，存于 shape 的 score 字段）
    - 单图标注数 > 3                （疑似过分割）
    - 掩码面积占比 <0.2% 或 >40%    （碎屑 / 背景整块）
"""
import argparse
import csv
import json
from pathlib import Path


def qc_one(json_path):
    """返回 (slug, 文件名, 疑点列表)；无返回 None 表示通过质检"""
    data = json.load(open(json_path, encoding="utf-8"))
    shapes = data.get("shapes", [])
    h, w = data.get("imageHeight"), data.get("imageWidth")
    issues = []
    if not shapes:
        issues.append("零标注")
    if len(shapes) > 3:
        issues.append(f"标注数过多({len(shapes)})")
    for s in shapes:
        score = s.get("score")
        if isinstance(score, (int, float)) and score < 0.85:
            issues.append(f"低置信度({score:.2f})")
            break
    if h and w and shapes:
        for s in shapes:
            pts = s.get("points") or []
            if len(pts) < 3:
                issues.append("退化多边形")
                break
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys)) / (w * h)
            if area < 0.002:
                issues.append("掩码过小(<0.2%)")
                break
            if area > 0.4:
                issues.append("掩码过大(>40%)")
                break
    return issues


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="output/sam_pilot/review")
    ap.add_argument("--out", default="output/sam_pilot/qc_report.csv")
    args = ap.parse_args()

    root = Path(args.dir)
    rows, pass_n = [], 0
    for jf in sorted(root.rglob("*.json")):
        issues = qc_one(jf)
        if not issues:
            pass_n += 1
            continue
        rows.append({
            "slug": jf.parent.name,
            "filename": jf.stem,
            "issues": " | ".join(issues),
            "path": jf.as_posix(),
        })

    rows.sort(key=lambda r: -len(r["issues"].split("|")))  # 疑点多的排前面
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["slug", "filename", "issues", "path"])
        w.writeheader()
        w.writerows(rows)

    total = pass_n + len(rows)
    print(f"共 {total} 张 | 通过质检 {pass_n} 张（直接采用）| 待复核 {len(rows)} 张")
    print(f"复核清单 -> {args.out}（按疑点数量排序，从上往下复核即可）")


if __name__ == "__main__":
    main()
