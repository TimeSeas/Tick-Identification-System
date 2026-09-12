"""
给 X-AnyLabeling 标注 JSON 补齐 2.4.x 校验必需字段（imagePath/imageHeight/imageWidth）
旧版 sam_auto_label.py 生成的 JSON 缺这些字段，2.4.x 打开会提示"不是有效的标签文件"

用法：
    python scripts/fix_xanylabeling_json.py --dir output/sam_pilot/review
"""
import argparse
import json
from pathlib import Path

from PIL import Image


def fix_dir(d):
    d = Path(d)
    fixed = 0
    for jf in sorted(d.rglob("*.json")):
        data = json.load(open(jf, encoding="utf-8"))
        if "imageData" in data and all("visible" in s for s in data.get("shapes", [])):
            continue   # 已是完整新版格式，跳过
        # 找同基线的图片文件（后缀不一定是 jpg）
        cands = [c for c in jf.parent.glob(jf.stem + ".*")
                 if c.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        if not cands:
            print(f"  无对应图片，跳过: {jf}")
            continue
        with Image.open(cands[0]) as im:
            w, h = im.size
        data["version"] = "2.4.2"
        data["imagePath"] = cands[0].name
        data["imageData"] = None      # 2.4.x 校验必需；null 表示不内嵌图片数据
        data["imageHeight"] = h
        data["imageWidth"] = w
        for s in data.get("shapes", []):
            s.setdefault("visible", True)   # 2.4.x shape 必需字段
        json.dump(data, open(jf, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        fixed += 1
    print(f"{d}: 修复 {fixed} 个 JSON")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="标注 JSON 所在目录（递归处理子目录）")
    args = ap.parse_args()
    fix_dir(args.dir)


if __name__ == "__main__":
    main()
