"""
推理结果可视化：在原图上画检测框 + 中文蜱种标签，输出 base64 JPEG
契约：draw_result(image_bytes, detections) -> str(base64)
detections 元素含 bbox / species_cn / species_slug / confidence / status / score
"""
import base64
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# 状态配色：confirmed 绿色（可信）、uncertain 橙红（建议人工确认）
STATUS_COLORS = {"confirmed": (34, 139, 34), "uncertain": (230, 126, 34)}
NO_TICK_TEXT = "未检测到蜱虫"


def load_font(size=22):
    """加载中文字体（Windows 黑体），失败回退默认字体"""
    for fp in ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"]:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def draw_result(image_bytes: bytes, detections: list[dict]) -> str:
    """画标注图并返回 base64 字符串（data:image/jpeg;base64,xxx 格式，前端可直接用）"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = load_font(22)

    if detections:
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            color = STATUS_COLORS.get(d.get("status", "confirmed"), (34, 139, 34))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            # 标签文字：中文名 (slug) 置信度% [状态]
            label = (f"{d['species_cn']} ({d['species_slug']}) "
                     f"{d['confidence']:.1f}% "
                     f"{'已确认' if d['status'] == 'confirmed' else '待确认'}")
            # 文字背景条：保证任何背景下都清晰可读
            w = int(draw.textlength(label, font=font)) + 12
            h = int(font.size * 1.4)
            ty = max(0, y1 - h - 4)  # 框上方放标签；贴顶时改放框内
            draw.rectangle([x1, ty, x1 + w, ty + h], fill=(0, 0, 0))
            draw.text((x1 + 6, ty + 3), label, fill=(255, 255, 255), font=font)
    else:
        # 未检出：原图顶部加提示条
        draw.rectangle([0, 0, img.width, 40], fill=(60, 60, 60))
        draw.text((12, 8), NO_TICK_TEXT, fill=(255, 255, 255), font=font)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
