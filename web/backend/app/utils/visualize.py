"""
推理结果可视化：在原图上画检测框 + 中文蜱种标签，输出 base64 JPEG
契约：draw_result(image_bytes, detections) -> str(base64)
detections 元素含 bbox / species_cn / species_slug / confidence / status / score
"""
import base64
import io
from pathlib import Path

from PIL import ImageDraw, ImageFont
from app.utils.images import load_rgb_image

# 状态配色：confirmed 绿色（可信）、uncertain 橙红（建议人工确认）
STATUS_COLORS = {"confirmed": (34, 139, 34), "uncertain": (230, 126, 34)}
NO_TICK_TEXT = "未检测到蜱虫"


def load_font(size=22):
    """加载中文字体：本地开发用 Windows 字体，容器内用 Noto CJK
    找不到任何中文字体时回退 PIL 默认字体（中文会变方块，仅作最后兜底）"""
    for fp in [
        # Linux 容器（Dockerfile 已安装 fonts-noto-cjk）
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        # Windows 本地开发
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
    ]:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def draw_label(draw, text, font, position, image_size, background):
    """Fit a label inside the image, wrapping Chinese names at character boundaries."""
    width, height = image_size
    padding = min(6, max(0, (width - 1) // 4))
    available = max(1, width - padding * 2)
    lines, line = [], ''
    for char in text:
        if line and draw.textlength(line + char, font=font) > available:
            lines.append(line)
            line = ''
        line += char
    if line:
        lines.append(line)
    line_height = max(1, font.getbbox('Ag')[3] + 4)
    lines = lines[:max(1, (height - padding * 2) // line_height)]
    box_width = min(width, int(max((draw.textlength(s, font=font) for s in lines), default=0)) + padding * 2)
    box_height = min(height, line_height * len(lines) + padding * 2)
    x = max(0, min(int(position[0]), width - box_width))
    y = max(0, min(int(position[1]) - box_height - 4, height - box_height))
    draw.rectangle((x, y, x + box_width - 1, y + box_height - 1), fill=background)
    for index, line in enumerate(lines):
        draw.text((x + padding, y + padding + index * line_height), line, fill='white', font=font)


def draw_result(image_bytes: bytes, detections: list[dict], top_label: str = None) -> str:
    """画标注图并返回 base64 字符串（data:image/jpeg;base64,xxx 格式，前端可直接用）
    top_label：无检测框时的顶部提示文字（如"这是蚜虫（非蜱虫）"），缺省用"未检测到蜱虫"
    """
    img = load_rgb_image(image_bytes)
    draw = ImageDraw.Draw(img)
    font = load_font(min(22, max(8, img.width // 12)))

    if detections:
        for index, d in enumerate(detections, 1):
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            color = STATUS_COLORS.get(d.get("status", "confirmed"), (34, 139, 34))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            confirmed = d.get('status') == 'confirmed'
            name = d['species_cn'] if confirmed else f"疑似{d['candidate_cn']}" if d.get('candidate_cn') else '蜱种待确认'
            label = f"#{index} {name} {d['confidence']:.1f}% {'已确认' if confirmed else '待确认'}"
            draw_label(draw, label, font, (x1, y1), img.size, (0, 0, 0))
    else:
        # 无检测框：原图顶部加提示条
        draw_label(draw, top_label or NO_TICK_TEXT, font, (0, 0), img.size, (60, 60, 60))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
