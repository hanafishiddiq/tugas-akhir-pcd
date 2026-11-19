"""Create the native-size annotation and the exact binary coverage mask."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vision import CLASSES, roi_mask


def angka(n):
    return f"{n:,}".replace(",", ".")


def persen(value):
    return f"{value:.2f}".replace(".", ",") + "%"


def font(size, bold=False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    path = Path("/System/Library/Fonts/Supplemental") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def simpan(rgb, result, roi, camera_name, out):
    summary = result["summary"]
    inside = [p for p in result["predictions"] if p["roi_membership"] == "inside"]
    road = roi_mask(roi)
    covered = np.zeros_like(road)
    for p in inside:
        x, y, right, bottom = p["bbox_xyxy"]
        covered[y:bottom, x:right] = True
    covered &= road
    if (int(covered.sum()), int(road.sum())) != (summary["covered_area_px"], summary["roi_area_px"]):
        raise ValueError("Luas arsiran berbeda dari ringkasan model.")
    Image.fromarray(covered.astype("uint8") * 255).save(out / "cakupan.png")

    tinted = rgb.copy()
    tinted[covered] = (0.74 * rgb[covered] + 0.26 * np.array([249, 185, 56])).astype(np.uint8)
    width, height = roi["width"], roi["height"]
    scale = width / 1280
    pad = round(17 * scale)
    canvas = Image.new("RGB", (width, height + round(138 * scale)), "#15282e")
    canvas.paste(Image.fromarray(tinted), (0, 0))
    draw = ImageDraw.Draw(canvas)
    label_font = font(round(14 * scale), True)
    polygon = [tuple(v) for v in roi["polygon_xy"]]
    draw.line(polygon + [polygon[0]], fill="#43eeef", width=round(3 * scale))
    for p in inside:
        x, y, right, bottom = p["bbox_xyxy"]
        draw.rectangle((x, y, right-1, bottom-1), outline="#ffd06e", width=round(2 * scale))
        label = p["class_name"]
        box = draw.textbbox((0, 0), label, font=label_font)
        label_width = box[2] - box[0] + 8
        label_height = box[3] - box[1] + 6
        left = min(x, width - label_width)
        top = max(0, y - label_height)
        draw.rectangle((left, top, left+label_width, top+label_height), fill="#19282b")
        draw.text((left+4, top+2-box[1]), label, fill="white", font=label_font)

    draw.text((pad, height + round(9*scale)), camera_name + "  /  v45", font=font(round(15*scale)), fill="#acd8d8")
    draw.text((pad, height + round(32*scale)), "Kepadatan visual  " + persen(summary["coverage_percent"]),
              font=font(round(34*scale), True), fill="white")
    counts = summary["count_by_class"]
    parts = [f"{c.capitalize()} {counts[c]}" for c in CLASSES]
    draw.text((pad, height + round(80*scale)), "     ".join(parts) + f"     |     Total {summary['vehicle_count']}",
              font=font(round(20*scale), True), fill="#ffcf79")
    area = f"{angka(summary['covered_area_px'])} / {angka(summary['roi_area_px'])} piksel"
    draw.text((pad, height + round(108*scale)), f"Garis biru muda: ROI jalan.   Arsiran emas: cakupan kotak yang dihitung ({area}).",
              font=font(round(16*scale)), fill="white")
    canvas.save(out / "hasil.jpg", quality=94, subsampling=0)
