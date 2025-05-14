import cv2
import numpy as np
import qrcode
from PIL import Image, ImageDraw, ImageFont
import os

# === Configuration ===
config = {
    "canvas_size": (850, 1100),
    "font_paths": {
        "header": "arial.ttf",
        "field": "arial.ttf"
    },
    "fonts": {
        "header_size": 24,
        "field_size": 18
    },
    "header_text": {
        "title": {"text": "FIRST SEMESTER", "position": (300, 40)},
        "name": {"label": "Name", "value": "Test Student", "position": (40, 90)},
        "year": {"label": "Year", "value": "", "position": (350, 90)},
        "strand": {"label": "Strand", "value": "", "position": (500, 90)},
        "subject": {"label": "Subject", "value": "", "position": (680, 90)},
    },
    "qr_code": {
        "data": "Test Student",
        "position": (375, 600),  # center
        "size": 100
    },
    "markers": {
        "top_left": "markers/top_left.png",
        "top_right": "markers/top_right.png",
        "bottom_left": "markers/bottom_left.png",
        "bottom_right": "markers/bottom_right.png"
    },
    "bubble_section": {
        "num_items": 60,
        "columns": 2,
        "choices": ['A', 'B', 'C', 'D', 'E'],
        "start": (80, 160),
        "spacing": {
            "line": 30,
            "bubble": 30,
        },
        "radius": 10
    },
    "output": {
        "filename": "custom_answer_sheet.png"
    }
}

# === Initialize Sheet ===
width, height = config["canvas_size"]
sheet = np.ones((height, width), dtype=np.uint8) * 255
sheet_pil = Image.fromarray(sheet).convert("RGB")
draw = ImageDraw.Draw(sheet_pil)
font_header = ImageFont.truetype(
    config["font_paths"]["header"], config["fonts"]["header_size"])
font_field = ImageFont.truetype(
    config["font_paths"]["field"], config["fonts"]["field_size"])

# === Draw Header With Underlines ===
for key, entry in config["header_text"].items():
    if key == "title":
        draw.text(entry["position"], entry["text"],
                  font=font_header, fill=(0, 0, 0))
    else:
        label = f"{entry['label']}:"
        label_pos = entry["position"]
        value_offset_x = 10 + draw.textlength(label, font=font_field)

        # Draw label
        draw.text(label_pos, label, font=font_field, fill=(0, 0, 0))

        # Draw underline (static length, adjustable)
        underline_start = (label_pos[0] + value_offset_x, label_pos[1] + 20)
        underline_end = (underline_start[0] + 180, underline_start[1])
        draw.line([underline_start, underline_end], fill=(0, 0, 0), width=1)

        # Draw value (optional, can be omitted to simulate user input area)
        draw.text((underline_start[0] + 5, label_pos[1]),
                  entry["value"], font=font_field, fill=(0, 0, 0))

# === Draw Corner Markers ===
sheet_cv = np.array(sheet_pil.convert("L"))
for pos, path in config["markers"].items():
    marker = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if marker is None:
        continue
    h, w = marker.shape
    if pos == "top_left":
        sheet_cv[0:h, 0:w] = marker
    elif pos == "top_right":
        sheet_cv[0:h, -w:] = marker
    elif pos == "bottom_left":
        sheet_cv[-h:, 0:w] = marker
    elif pos == "bottom_right":
        sheet_cv[-h:, -w:] = marker

# === Draw QR Code ===
qr = qrcode.make(config["qr_code"]["data"]).resize(
    (config["qr_code"]["size"], config["qr_code"]["size"]))
qr_arr = np.array(qr.convert("L"))
x, y = config["qr_code"]["position"]
sheet_cv[y:y + qr_arr.shape[0], x:x + qr_arr.shape[1]] = qr_arr

# === Draw Bubbles ===
bubble_cfg = config["bubble_section"]
bubbles_per_column = bubble_cfg["num_items"] // bubble_cfg["columns"]
sheet_rgb = cv2.cvtColor(sheet_cv, cv2.COLOR_GRAY2BGR)

for col in range(bubble_cfg["columns"]):
    for i in range(bubbles_per_column):
        item_no = i + 1 + col * bubbles_per_column
        y = bubble_cfg["start"][1] + i * bubble_cfg["spacing"]["line"]
        x = bubble_cfg["start"][0] + col * 400
        cv2.putText(sheet_rgb, f"{item_no}.", (x - 35, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        for j, _ in enumerate(bubble_cfg["choices"]):
            cx = x + j * bubble_cfg["spacing"]["bubble"]
            cy = y
            cv2.circle(sheet_rgb, (cx, cy), bubble_cfg["radius"], (0, 0, 0), 1)

# === Save Output ===
cv2.imwrite(config["output"]["filename"], sheet_rgb)
print(f"Saved to: {config['output']['filename']}")
