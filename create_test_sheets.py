import cv2
import numpy as np
import qrcode
from PIL import ImageFont, ImageDraw, Image

# Customizable parameters
student_name = "Test Student"
year = ""
strand = ""
subject = ""
qr_data = student_name
num_items = 60
columns = 2  # number of columns to split items into
choices = ['A', 'B', 'C', 'D', 'E']
bubbles_per_column = num_items // columns

# Layout constants
sheet_width, sheet_height = 850, 1100
margin_x, margin_y = 80, 150
line_spacing = 28
bubble_spacing = 30
bubble_radius = 10
header_font_size = 20
field_font_size = 16

# Create blank white sheet
sheet = np.ones((sheet_height, sheet_width), dtype=np.uint8) * 255
sheet_img = Image.fromarray(sheet)
draw = ImageDraw.Draw(sheet_img)
font_header = ImageFont.truetype("arial.ttf", header_font_size)
font_field = ImageFont.truetype("arial.ttf", field_font_size)

# Header
draw.text((sheet_width // 2 - 80, 40),
          "FIRST SEMESTER", font=font_header, fill=0)
draw.text((40, 80), f"Name: {student_name}", font=font_field, fill=0)
draw.text((sheet_width // 2 - 80, 80),
          f"Year: {year}", font=font_field, fill=0)
draw.text((sheet_width // 2 + 80, 80),
          f"Strand: {strand}", font=font_field, fill=0)
draw.text((sheet_width - 200, 80),
          f"Subject: {subject}", font=font_field, fill=0)

# Draw corner markers
corner_size = 20
cv2.rectangle(sheet, (0, 0), (corner_size, corner_size), 0, -1)
cv2.rectangle(sheet, (sheet_width - corner_size, 0),
              (sheet_width, corner_size), 0, -1)
cv2.rectangle(sheet, (0, sheet_height - corner_size),
              (corner_size, sheet_height), 0, -1)
cv2.rectangle(sheet, (sheet_width - corner_size, sheet_height - corner_size),
              (sheet_width, sheet_height), 0, -1)

# Draw bubbles
start_x = margin_x
start_y = margin_y
for col in range(columns):
    for i in range(bubbles_per_column):
        item_no = i + 1 + col * bubbles_per_column
        y = start_y + i * line_spacing
        x = start_x + col * 350
        draw.text((x - 30, y - 10), f"{item_no}.", font=font_field, fill=0)
        for j, choice in enumerate(choices):
            cx = x + j * bubble_spacing
            cy = y
            cv2.circle(np.array(sheet_img), (cx, cy), bubble_radius, 0, 1)

# Generate QR code
qr_img = qrcode.make(qr_data)
qr_img = qr_img.resize((100, 100))
qr_arr = np.array(qr_img.convert("L"))
qr_x = sheet_width // 2 - 50
qr_y = start_y + (bubbles_per_column // 2) * line_spacing - 50
sheet[qr_y:qr_y + 100, qr_x:qr_x + 100] = qr_arr

# Convert final image to OpenCV format and save
final_sheet = np.array(sheet_img)
cv2.imwrite("images/custom_answer_sheet.png", final_sheet)
