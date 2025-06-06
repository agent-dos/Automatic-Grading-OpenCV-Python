import cv2
import numpy as np
from PIL import Image
from pyzbar import pyzbar

from qr_code import detect_qr_code

# === Constants ===
epsilon = 10  # image error sensitivity
test_sensitivity_epsilon = 30  # bubble darkness error sensitivity
answer_choices = ['A', 'B', 'C', 'D', 'E', '?']

# Paper and bubble geometry (A4 rendered at 103 DPI, 850x1202 px)
scaling = [850.0, 1202.0]
radius = 10.0 / scaling[0]
spacing = [35.0 / scaling[0], 32.0 / scaling[1]]

# Marker tags
tags = [
    cv2.imread("markers/top_left.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("markers/top_right.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("markers/bottom_left.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("markers/bottom_right.png", cv2.IMREAD_GRAYSCALE)
]

# Sheet Configuration
NUM_COLUMNS = 2
NUM_ITEMS_PER_COLUMN = 30
NUM_CHOICES = 5

# Bounding box dimensions as multiple of radius
BBOX_SCALE_X = 1.8  # width multiplier
BBOX_SCALE_Y = 1.2  # height multiplier (typically same as diameter)

# Column anchor points (normalized relative to paper width and height)
COLUMN_ORIGINS = [
    [126 / scaling[0], 59 / scaling[1]],     # left column
    [618 / scaling[0], 59 / scaling[1]]      # right column
]

# Horizontal spacing between choices (normalized width units)
CHOICE_SPACING_X = 41 / scaling[0]  # adjust if needed

# Vertical spacing between items
ITEM_SPACING_Y = 37.2 / scaling[1]

ANSWER_FONT_SCALE = 0.6
ANSWER_FONT_THICKNESS = 2

# Variable aliasing to make it compatible
columns = COLUMN_ORIGINS


def ProcessPage(paper):
    answers = []
    gray_paper = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)

    # Locate markers
    corners = FindCorners(paper)
    if corners is None:
        return [-1], paper, [-1]

    dimensions = [corners[1][0] - corners[0][0], corners[2][1] - corners[0][1]]

    # Iterate over each column and question
    for k in range(NUM_COLUMNS):
        for i in range(NUM_ITEMS_PER_COLUMN):
            questions = []
            for j in range(NUM_CHOICES):
                # Bounding box of the bubble
                x_center = (columns[k][0] + j * CHOICE_SPACING_X) * \
                    dimensions[0] + corners[0][0]
                y_center = (columns[k][1] + i * ITEM_SPACING_Y) * \
                    dimensions[1] + corners[0][1]
                box_w = radius * BBOX_SCALE_X * dimensions[0]
                box_h = radius * BBOX_SCALE_Y * dimensions[1]

                x1 = int(x_center - box_w)
                x2 = int(x_center + box_w)
                y1 = int(y_center - box_h)
                y2 = int(y_center + box_h)

                cv2.rectangle(paper, (x1, y1), (x2, y2), (255, 0, 0), 1)
                questions.append(gray_paper[y1:y2, x1:x2])

            # Darkness analysis
            means = [np.mean(q) if q.size > 0 else 255 for q in questions]
            min_arg = np.argmin(means)
            min_val = means[min_arg]

            # Double bubble detection
            means[min_arg] = 255
            second_min = np.min(means)
            if second_min - min_val < test_sensitivity_epsilon:
                min_arg = NUM_CHOICES  # '?'

            # Annotate and save
            x_text = int((columns[k][0] - radius * 10) *
                         dimensions[0] + corners[0][0])
            y_text = int(y_center + 0.5 * radius * dimensions[1])
            cv2.putText(paper, answer_choices[min_arg], (x_text, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, ANSWER_FONT_SCALE, (0, 150, 0), ANSWER_FONT_THICKNESS)
            answers.append(answer_choices[min_arg])

    codes = detect_qr_code(gray_paper, paper, dimensions)

    return answers, paper, codes


def FindCorners(paper):
    gray_paper = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    ratio = len(paper[0]) / 816.0
    if ratio == 0:
        return -1

    corners = []
    for tag in tags:
        tag = cv2.resize(tag, (0, 0), fx=ratio, fy=ratio)
        conv = cv2.filter2D(np.float32(cv2.bitwise_not(
            gray_paper)), -1, np.float32(cv2.bitwise_not(tag)))
        max_pos = np.unravel_index(conv.argmax(), conv.shape)
        corners.append([max_pos[1], max_pos[0]])

    for corner in corners:
        cv2.rectangle(paper,
                      (corner[0] - int(ratio * 25),
                       corner[1] - int(ratio * 25)),
                      (corner[0] + int(ratio * 25),
                       corner[1] + int(ratio * 25)),
                      (0, 255, 0), 2)

    if corners[0][0] - corners[2][0] > epsilon or \
       corners[1][0] - corners[3][0] > epsilon or \
       corners[0][1] - corners[1][1] > epsilon or \
       corners[2][1] - corners[3][1] > epsilon:
        return None

    return corners
