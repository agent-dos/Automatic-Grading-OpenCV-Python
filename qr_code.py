import cv2
import numpy as np
from pyzbar import pyzbar

QR_FONT_SCALE = 0.4
QR_FONT_THICKNESS = 1


def detect_qr_code(gray_paper, paper, dimensions):
    # QR Code decoding
    decoded_objects = pyzbar.decode(gray_paper)
    codes = [obj.data.decode('utf-8')
             for obj in decoded_objects] if decoded_objects else None

    # Annotate name from QR
    if codes is not None:
        cv2.putText(paper, codes[0],
                    (int(0.28 * dimensions[0]), int(0.125 * dimensions[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, QR_FONT_SCALE, (0, 0, 0), QR_FONT_THICKNESS)
    else:
        codes = [-1]
    return codes


def get_qr_roi_bounds(gray_image):
    """Returns bounding box of first QR code if found, else None."""
    decoded_objects = pyzbar.decode(gray_image)
    if decoded_objects:
        x, y, w, h = decoded_objects[0].rect
        return (x, y, w, h)
    return None


def get_qr_exclusion_mask(gray_image):
    decoded = pyzbar.decode(gray_image)
    mask = np.ones_like(gray_image, dtype=np.uint8) * 255

    if decoded:
        x, y, w, h = decoded[0].rect
        cv2.rectangle(mask, (x, y), (x + w, y + h), 0, thickness=-1)

        # Expand the exclusion slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.erode(mask, kernel, iterations=1)

    return mask
