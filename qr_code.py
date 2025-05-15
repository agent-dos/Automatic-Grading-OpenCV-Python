import cv2
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