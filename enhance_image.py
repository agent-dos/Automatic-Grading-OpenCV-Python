import cv2
import numpy as np


def image_enhancer(img, blur_ksize, block_size, C, morph_kernel_size):
    # Resize image to 1080px width while keeping aspect ratio
    target_width = 1080
    height, width = img.shape[:2]
    scale_ratio = target_width / width
    target_height = int(height * scale_ratio)
    img = cv2.resize(img, (target_width, target_height),
                     interpolation=cv2.INTER_AREA)

    # Grayscale + enhancement pipeline
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C
    )
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    enhanced = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
