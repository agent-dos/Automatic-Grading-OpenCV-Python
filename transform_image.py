import cv2
import numpy as np

# Marker templates (ensure path is correct or relative to execution context)
marker_paths = [
    "markers/top_left.png",
    "markers/top_right.png",
    "markers/bottom_right.png",
    "markers/bottom_left.png"
]

# Static expected marker locations on the 850x1202 sheet
EXPECTED_MARKER_POSITIONS = np.float32([
    [120, 180],     # top-left
    [730, 180],     # top-right
    [730, 1102],    # bottom-right
    [120, 1102]     # bottom-left
])


def detect_marker_positions(image_gray):
    """Finds actual marker positions via template matching."""
    positions = []

    for path in marker_paths:
        template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(f"Marker template not found: {path}")

        result = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        w, h = template.shape[::-1]
        center = (max_loc[0] + w // 2, max_loc[1] + h // 2)
        positions.append(center)

    return np.float32(positions)


def transform_paper_image(image):
    """Performs static marker-based transformation to 850x1202 pixels."""
    original_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    try:
        actual_positions = detect_marker_positions(gray)
        # Warp using detected marker centers and expected layout
        M = cv2.getPerspectiveTransform(
            actual_positions, EXPECTED_MARKER_POSITIONS)
        warped = cv2.warpPerspective(original_image, M, (850, 1202))

        # For preview, draw detected points
        preview_img = original_image.copy()
        for pt in actual_positions:
            cv2.circle(preview_img, (int(pt[0]), int(
                pt[1])), 10, (0, 0, 255), -1)

        return preview_img, warped, None, "static_marker", actual_positions.tolist()

    except Exception as e:
        print(f"[Static Marker Transform Error] {e}")
        blank = np.ones((1202, 850, 3), dtype=np.uint8) * 255
        return image, blank, None, "fallback", []
