import cv2
import numpy as np

# Marker templates (ensure path is correct)
marker_paths = [
    "markers/top_left.png",
    "markers/top_right.png",
    "markers/bottom_right.png",
    "markers/bottom_left.png"
]

# Expected output positions for 850x1202 A4 layout
EXPECTED_MARKER_POSITIONS = np.float32([
    [120, 180],     # top-left
    [730, 180],     # top-right
    [730, 1102],    # bottom-right
    [120, 1102]     # bottom-left
])


def detect_marker_positions(image_gray):
    """Original: Find marker centers via template matching (single scale)."""
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


def detect_marker_positions_multiscale(image_gray, scale_range=[0.9, 1.0, 1.1, 1.2]):
    """Augmented: Detect marker positions using multi-scale template matching."""
    positions = []

    for path in marker_paths:
        best_score = -np.inf
        best_center = None

        base_template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if base_template is None:
            raise FileNotFoundError(f"Marker template not found: {path}")

        for scale in scale_range:
            scaled_template = cv2.resize(
                base_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            result = cv2.matchTemplate(
                image_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                w, h = scaled_template.shape[::-1]
                best_center = (max_loc[0] + w // 2, max_loc[1] + h // 2)

        if best_center is not None:
            positions.append(best_center)
        else:
            raise RuntimeError(f"No matching found for marker: {path}")

    return np.float32(positions)


def try_contour_transform(image):
    """Try to warp image based on largest 4-point contour."""
    ratio = image.shape[1] / 500.0
    resized = cv2.resize(image, (0, 0), fx=1 / ratio, fy=1 / ratio)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 250, 300)

    contours, _ = cv2.findContours(
        edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            points = np.float32([pt[0] for pt in approx])
            mx = np.mean(points[:, 0])
            my = np.mean(points[:, 1])
            points = sorted(points, key=lambda x: (np.arctan2(
                x[0] - mx, x[1] - my) + 0.5 * np.pi) % (2 * np.pi), reverse=True)
            points = np.float32(points) * ratio
            desired_size = (850, 1202)
            dst_points = np.float32([[0, 0], [desired_size[0], 0],
                                     [desired_size[0], desired_size[1]], [0, desired_size[1]]])
            M = cv2.getPerspectiveTransform(points, dst_points)
            warped = cv2.warpPerspective(image, M, desired_size)
            return warped, approx
    return None, None


def transform_paper_image(image, use_multiscale=True):
    """Dual-stage transformation: contour first, then marker alignment."""
    original_image = image.copy()

    # Stage 1: Try contour-based normalization
    warped, largest_contour = try_contour_transform(original_image)
    base_for_marker = warped if warped is not None else original_image

    try:
        gray = cv2.cvtColor(base_for_marker, cv2.COLOR_BGR2GRAY)
        if use_multiscale:
            actual_positions = detect_marker_positions_multiscale(
                gray, scale_range=[0.9, 1.0, 1.1, 1.2])
        else:
            actual_positions = detect_marker_positions(gray)

        M = cv2.getPerspectiveTransform(
            actual_positions, EXPECTED_MARKER_POSITIONS)
        final_warped = cv2.warpPerspective(base_for_marker, M, (850, 1202))

        preview_img = base_for_marker.copy()
        for pt in actual_positions:
            cv2.circle(preview_img, (int(pt[0]), int(
                pt[1])), 10, (0, 0, 255), -1)

        return preview_img, final_warped, largest_contour, "dual_stage", actual_positions.tolist()

    except Exception as e:
        print(f"[Marker Transform Error] {e}")
        blank = np.ones((1202, 850, 3), dtype=np.uint8) * 255
        return base_for_marker, blank, largest_contour, "fallback", []
