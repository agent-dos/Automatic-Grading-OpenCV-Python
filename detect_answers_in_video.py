import cv2
import logging
import numpy as np
from detect_answer import answer_detector
from enhance_image import image_enhancer

# === Logger Setup ===
logging.basicConfig(
    filename='videos/video_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# === Configuration ===
VIDEO_PATH = "videos\\20250515_005333.mp4"
OUTPUT_PATH = "videos\\output_three_views.avi"
FRAME_SKIP = 1

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    logging.error("Cannot open video file.")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Force portrait orientation if landscape
if width > height:
    width, height = height, width  # swap dimensions

# Define writer for 3 views side-by-side
combined_width = width * 3
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps /
                      FRAME_SKIP, (combined_width, height))

frame_index = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
logging.info(f"Processing video: {VIDEO_PATH}")
logging.info(
    f"Total frames: {total_frames}, FPS: {fps}, Portrait resolution: {width}x{height}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Force portrait orientation
    if frame.shape[1] > frame.shape[0]:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    if frame_index % FRAME_SKIP == 0:
        logging.info(f"Processing frame {frame_index}")
        try:
            original = frame.copy()

            # Run answer detector
            enhanced, paper_with_answers, biggestContour, answers, codes = answer_detector(
                frame.copy())

            # Draw contour on enhanced image
            if biggestContour is not None:
                cv2.drawContours(
                    enhanced, [biggestContour], -1, (0, 255, 0), 2)

            # Annotate answers on paper
            if answers != -1 and answers != [-1]:
                for i, ans in enumerate(answers):
                    y_pos = 30 + i * 20
                    cv2.putText(paper_with_answers, f"{i+1}: {ans}", (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Resize all views to match height and width
            original_resized = cv2.resize(original, (width, height))
            enhanced_resized = cv2.resize(enhanced, (width, height))
            paper_resized = cv2.resize(paper_with_answers, (width, height))

            combined = np.hstack(
                (original_resized, enhanced_resized, paper_resized))
            out.write(combined)

        except Exception as e:
            logging.error(f"Error in frame {frame_index}: {e}")
            fallback = np.hstack((frame, frame, frame))
            out.write(fallback)

    else:
        out.write(np.hstack((frame, frame, frame)))

    frame_index += 1

cap.release()
out.release()
logging.info(f"✅ Three-view video saved to {OUTPUT_PATH}")
