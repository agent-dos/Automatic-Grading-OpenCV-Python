import cv2
import logging
import numpy as np
from detect_answer import answer_detector
from enhance_image import image_enhancer
from datetime import datetime

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
USE_WEBCAM = False  # Set to False to use video file
VIDEO_PATH = "videos\\test_video.mp4"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = f"videos/output_three_views_{timestamp}.avi"
FRAME_SKIP = 1

# === Open capture source ===
if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
    logging.info("Using webcam input...")
else:
    cap = cv2.VideoCapture(VIDEO_PATH)
    logging.info(f"Using video file: {VIDEO_PATH}")

if not cap.isOpened():
    logging.error("Cannot open capture source.")
    exit(1)

# === Determine frame size and fps ===
ret, test_frame = cap.read()
if not ret:
    logging.error("Unable to read frame for resolution detection.")
    exit(1)

if test_frame.shape[1] > test_frame.shape[0]:
    test_frame = cv2.rotate(test_frame, cv2.ROTATE_90_CLOCKWISE)

frame_height, frame_width = test_frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # fallback for webcam

# === Define writer for side-by-side view ===
combined_width = frame_width * 3
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps /
                      FRAME_SKIP, (combined_width, frame_height))

frame_index = 0
logging.info(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}")

# === Main processing loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        if not USE_WEBCAM:
            break
        continue  # skip broken webcam frames

    if frame.shape[1] > frame.shape[0]:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    if frame_index % FRAME_SKIP == 0:
        try:
            original = frame.copy()

            # Detect answers and enhanced image
            enhanced, paper_with_answers, biggestContour, answers, codes = answer_detector(
                frame.copy())

            # Draw contour on enhanced
            if biggestContour is not None:
                cv2.drawContours(
                    enhanced, [biggestContour], -1, (0, 255, 0), 2)

            # Annotate answers
            if answers != -1 and answers != [-1]:
                for i, ans in enumerate(answers):
                    y = 30 + i * 20
                    cv2.putText(paper_with_answers, f"{i+1}: {ans}", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Resize all images to match
            original_resized = cv2.resize(
                original, (frame_width, frame_height))
            enhanced_resized = cv2.resize(
                enhanced, (frame_width, frame_height))
            paper_resized = cv2.resize(
                paper_with_answers, (frame_width, frame_height))

            combined = np.hstack(
                (original_resized, enhanced_resized, paper_resized))
            
            # Annotate frame count on image
            cv2.putText(combined, f"Frame: {frame_index}", (10, frame_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Write and optionally display
            out.write(combined)

            if USE_WEBCAM:
                cv2.imshow("Three-View OMR", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Log every 10 frames
            if frame_index % 10 == 0:
                logging.info(f"Rendered frame: {frame_index}")

        except Exception as e:
            logging.error(f"Error in frame {frame_index}: {e}")
            fallback = np.hstack((frame, frame, frame))
            out.write(fallback)

    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()
logging.info(f"✅ Output video saved to {OUTPUT_PATH}")
