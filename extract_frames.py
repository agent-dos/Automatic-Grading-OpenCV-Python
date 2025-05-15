import cv2
import os


def extract_frames(video_path):
    # Get video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("images", video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        frame_filename = os.path.join(
            output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {output_dir}")


# Example usage
if __name__ == "__main__":
    video_file = "videos\\test_video.mp4"  # Replace with your video path
    extract_frames(video_file)
