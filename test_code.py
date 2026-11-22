"""
Shape Classifier — Live Webcam Inference

Loads the trained YOLO model and runs real-time shape detection
(circle, square, triangle) on your webcam feed.

Usage:
    python inference.py

Controls:
    Q — quit
"""

import cv2
import time
from ultralytics import YOLO

MODEL_PATH           = "models/best.pt"
CONFIDENCE_THRESHOLD = 0.5
CAMERA_INDEX         = 0   # change to 1, 2, etc. if you have multiple cameras


def main():
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: could not open camera index {CAMERA_INDEX}")
        return

    print("Starting live detection — press Q to quit")

    prev_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from camera")
            break

        # Run detection on the current frame
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        annotated_frame = results[0].plot()

        # FPS counter
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time

        cv2.putText(
            annotated_frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        # Detected shape count
        num_detections = len(results[0].boxes)
        cv2.putText(
            annotated_frame, f"Detections: {num_detections}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        cv2.imshow("Shape Classifier - Live", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
