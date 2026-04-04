import time

import cv2

from emotion_utils import EmotionRecognizer, FaceTracker


def main():
    recognizer = EmotionRecognizer()
    tracker = FaceTracker(recognizer=recognizer, smoothing_window=5, max_distance=120, max_misses=6)

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Could not access the webcam. Make sure a camera is connected and not used elsewhere.")

    window_name = "Emotion Detection - Real Time"
    previous_time = time.time()

    print("Starting webcam emotion detection. Press Q in the video window to quit.")

    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break

            frame = cv2.flip(frame, 1)
            gray_frame, faces = recognizer.detect_faces(frame)

            current_detections = []
            for bbox in faces:
                probabilities = recognizer.predict_probabilities(gray_frame, bbox)
                current_detections.append(
                    {
                        "bbox": bbox,
                        "probabilities": probabilities,
                    }
                )

            smoothed_predictions = tracker.update(current_detections)
            annotated_frame = recognizer.annotate_image(frame.copy(), smoothed_predictions)

            current_time = time.time()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            hud_text = f"Faces: {len(smoothed_predictions)} | FPS: {fps:.1f} | Press Q to exit"
            cv2.rectangle(annotated_frame, (12, 12), (470, 50), (12, 18, 35), -1)
            cv2.putText(
                annotated_frame,
                hud_text,
                (24, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (240, 248, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
