import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
PROCESSED_DIR = STATIC_DIR / "processed"
METADATA_PATH = MODEL_DIR / "emotion_metadata.json"

DEFAULT_MODEL_CANDIDATES = [
    MODEL_DIR / "emotion_model.keras",
    MODEL_DIR / "emotion_model.h5",
    BASE_DIR / "emotion_model.h5",
]

CANONICAL_LABELS = {
    "Angry": "Angry",
    "Fear": "Fear",
    "Happy": "Happy",
    "Sad": "Sad",
    "Suprise": "Surprise",
    "Surprise": "Surprise",
    "Neutral": "Neutral",
}

DEFAULT_LABELS = ["Angry", "Fear", "Happy", "Sad", "Surprise"]

EMOTION_EMOJIS = {
    "Angry": "\U0001F621",
    "Fear": "\U0001F628",
    "Happy": "\U0001F604",
    "Sad": "\U0001F622",
    "Surprise": "\U0001F632",
    "Neutral": "\U0001F610",
    "Not Sure": "\U0001F914",
}


def ensure_runtime_directories():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def canonicalize_label(label):
    if not label:
        return "Unknown"
    return CANONICAL_LABELS.get(label.strip(), label.strip().title())


def load_metadata(metadata_path=METADATA_PATH):
    metadata = {
        "labels": list(DEFAULT_LABELS),
        "input_size": [48, 48],
        "image_confidence_threshold": 0.35,
        "image_confidence_margin": 0.00,
        "confidence_threshold": 0.40,
        "confidence_margin": 0.02,
    }

    if Path(metadata_path).exists():
        with Path(metadata_path).open("r", encoding="utf-8") as file:
            loaded = json.load(file)
        metadata.update(loaded)

    labels = metadata.get("labels") or metadata.get("raw_labels") or list(DEFAULT_LABELS)
    metadata["labels"] = [canonicalize_label(label) for label in labels]
    
    # If loaded labels list is empty or seems wrong, fall back to DEFAULT_LABELS
    if not metadata["labels"] or len(metadata["labels"]) == 0:
        metadata["labels"] = list(DEFAULT_LABELS)
        
    metadata["input_size"] = tuple(metadata.get("input_size", [48, 48]))
    metadata["image_confidence_threshold"] = float(metadata.get("image_confidence_threshold", 0.35))
    metadata["image_confidence_margin"] = float(metadata.get("image_confidence_margin", 0.00))
    metadata["confidence_threshold"] = float(metadata.get("confidence_threshold", 0.40))
    metadata["confidence_margin"] = float(metadata.get("confidence_margin", 0.02))
    return metadata


def resolve_model_path(explicit_model_path=None, metadata_path=METADATA_PATH):
    candidates = []

    if explicit_model_path:
        explicit_path = Path(explicit_model_path)
        candidates.append(explicit_path if explicit_path.is_absolute() else BASE_DIR / explicit_path)

    if Path(metadata_path).exists():
        metadata = load_metadata(metadata_path)
        metadata_model = metadata.get("model_path")
        if metadata_model:
            metadata_model_path = Path(metadata_model)
            candidates.append(
                metadata_model_path
                if metadata_model_path.is_absolute()
                else BASE_DIR / metadata_model_path
            )

    candidates.extend(DEFAULT_MODEL_CANDIDATES)

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if str(resolved) not in seen:
            unique_candidates.append(candidate)
            seen.add(str(resolved))

    existing_candidates = [candidate for candidate in unique_candidates if candidate.exists()]
    if existing_candidates:
        return max(existing_candidates, key=lambda item: item.stat().st_mtime)

    return unique_candidates[0]


def bbox_center(bbox):
    x, y, width, height = bbox
    return x + (width // 2), y + (height // 2)


@dataclass
class FaceTrack:
    track_id: int
    bbox: tuple
    center: tuple
    history: deque = field(default_factory=deque)
    misses: int = 0
    candidate_label: str = ""
    candidate_count: int = 0
    locked_label: str = ""


class FaceTracker:
    def __init__(self, recognizer, smoothing_window=7, max_distance=110, max_misses=12):
        self.recognizer = recognizer
        self.smoothing_window = max(3, smoothing_window)
        self.max_distance = max_distance
        self.max_misses = max_misses
        self.tracks = {}
        self.next_track_id = 1

    def update(self, detections):
        if not detections:
            stale_track_ids = []
            for track_id, track in self.tracks.items():
                track.misses += 1
                if track.misses > self.max_misses:
                    stale_track_ids.append(track_id)

            for track_id in stale_track_ids:
                del self.tracks[track_id]

            return []

        detection_centers = [bbox_center(detection["bbox"]) for detection in detections]
        candidate_matches = []

        for track_id, track in self.tracks.items():
            for detection_index, center in enumerate(detection_centers):
                distance = math.dist(track.center, center)
                if distance <= self.max_distance:
                    candidate_matches.append((distance, track_id, detection_index))

        assigned_tracks = set()
        assigned_detections = set()
        matches = {}

        for _, track_id, detection_index in sorted(candidate_matches, key=lambda item: item[0]):
            if track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            matches[detection_index] = track_id
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)

        seen_track_ids = set()
        smoothed_predictions = []

        for detection_index, detection in enumerate(detections):
            bbox = tuple(detection["bbox"])
            center = detection_centers[detection_index]
            track_id = matches.get(detection_index)

            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = FaceTrack(
                    track_id=track_id,
                    bbox=bbox,
                    center=center,
                    history=deque(maxlen=self.smoothing_window),
                )

            track = self.tracks[track_id]
            track.bbox = bbox
            track.center = center
            track.misses = 0
            track.history.append(np.asarray(detection["probabilities"], dtype=np.float32))

            averaged_probabilities = np.mean(np.stack(track.history, axis=0), axis=0)
            prediction = self.recognizer.build_prediction(
                averaged_probabilities,
                bbox=bbox,
                track_id=track_id,
                mode="webcam",
            )
            stabilized_prediction = self.recognizer.stabilize_webcam_prediction(track, prediction)
            smoothed_predictions.append(stabilized_prediction)
            seen_track_ids.add(track_id)

        stale_track_ids = []
        for track_id, track in self.tracks.items():
            if track_id not in seen_track_ids:
                track.misses += 1
                if track.misses > self.max_misses:
                    stale_track_ids.append(track_id)

        for track_id in stale_track_ids:
            del self.tracks[track_id]

        return sorted(
            smoothed_predictions,
            key=lambda item: item["bbox"][2] * item["bbox"][3],
            reverse=True,
        )


class EmotionRecognizer:
    def __init__(self, model_path=None, metadata_path=METADATA_PATH, confidence_threshold=None):
        ensure_runtime_directories()
        self.metadata = load_metadata(metadata_path)
        self.labels = self.metadata["labels"]
        # Validate that label count is sane — if it is 0 or suspiciously large, reset to defaults
        if not self.labels or len(self.labels) > 20:
            self.labels = list(DEFAULT_LABELS)
        self.input_size = tuple(self.metadata["input_size"])
        self.image_confidence_threshold = float(self.metadata.get("image_confidence_threshold", 0.35))
        self.image_confidence_margin = float(self.metadata.get("image_confidence_margin", 0.00))
        self.confidence_threshold = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else float(self.metadata["confidence_threshold"])
        )
        self.confidence_margin = float(self.metadata.get("confidence_margin", 0.02))
        self.webcam_confidence_threshold = float(
            self.metadata.get("webcam_confidence_threshold", 0.35)
        )
        self.webcam_confidence_margin = float(
            self.metadata.get("webcam_confidence_margin", 0.02)
        )
        self.webcam_promotion_threshold = float(
            self.metadata.get("webcam_promotion_threshold", 0.30)
        )
        self.webcam_stability_frames = int(
            self.metadata.get("webcam_stability_frames", 3)
        )
        self.model_path = resolve_model_path(model_path, metadata_path)
        self.model = None
        self.model_handles_rescaling = False
        self.detection_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.face_clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(4, 4))
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.alt_face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        )

        if self.face_detector.empty():
            raise FileNotFoundError("OpenCV Haarcascade could not be loaded.")

    def load(self):
        if self.model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    "No trained emotion model was found. Train the model first with "
                    "`python train_model.py`."
                )
            self.model = load_model(self.model_path, compile=False)
            self.model_handles_rescaling = any(
                layer.__class__.__name__ == "Rescaling"
                for layer in self.model.layers
            )
            # Old sequential models have no Rescaling layer — preprocessing divides by 255 manually.
        return self.model

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe_gray = self.detection_clahe.apply(gray)
        softened_clahe = cv2.GaussianBlur(clahe_gray, (3, 3), 0)
        min_face = max(32, int(min(gray.shape[:2]) * 0.12))
        detection_attempts = [
            (softened_clahe, 1.12, 6, min_face),
            (clahe_gray, 1.1, 5, max(28, min_face - 8)),
            (gray, 1.15, 5, min_face),
            (gray, 1.1, 4, max(24, min_face - 12)),
        ]

        detectors = [self.face_detector]
        if self.alt_face_detector is not None and not self.alt_face_detector.empty(): # Guard with is not None for compatibility across OpenCV builds
            detectors.append(self.alt_face_detector)

        faces = ()
        for detector in detectors:
            for image_for_detection, scale_factor, min_neighbors, min_size in detection_attempts:
                faces = detector.detectMultiScale(
                    image_for_detection,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=(min_size, min_size),
                )
                if len(faces) > 0:
                    break
            if len(faces) > 0:
                break

        ordered_faces = sorted(faces, key=lambda item: item[2] * item[3], reverse=True)
        return gray, [tuple(int(value) for value in face) for face in ordered_faces]

    def preprocess_face(self, gray_image, bbox):
        x, y, width, height = bbox
        padding = int(min(width, height) * 0.12)

        x0 = max(x - padding, 0)
        y0 = max(y - padding, 0)
        x1 = min(x + width + padding, gray_image.shape[1])
        y1 = min(y + height + padding, gray_image.shape[0])

        face = gray_image[y0:y1, x0:x1]
        if face.size == 0:
            raise ValueError("Detected face region is empty.")

        face = cv2.resize(face, self.input_size, interpolation=cv2.INTER_AREA)
        normalized_face = cv2.normalize(face, None, 0, 255, cv2.NORM_MINMAX)
        clahe_face = self.face_clahe.apply(normalized_face)
        enhanced_face = cv2.addWeighted(normalized_face, 0.6, clahe_face, 0.4, 0)
        enhanced_face = cv2.GaussianBlur(enhanced_face, (3, 3), 0)

        variants = [
            normalized_face.astype("float32"),
            enhanced_face.astype("float32"),
        ]

        batch = np.stack(variants, axis=0)[..., np.newaxis]
        if not self.model_handles_rescaling:
            batch /= 255.0
        return batch

    def predict_probabilities(self, gray_image, bbox):
        model = self.load()
        prepared_face = self.preprocess_face(gray_image, bbox)
        predictions = model.predict(prepared_face, verbose=0)
        probabilities = np.mean(predictions, axis=0)
        return np.asarray(probabilities, dtype=np.float32)

    def build_prediction(self, probabilities, bbox, track_id=None, mode="image"):
        if len(self.labels) != len(probabilities):
            # Label count mismatch — use DEFAULT_LABELS trimmed or padded to match
            fallback = list(DEFAULT_LABELS)
            if len(fallback) >= len(probabilities):
                labels = fallback[:len(probabilities)]
            else:
                labels = fallback + [f"Class {i}" for i in range(len(fallback), len(probabilities))]
        else:
            labels = self.labels

        probabilities = np.asarray(probabilities, dtype=np.float32)
        best_index = int(np.argmax(probabilities))
        label = labels[best_index]
        confidence = float(probabilities[best_index])
        sorted_indices = np.argsort(probabilities)[::-1]
        runner_up_confidence = float(probabilities[sorted_indices[1]]) if len(probabilities) > 1 else 0.0
        confidence_gap = confidence - runner_up_confidence

        if mode == "webcam":
            threshold = self.confidence_threshold
            margin = self.confidence_margin
        else:
            threshold = self.image_confidence_threshold
            margin = self.image_confidence_margin

        is_sure = (
            confidence >= threshold
            and confidence_gap >= margin
        )
        display_label = label if is_sure else "Not Sure"
        display_emoji = EMOTION_EMOJIS.get(display_label, "\U0001F642")
        result_text = f"{display_label} ({confidence * 100:.1f}%)"
        result_display = f"{display_emoji} {result_text}"

        return {
            "track_id": track_id,
            "bbox": [int(value) for value in bbox],
            "label": label,
            "display_label": display_label,
            "confidence": confidence,
            "confidence_percentage": round(confidence * 100, 2),
            "confidence_text": f"{confidence * 100:.1f}%",
            "confidence_gap": round(confidence_gap * 100, 2),
            "confidence_gap_ratio": confidence_gap,
            "mode": mode,
            "is_sure": is_sure,
            "is_stable": False,
            "emoji": display_emoji,
            "result_text": result_text,
            "result_display": result_display,
            "overlay_text": result_text,
        }

    def _promote_prediction(self, prediction, label):
        promoted_prediction = dict(prediction)
        display_emoji = EMOTION_EMOJIS.get(label, "\U0001F642")
        result_text = f"{label} ({prediction['confidence'] * 100:.1f}%)"
        promoted_prediction["display_label"] = label
        promoted_prediction["emoji"] = display_emoji
        promoted_prediction["result_text"] = result_text
        promoted_prediction["result_display"] = f"{display_emoji} {result_text}"
        promoted_prediction["overlay_text"] = result_text
        promoted_prediction["is_stable"] = True
        return promoted_prediction

    def stabilize_webcam_prediction(self, track, prediction):
        label = prediction["label"]
        confidence = prediction["confidence"]

        if track.candidate_label == label:
            track.candidate_count += 1
        else:
            track.candidate_label = label
            track.candidate_count = 1

        # Immediately promote if model is confident enough
        if confidence >= self.webcam_confidence_threshold:
            track.locked_label = label
            return self._promote_prediction(prediction, label)

        # Promote if same label has been seen consistently for stability_frames
        if track.candidate_count >= self.webcam_stability_frames:
            track.locked_label = label
            return self._promote_prediction(prediction, label)

        # Promote if this matches the previously locked label and is above promotion threshold
        if track.locked_label == label and confidence >= self.webcam_promotion_threshold:
            return self._promote_prediction(prediction, label)

        # Promote if model says is_sure
        if prediction["is_sure"]:
            track.locked_label = label
            return self._promote_prediction(prediction, label)

        return prediction

    def annotate_image(self, image, detections):
        for detection in detections:
            x, y, width, height = detection["bbox"]
            color = (45, 212, 191) if (detection["is_sure"] or detection.get("is_stable")) else (148, 163, 184)
            label_text = detection.get("overlay_text") or detection["result_text"]

            cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)

            text_size, baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                2,
            )
            text_x = x
            text_y = max(28, y - 10)
            overlay_top = text_y - text_size[1] - 10
            overlay_bottom = text_y + baseline - 6

            cv2.rectangle(
                image,
                (text_x, overlay_top),
                (text_x + text_size[0] + 16, overlay_bottom),
                color,
                thickness=-1,
            )
            cv2.putText(
                image,
                label_text,
                (text_x + 8, text_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (9, 14, 25),
                2,
                cv2.LINE_AA,
            )

        return image

    def analyze_frame(self, image, tracker=None, mode="image"):
        if image is None or not hasattr(image, "shape") or image.size == 0:
            raise ValueError("Invalid image frame received for emotion analysis.")

        frame_height, frame_width = image.shape[:2]
        gray_image, faces = self.detect_faces(image)

        if not faces:
            return {
                "success": False,
                "message": "No face detected in the current frame.",
                "faces_detected": 0,
                "detections": [],
                "primary": None,
                "frame_width": int(frame_width),
                "frame_height": int(frame_height),
            }

        if tracker is not None:
            frame_candidates = []
            for bbox in faces:
                probabilities = self.predict_probabilities(gray_image, bbox)
                frame_candidates.append(
                    {
                        "bbox": bbox,
                        "probabilities": probabilities,
                    }
                )
            detections = tracker.update(frame_candidates)
        else:
            detections = []
            for bbox in faces:
                probabilities = self.predict_probabilities(gray_image, bbox)
                detections.append(self.build_prediction(probabilities, bbox, mode=mode))

        primary = detections[0]
        message = (
            f"Detected {len(detections)} face(s). Primary emotion: "
            f"{primary['result_text']}."
        )

        return {
            "success": True,
            "message": message,
            "faces_detected": len(detections),
            "primary": primary,
            "detections": detections,
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
        }

    def analyze_cropped_face_image(self, image, mode="full_image"):
        if image is None or not hasattr(image, "shape") or image.size == 0:
            raise ValueError("Invalid image received for cropped-face analysis.")

        frame_height, frame_width = image.shape[:2]
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bbox = (
            0,
            0,
            max(1, frame_width),
            max(1, frame_height),
        )
        probabilities = self.predict_probabilities(gray_image, bbox)
        detection = self.build_prediction(probabilities, bbox, mode="image")
        detection["detection_mode"] = mode

        message = (
            f"No face detected in the image. Running full-image emotion analysis as fallback. "
            f"Primary emotion: {detection['result_text']}."
        )

        return {
            "success": True,
            "message": message,
            "faces_detected": 1,
            "primary": detection,
            "detections": [detection],
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
        }

    def analyze_image(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Invalid image. OpenCV could not read the uploaded file.")

        height, width = image.shape[:2]
        if max(height, width) <= 96:
            analysis = self.analyze_cropped_face_image(image, mode="small_image_fallback")
            analysis["annotated_image"] = self.annotate_image(image.copy(), analysis["detections"])
            return analysis

        analysis = self.analyze_frame(image)
        if not analysis["success"]:
            fallback_analysis = self.analyze_cropped_face_image(image, mode="no_face_fallback")
            fallback_analysis["annotated_image"] = self.annotate_image(
                image.copy(),
                fallback_analysis["detections"],
            )
            return fallback_analysis

        analysis["annotated_image"] = self.annotate_image(image.copy(), analysis["detections"])
        return analysis
