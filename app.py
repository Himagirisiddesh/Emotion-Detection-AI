import base64
import binascii
from datetime import datetime
import logging
from pathlib import Path
from threading import Lock
import time

import cv2
import numpy as np
import pickle
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from emotion_utils import (
    BASE_DIR,
    EMOTION_EMOJIS,
    EmotionRecognizer,
    FaceTracker,
    ensure_runtime_directories,
    resolve_model_path,
)
from stress_text_utils import (
    TEXT_EMOJIS,
    TextStressPredictor,
    combine_stress_assessment,
    resolve_text_model_path,
)
from stress_history import StressHistory

app = Flask(__name__)
app.config["SECRET_KEY"] = "emotion-lens-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.logger.setLevel(logging.INFO)

UPLOAD_DIR = BASE_DIR / "static" / "uploads"
PROCESSED_DIR = BASE_DIR / "static" / "processed"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STREAM_TTL_SECONDS = 30

ensure_runtime_directories()

_recognizer = None
_text_predictor = None
_bio_model = None
_stream_sessions = {}
_stress_histories = {}
_stream_lock = Lock()


def get_recognizer():
    global _recognizer

    if _recognizer is None:
        _recognizer = EmotionRecognizer()
    return _recognizer


def get_text_predictor():
    global _text_predictor

    if _text_predictor is None:
        _text_predictor = TextStressPredictor()
    return _text_predictor


def get_bio_model():
    global _bio_model
    
    if _bio_model is None:
        model_path = Path("model/bio_stress_model.pkl")
        if not model_path.exists():
            raise FileNotFoundError("Bio model not found. Please train it first.")
            
        with open(model_path, "rb") as f:
            _bio_model = pickle.load(f)
        app.logger.info("Bio stress model loaded and cached.")
        
    return _bio_model


def serialize_text_prediction(prediction):
    return {
        "status": prediction["status"],
        "stress": prediction["stress_level"],
        "confidence": prediction["confidence"],
        "confidence_text": prediction["confidence_text"],
    }


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def build_base_context():
    face_model_path = resolve_model_path()
    text_model_path = resolve_text_model_path()
    return {
        "model_ready": face_model_path.exists(),
        "model_name": face_model_path.name if face_model_path.exists() else "No face model found",
        "text_model_ready": text_model_path.exists(),
        "text_model_name": text_model_path.name if text_model_path.exists() else "No text model found",
        "emotion_emojis": EMOTION_EMOJIS,
        "text_emojis": TEXT_EMOJIS,
        "static_version": int(time.time()),
    }


def save_uploaded_image(file_storage):
    filename = secure_filename(file_storage.filename or "")
    if not filename:
        raise ValueError("Please choose a valid image file.")

    if not allowed_file(filename):
        raise ValueError("Unsupported file type. Please upload JPG, JPEG, PNG, BMP, or WEBP.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    final_name = f"{timestamp}{Path(filename).suffix.lower()}"
    saved_path = UPLOAD_DIR / final_name
    file_storage.save(saved_path)

    image = cv2.imread(str(saved_path))
    if image is None:
        if saved_path.exists():
            saved_path.unlink()
        raise ValueError("The uploaded file is not a valid image.")

    return saved_path


def save_annotated_image(image, stem):
    annotated_name = f"{stem}_annotated.jpg"
    annotated_path = PROCESSED_DIR / annotated_name
    cv2.imwrite(str(annotated_path), image)
    return annotated_path


def decode_browser_frame(image_payload):
    if not image_payload:
        raise ValueError("No image frame was received from the browser.")

    encoded_section = image_payload.split(",", 1)[-1]

    try:
        image_bytes = base64.b64decode(encoded_section)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Frame payload is not valid base64 image data.") from exc

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not decode the webcam frame.")

    return frame


def prune_stream_sessions():
    now = time.time()
    expired_stream_ids = [
        stream_id
        for stream_id, session in _stream_sessions.items()
        if now - session["last_seen"] > STREAM_TTL_SECONDS
    ]

    for stream_id in expired_stream_ids:
        del _stream_sessions[stream_id]
        if stream_id in _stress_histories:
            del _stress_histories[stream_id]


def get_stream_tracker(stream_id):
    recognizer = get_recognizer()

    with _stream_lock:
        prune_stream_sessions()
        session = _stream_sessions.get(stream_id)

        if session is None:
            session = {
                "tracker": FaceTracker(
                    recognizer=recognizer,
                    smoothing_window=4,
                    max_distance=110,
                    max_misses=8,
                ),
                "last_seen": time.time(),
            }
            _stream_sessions[stream_id] = session
        else:
            session["last_seen"] = time.time()

        return session["tracker"]


def remove_stream_tracker(stream_id):
    with _stream_lock:
        if stream_id in _stream_sessions:
            del _stream_sessions[stream_id]
        if stream_id in _stress_histories:
            del _stress_histories[stream_id]


def maybe_predict_text(text_input):
    cleaned_input = (text_input or "").strip()
    if not cleaned_input:
        return None
    return get_text_predictor().predict(cleaned_input)


def build_multimodal_payload(face_analysis=None, text_prediction=None, bio_prediction=None):
    primary_face = None
    if face_analysis and face_analysis.get("success"):
        primary_face = face_analysis.get("primary")

    return combine_stress_assessment(primary_face, text_prediction, bio_prediction)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", **build_base_context())


@app.route("/predict", methods=["POST"])
def predict():
    context = build_base_context()
    uploaded_file = request.files.get("image")

    if uploaded_file is None or uploaded_file.filename == "":
        context["error_message"] = "Please upload an image before running prediction."
        return render_template("index.html", **context)

    try:
        saved_path = save_uploaded_image(uploaded_file)
        recognizer = get_recognizer()
        face_analysis = recognizer.analyze_image(saved_path)

        context["uploaded_image"] = f"uploads/{saved_path.name}"

        if face_analysis["success"]:
            annotated_path = save_annotated_image(face_analysis["annotated_image"], saved_path.stem)
            context["annotated_image"] = f"processed/{annotated_path.name}"
            context["result"] = {
                key: value
                for key, value in face_analysis.items()
                if key != "annotated_image"
            }
            context["status_message"] = face_analysis["message"]
        else:
            context["error_message"] = face_analysis["message"]

    except FileNotFoundError as exc:
        context["error_message"] = str(exc)
    except ValueError as exc:
        context["error_message"] = str(exc)
    except Exception as exc:
        context["error_message"] = f"Prediction failed: {exc}"

    return render_template("index.html", **context)


@app.route("/predict-frame", methods=["POST"])
def predict_frame():
    payload = request.get_json(silent=True) or {}
    stream_id = str(payload.get("stream_id") or "default")
    frame_payload = payload.get("image")
    text_input = payload.get("text", "")
    request_started_at = time.perf_counter()

    if not frame_payload:
        return jsonify({"success": False, "message": "No webcam frame was sent."}), 400

    try:
        frame = decode_browser_frame(frame_payload)
        face_analysis = get_recognizer().analyze_frame(
            frame,
            tracker=get_stream_tracker(stream_id),
            mode="webcam",
        )
        text_prediction = maybe_predict_text(text_input)
        multimodal_result = build_multimodal_payload(face_analysis, text_prediction, None)

        response_payload = dict(face_analysis)
        response_payload["text_result"] = (
            serialize_text_prediction(text_prediction) if text_prediction is not None else None
        )
        response_payload["multimodal_result"] = multimodal_result
        
        if multimodal_result is not None:
            if stream_id not in _stress_histories:
                _stress_histories[stream_id] = StressHistory(max_window=30)
            
            confidences = []
            if face_analysis and face_analysis.get("success") and face_analysis.get("primary"):
                confidences.append(face_analysis["primary"].get("confidence", 0.0))
            if text_prediction and "confidence" in text_prediction:
                confidences.append(text_prediction["confidence"])
                
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            _stress_histories[stream_id].add(multimodal_result["score"], avg_confidence)
            response_payload["stress_history"] = _stress_histories[stream_id].summary()
        else:
            response_payload["stress_history"] = None
            
        response_payload["server_latency_ms"] = round((time.perf_counter() - request_started_at) * 1000, 1)
        return jsonify(response_payload)
    except FileNotFoundError as exc:
        return jsonify({"success": False, "message": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "message": f"Live prediction failed: {exc}"}), 500


@app.route("/predict-text", methods=["POST"])
def predict_text():
    try:
        data = request.get_json(silent=True) or {}
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field in request body."}), 400

        text_input = str(data["text"]).strip()
        if not text_input:
            return jsonify({"error": "Text input cannot be empty."}), 400

        prediction = get_text_predictor().predict(text_input)
        return jsonify(serialize_text_prediction(prediction))

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Text prediction failed")
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.route("/predict-bio", methods=["POST"])
def predict_bio():
    try:
        bio_model = get_bio_model()
            
        data = request.get_json(silent=True) or {}
        if not data:
            return jsonify({"error": "Missing JSON body."}), 400
            
        expected_fields = {"heart_rate", "eda", "respiration", "temperature"}
        missing_fields = expected_fields - set(data.keys())
        if missing_fields:
            app.logger.warning("predict-bio: missing fields %s \u2014 using defaults. Prediction may be unreliable.", sorted(missing_fields))
            
        # Extract features
        hr = float(data.get("heart_rate", 70))
        eda = float(data.get("eda", 3.0))
        resp = float(data.get("respiration", 15))
        temp = float(data.get("temperature", 36.5))
        
        print(f"[Bio Model] Received values -> HR: {hr}, EDA: {eda}, Resp: {resp}, Temp: {temp}")
        
        features = np.array([[hr, eda, resp, temp]])
        prediction = bio_model.predict(features)[0]
        probs = bio_model.predict_proba(features)[0]
        
        # Calculate confidence using distance from decision boundary
        prob_stress = probs[1] if probs.size > 1 else (1.0 if prediction == 1 else 0.0)
        confidence = abs(prob_stress - 0.5) * 2
        confidence = min(confidence, 0.95)
        
        if prediction == 1:
            status = "Stressed"
            message = "You seem stressed. Try taking a short break and relax."
        else:
            status = "Not Stressed"
            message = "You are doing well. Keep maintaining your routine."
            
        final_conf = round(confidence * 100, 2)
        print(f"[Bio Model] Prediction -> Status: {status}, Confidence: {final_conf}%")
        
        return jsonify({
            "status": status,
            "confidence": final_conf,
            "message": message
        })
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        app.logger.exception("Bio prediction failed")
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
