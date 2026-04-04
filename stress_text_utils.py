import json
import logging
import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TEXT_DATA_DIR = BASE_DIR / "stress"
DEFAULT_TEXT_DATASET_PATH = TEXT_DATA_DIR / "Stress.csv"
TEXT_MODEL_PATH = MODEL_DIR / "text_stress_model.pkl"
TEXT_METADATA_PATH = MODEL_DIR / "text_stress_metadata.json"
SUPPORTED_VECTORIZERS = (TfidfVectorizer, CountVectorizer)
INVALID_MODEL_ERROR = "Invalid model: Only binary Stress.csv model allowed"
logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

LABEL_TO_NAME = {
    0: "Not Stressed",
    1: "Stressed",
}

PREDICTION_TO_STRESS_LEVEL = {
    0: "LOW",
    1: "HIGH",
}

TEXT_EMOJIS = {
    "Not Stressed": "\U0001F642",
    "Stressed": "\U0001F61F",
}

FACE_STRESS_SCORES = {
    "Happy": 0,
    "Neutral": 0,
    "Sad": 1,
    "Angry": 2,
    "Fear": 2,
    "Surprise": 1,
    "Not Sure": 1,
}

TEXT_STRESS_SCORES = {
    0: 0,
    1: 2,
}

STRESS_LEVEL_EMOJIS = {
    "LOW": "\U0001F7E2",
    "HIGH": "\U0001F534",
    "Low": "\U0001F7E2",
    "Medium": "\U0001F7E0",
    "High": "\U0001F534",
    "Unknown": "\U0001F914",
}

CUSTOM_STOP_WORDS = {
    "amp",
    "im",
    "ive",
    "id",
    "youre",
    "theyre",
    "thats",
    "theres",
}

STOP_WORDS = (set(ENGLISH_STOP_WORDS) - {"no", "not", "nor"}) | CUSTOM_STOP_WORDS

SLANG_REPLACEMENTS = {
    "i'm": "i am",
    "im": "i am",
    "i've": "i have",
    "ive": "i have",
    "don't": "do not",
    "dont": "do not",
    "can't": "can not",
    "cant": "can not",
    "won't": "will not",
    "wont": "will not",
    "didn't": "did not",
    "didnt": "did not",
    "isn't": "is not",
    "isnt": "is not",
    "aren't": "are not",
    "arent": "are not",
    "wasn't": "was not",
    "wasnt": "was not",
    "weren't": "were not",
    "werent": "were not",
    "idk": "i do not know",
    "u": "you",
    "ur": "your",
    "rn": "right now",
    "bc": "because",
    "bcz": "because",
    "pls": "please",
    "plz": "please",
}

URGENT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(can(?:not|'t)\s+take\s+it\s+anymore)\b",
        r"\b(breaking\s+down)\b",
        r"\b(having\s+a\s+panic\s+attack)\b",
        r"\b(feel\s+like\s+giving\s+up)\b",
        r"\b(want\s+to\s+disappear)\b",
        r"\b(want\s+to\s+die)\b",
        r"\b(hurt\s+myself)\b",
        r"\b(harm\s+myself)\b",
    ]
]

CONTEXT_KEYWORDS = {
    "work": {"work", "job", "office", "boss", "deadline", "project", "meeting", "manager"},
    "study": {"exam", "study", "school", "college", "university", "assignment", "grades", "test"},
    "sleep": {"sleep", "tired", "insomnia", "rest", "exhausted", "fatigue"},
    "family": {"family", "mother", "mom", "father", "dad", "parents", "home", "sibling"},
    "relationship": {"relationship", "partner", "boyfriend", "girlfriend", "marriage", "breakup"},
    "money": {"money", "debt", "rent", "loan", "salary", "bills", "financial"},
    "health": {"health", "doctor", "hospital", "sick", "pain", "illness", "anxiety"},
}


def ensure_text_model_directory():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


class TextModelValidationError(ValueError):
    """Raised when the saved text model does not match the binary Stress.csv format."""


def _replace_slang(text):
    normalized = f" {text} "
    for source, target in sorted(SLANG_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = rf"(?<![a-z]){re.escape(source)}(?![a-z])"
        normalized = re.sub(pattern, f" {target} ", normalized)
    return normalized.strip()


def basic_clean_text(text):
    if not isinstance(text, str):
        text = ""

    text = text.strip().lower()
    if not text:
        return ""

    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    text = _replace_slang(text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text):
    if not text:
        return []
    return re.findall(r"[a-z']+", text)


def normalize_text(text):
    cleaned = basic_clean_text(text)
    if not cleaned:
        return ""

    raw_tokens = tokenize_text(cleaned)
    filtered_tokens = [token for token in raw_tokens if token not in STOP_WORDS]

    if not filtered_tokens:
        filtered_tokens = raw_tokens

    return " ".join(filtered_tokens)


def resolve_text_model_path(explicit_path=None):
    ensure_text_model_directory()
    model_path = TEXT_MODEL_PATH.resolve()

    if explicit_path is not None:
        requested_path = Path(explicit_path)
        if not requested_path.is_absolute():
            requested_path = (BASE_DIR / requested_path).resolve()
        if requested_path != model_path:
            raise TextModelValidationError(INVALID_MODEL_ERROR)

    if model_path.suffix.lower() != ".pkl":
        raise TextModelValidationError(INVALID_MODEL_ERROR)

    return model_path


def load_text_metadata(metadata_path=TEXT_METADATA_PATH):
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Text model metadata is missing. Retrain the binary Stress.csv model with `python train_text_model.py`."
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    dataset_name = Path(str(metadata.get("dataset_path", ""))).name.lower()
    if dataset_name != DEFAULT_TEXT_DATASET_PATH.name.lower():
        raise TextModelValidationError(INVALID_MODEL_ERROR)

    model_name = Path(str(metadata.get("model_path", ""))).name
    if model_name != TEXT_MODEL_PATH.name:
        raise TextModelValidationError(INVALID_MODEL_ERROR)

    raw_labels = metadata.get("labels", [])
    try:
        labels = [int(label) for label in raw_labels]
    except (TypeError, ValueError) as exc:
        raise TextModelValidationError(INVALID_MODEL_ERROR) from exc

    if set(labels) != {0, 1}:
        raise TextModelValidationError("Wrong model loaded")

    metadata["labels"] = sorted(labels)
    metadata["decision_threshold"] = float(metadata.get("decision_threshold", 0.55))
    metadata["confidence_threshold"] = float(metadata.get("confidence_threshold", 0.65))
    metadata["label_names"] = {0: "Not Stressed", 1: "Stressed"}
    return metadata


def _load_serialized_model(model_path):
    model_path = Path(model_path)
    if model_path.suffix.lower() != ".pkl":
        raise TextModelValidationError(INVALID_MODEL_ERROR)
    with model_path.open("rb") as file:
        return pickle.load(file)


def validate_binary_pipeline(pipeline):
    if not hasattr(pipeline, "named_steps"):
        raise TextModelValidationError(INVALID_MODEL_ERROR)

    vectorizer = pipeline.named_steps.get("vectorizer")
    classifier = pipeline.named_steps.get("classifier")
    if vectorizer is None or classifier is None:
        raise TextModelValidationError(INVALID_MODEL_ERROR)

    if not isinstance(vectorizer, SUPPORTED_VECTORIZERS):
        raise TextModelValidationError(INVALID_MODEL_ERROR)

    if not hasattr(classifier, "classes_") or not hasattr(pipeline, "predict_proba"):
        raise TextModelValidationError(INVALID_MODEL_ERROR)

    try:
        labels = [int(label) for label in classifier.classes_]
    except (TypeError, ValueError) as exc:
        raise TextModelValidationError(INVALID_MODEL_ERROR) from exc

    if set(labels) != {0, 1}:
        raise TextModelValidationError("Wrong model loaded")

    return sorted(labels)


def detect_urgent_language(text):
    candidate_text = basic_clean_text(text)
    if not candidate_text:
        return None

    for pattern in URGENT_PATTERNS:
        match = pattern.search(candidate_text)
        if match:
            return match.group(0)
    return None


def detect_context_keywords(text):
    tokens = set(tokenize_text(basic_clean_text(text)))
    matches = set()

    for context_name, keywords in CONTEXT_KEYWORDS.items():
        if tokens.intersection(keywords):
            matches.add(context_name)

    return matches


def _dedupe_preserve_order(items):
    seen = set()
    ordered_items = []

    for item in items:
        if item not in seen:
            ordered_items.append(item)
            seen.add(item)

    return ordered_items


def build_support_package(text, prediction, low_confidence=False, urgent_phrase=None):
    contexts = detect_context_keywords(text)

    if urgent_phrase:
        suggestions = [
            "Reach out to a trusted friend, family member, or local emergency support right now.",
            "Move to a safer space and avoid staying alone if you feel at risk.",
            "Pause everything else and focus on getting immediate human support.",
        ]
        message = (
            "Your message sounds highly distressed. Please seek immediate support from someone you trust "
            "or local emergency services if you feel unsafe."
        )
    elif prediction == 1:
        suggestions = [
            "Pause for one minute and take five slow breaths before continuing.",
            "Break the next task into one small step you can finish right now.",
            "Drink some water and take a short walk or screen break if possible.",
        ]
        message = "Your text suggests elevated stress right now. A short reset and one small next step can help."

        if "work" in contexts:
            suggestions.extend([
                "List your top three work tasks and finish only the first one before switching.",
                "Tell your manager or teammate early if your workload feels too heavy.",
            ])
        if "study" in contexts:
            suggestions.extend([
                "Study in 25-minute blocks with a 5-minute break between them.",
                "Summarize one topic at a time instead of trying to cover everything at once.",
            ])
        if "sleep" in contexts:
            suggestions.extend([
                "Reduce caffeine later in the day and aim for a simple wind-down routine tonight.",
                "Step away from your screen for a few minutes before bed.",
            ])
        if "family" in contexts or "relationship" in contexts:
            suggestions.extend([
                "Talk to one trusted person and explain what is stressing you in one clear sentence.",
                "Take a short pause before replying if emotions are running high.",
            ])
        if "money" in contexts:
            suggestions.extend([
                "Write down the money issue clearly and split it into today's action and later action.",
                "Ask for help early if a bill or payment is becoming overwhelming.",
            ])
    else:
        suggestions = [
            "Keep the habits that are helping you stay steady today.",
            "Take regular breaks, stay hydrated, and keep a balanced routine.",
            "Check in with yourself again later if your mood changes.",
        ]
        message = "Your text looks closer to low stress right now. Keep supporting the routines that help you feel stable."

    if low_confidence and not urgent_phrase:
        suggestions.append("If this result does not feel right, try entering a longer sentence with more detail.")
        message += " The model confidence is lower than usual, so a more detailed sentence may improve the estimate."

    suggestions = _dedupe_preserve_order(suggestions)[:5]
    return {
        "suggestion": suggestions[0],
        "suggestions": suggestions,
        "support_message": message,
    }


def build_text_prediction(
    raw_text,
    cleaned_text,
    prediction,
    confidence,
    probability_stress,
    source,
    reason="",
    low_confidence=False,
    urgent_phrase=None,
):
    prediction = int(prediction)
    confidence = max(0.0, min(float(confidence), 1.0))
    probability_stress = max(0.0, min(float(probability_stress), 1.0))
    probability_no_stress = 1.0 - probability_stress

    stress_level = PREDICTION_TO_STRESS_LEVEL[prediction]
    display_name = "Stressed" if prediction == 1 else "Not Stressed"
    stress_emoji = STRESS_LEVEL_EMOJIS[stress_level]
    result_text = display_name

    result = {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "prediction": prediction,
        "status": display_name,
        "confidence": confidence,
        "confidence_percentage": round(confidence * 100, 2),
        "confidence_text": f"{confidence * 100:.1f}%",
        "probability_stress": round(probability_stress, 4),
        "probability_no_stress": round(probability_no_stress, 4),
        "result_text": result_text,
        "result_display": result_text,
        "stress_score": TEXT_STRESS_SCORES[prediction],
        "stress_level": stress_level,
        "stress_level_display": f"{stress_emoji} {stress_level}",
        "source": source,
        "reason": reason,
        "is_low_confidence": low_confidence,
        "urgent_phrase": urgent_phrase,
    }

    result.update(build_support_package(raw_text, prediction, low_confidence=low_confidence, urgent_phrase=urgent_phrase))
    return result


class TextStressPredictor:
    def __init__(self, model_path=None, metadata_path=TEXT_METADATA_PATH):
        self.model_path = resolve_text_model_path(model_path)
        self.metadata_path = metadata_path
        self.metadata = load_text_metadata(metadata_path)
        self.pipeline = None
        self.labels = list(self.metadata.get("labels", [0, 1]))
        self.decision_threshold = float(self.metadata.get("decision_threshold", 0.55))
        self.confidence_threshold = float(self.metadata.get("confidence_threshold", 0.65))

    def load(self):
        if self.pipeline is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    "No trained binary text model was found. Train it first with `python train_text_model.py` using Stress.csv."
                )

            logger.info("Loading binary text model from %s", self.model_path)
            self.pipeline = _load_serialized_model(self.model_path)
            self.labels = validate_binary_pipeline(self.pipeline)

        return self.pipeline

    def predict(self, text):
        raw_text = text if isinstance(text, str) else ""
        if not raw_text.strip():
            raise ValueError("Text input cannot be empty.")

        cleaned_text = normalize_text(raw_text)
        if not cleaned_text:
            raise ValueError("Text input does not contain enough usable words after cleaning.")

        urgent_phrase = detect_urgent_language(raw_text)

        pipeline = self.load()
        probabilities = np.asarray(pipeline.predict_proba([raw_text])[0], dtype=np.float32)
        class_to_index = {int(label): index for index, label in enumerate(self.labels)}

        if 1 not in class_to_index:
            raise TextModelValidationError("Wrong model loaded")

        stress_index = class_to_index[1]
        probability_stress = float(probabilities[stress_index])
        prediction = 1 if probability_stress >= self.decision_threshold else 0
        confidence = probability_stress if prediction == 1 else 1.0 - probability_stress
        confidence = round(confidence, 4)

        source = "model"
        reason = "Prediction generated using the validated binary Stress.csv text pipeline."

        if urgent_phrase and prediction == 0:
            prediction = 1
            confidence = max(confidence, 0.65)
            probability_stress = max(probability_stress, 0.65)
            source = "rule+model"
            reason = "Urgent distress language detected, so the result was elevated to high stress."

        low_confidence = confidence < self.confidence_threshold

        return build_text_prediction(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            prediction=prediction,
            confidence=confidence,
            probability_stress=probability_stress,
            source=source,
            reason=reason,
            low_confidence=low_confidence,
            urgent_phrase=urgent_phrase,
        )


def summarize_face_for_stress(face_prediction):
    if not face_prediction:
        return None

    face_label = str(face_prediction.get("label") or face_prediction.get("display_label") or "Not Sure")
    face_confidence = float(face_prediction.get("confidence", 0.0))
    face_emoji = face_prediction.get("emoji") or ""

    return {
        "label": face_label,
        "emoji": face_emoji,
        "confidence": face_confidence,
        "confidence_text": f"{face_confidence * 100:.1f}%",
        "result_text": f"{face_label} ({face_confidence * 100:.1f}%)",
        "result_display": f"{face_emoji} {face_label} ({face_confidence * 100:.1f}%)".strip(),
        "stress_score": FACE_STRESS_SCORES.get(face_label, 1),
    }


def score_to_level(score):
    if score < 1.0:
        return "Low"
    if score < 2.0:
        return "Medium"
    return "High"


def combine_stress_assessment(face_prediction=None, text_prediction=None, bio_prediction=None):
    total_weight = 0.0
    total_weighted_score = 0.0
    
    components = []
    weights_used = {}

    face_summary = summarize_face_for_stress(face_prediction)
    if face_summary is not None and face_summary.get("stress_score") is not None:
        score = face_summary["stress_score"]
        weight = face_summary.get("confidence", 0.0)
        components.append(("Face", score, weight))

    if text_prediction is not None and text_prediction.get("stress_score") is not None:
        score = text_prediction["stress_score"]
        weight = text_prediction.get("confidence", 0.0)
        components.append(("Text", score, weight))
        
    if bio_prediction is not None:
        score = 2 if bio_prediction.get("status") == "Stressed" else 0
        raw_conf = bio_prediction.get("confidence", 0.0)
        if raw_conf > 1.0:
            raw_conf /= 100.0
        weight = min(raw_conf, 0.95)
        components.append(("Bio", score, weight))

    if not components:
        return None
        
    for name, score, weight in components:
        weights_used[name] = round(weight, 4)
        total_weight += weight
        total_weighted_score += score * weight

    if total_weight > 0:
        final_score = round(total_weighted_score / total_weight, 2)
    else:
        final_score = round(sum(score for _, score, _ in components) / len(components), 2)
        
    level = score_to_level(final_score)
    emoji = STRESS_LEVEL_EMOJIS[level]

    return {
        "level": level,
        "emoji": emoji,
        "score": final_score,
        "score_text": f"{final_score:.2f} / 2.00",
        "result_text": f"{level} ({final_score:.2f} / 2.00)",
        "result_display": f"{emoji} {level} ({final_score:.2f} / 2.00)",
        "components_used": [name for name, _, _ in components],
        "weights_used": weights_used,
        "mode": "multimodal" if len(components) > 1 else "single-modality",
    }
