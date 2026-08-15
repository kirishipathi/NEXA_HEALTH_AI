from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import joblib
import numpy as np
import tensorflow as tf

from model.gradcam import generate_gradcam_overlay

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODEL_DIR.parent
UPLOAD_DIR = ROOT_DIR / "static" / "uploads"


def _ensure_upload_directory(path: Path) -> Path:
    if path.exists() and not path.is_dir():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


FEATURE_COLUMNS = [
    "pain_level",
    "cycle_irregularity",
    "pain_during_intercourse",
    "pelvic_pressure",
    "heavy_bleeding",
    "infertility_history",
    "fatigue",
]

_RUNTIME_MODELS = None


def _load_runtime_models():
    global _RUNTIME_MODELS
    if _RUNTIME_MODELS is not None:
        return _RUNTIME_MODELS

    image_model = tf.keras.models.load_model(MODEL_DIR / "efficientnet_b0_model.keras")
    symptom_model = joblib.load(MODEL_DIR / "symptom_logistic_regression.joblib")
    with (MODEL_DIR / "fusion_config.json").open("r", encoding="utf-8") as file:
        fusion_config = json.load(file)
    _RUNTIME_MODELS = (image_model, symptom_model, fusion_config)
    return _RUNTIME_MODELS


def _symptoms_to_features(symptoms: str):
    text = (symptoms or "").lower()
    features = {
        "pain_level": 3,
        "cycle_irregularity": 0,
        "pain_during_intercourse": 0,
        "pelvic_pressure": 0,
        "heavy_bleeding": 0,
        "infertility_history": 0,
        "fatigue": 2,
    }

    if any(keyword in text for keyword in ["severe", "sharp", "intense", "cramping", "throbbing"]):
        features["pain_level"] = 8
    elif any(keyword in text for keyword in ["moderate", "pain", "ache"]):
        features["pain_level"] = 6
    elif any(keyword in text for keyword in ["mild", "slight"]):
        features["pain_level"] = 4

    if any(keyword in text for keyword in ["irregular", "missed", "late", "cycle"]):
        features["cycle_irregularity"] = 1

    if any(keyword in text for keyword in ["intercourse", "sex", "during sex", "deep pain"]):
        features["pain_during_intercourse"] = 1

    if any(keyword in text for keyword in ["pressure", "fullness", "pelvic", "heaviness"]):
        features["pelvic_pressure"] = 1

    if any(keyword in text for keyword in ["heavy bleeding", "bleeding", "menstrual", "flooding"]):
        features["heavy_bleeding"] = 1

    if any(keyword in text for keyword in ["infertility", "unable to conceive", "pregnancy issue", "fertility"]):
        features["infertility_history"] = 1

    if any(keyword in text for keyword in ["fatigue", "tired", "weak", "exhausted", "low energy"]):
        features["fatigue"] = 6

    if any(keyword in text for keyword in ["severe", "heavy", "pressure", "intense"]):
        features["pelvic_pressure"] = max(features["pelvic_pressure"], 1)

    return np.asarray([float(features[col]) for col in FEATURE_COLUMNS], dtype=np.float32).reshape(1, -1)


# 🚫 Validate Image (important hackathon feature)
def validate_image(path):
    img = cv2.imread(path)

    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Reject low-quality / blank / QR-like images
    if np.var(gray) < 50:
        return False

    return True


def generate_gradcam_for_upload(image_path: str):
    """Generate the Grad-CAM overlay for an uploaded image when the model graph supports it."""
    source = Path(image_path)
    upload_dir = _ensure_upload_directory(UPLOAD_DIR)
    overlay_path = upload_dir / f"{source.stem}_gradcam.png"

    try:
        generate_gradcam_overlay(source, overlay_path)
        return str(Path("static") / "uploads" / overlay_path.name)
    except Exception as exc:  # pragma: no cover - runtime fallback for exported nested-model limitation
        logger.warning("Grad-CAM generation failed for %s: %s", source, exc)
        return None


# 🧠 Multimodal Prediction (Image + Symptoms)
def predict_case(image_path, symptoms):
    image_model, symptom_model, fusion_config = _load_runtime_models()

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    # Match the training pipeline exactly: the model contains a Rescaling layer, so we must
    # pass the raw 0..255 pixel values and avoid a second normalization by 255.
    image_batch = image_resized.astype(np.float32)
    image_batch = np.expand_dims(image_batch, axis=0)

    image_prob = float(image_model.predict(image_batch, verbose=0)[0, 0])
    symptom_input = _symptoms_to_features(symptoms)
    symptom_prob = float(symptom_model.predict_proba(symptom_input)[0, 1])

    fused_prob = (fusion_config["image_weight"] * image_prob) + (fusion_config["symptom_weight"] * symptom_prob)
    if fused_prob >= 0.5:
        label = "Endometriosis Suspected"
        confidence = round(float(fused_prob) * 100, 2)
    else:
        label = "Normal"
        confidence = round(float(1.0 - fused_prob) * 100, 2)
    return label, confidence
