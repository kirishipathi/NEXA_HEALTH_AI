import cv2
import numpy as np
import random

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


# 🧠 Multimodal Prediction (Image + Symptoms)
def predict_case(image_path, symptoms):
    keywords = ["pain", "pelvic", "cramps", "infertility"]

    symptom_score = sum([1 for word in keywords if word in symptoms.lower()])

    # Simulated image confidence
    image_score = random.uniform(0.4, 0.9)

    text_score = symptom_score / len(keywords)

    final_score = (0.6 * image_score) + (0.4 * text_score)

    if final_score > 0.5:
        return "Endometriosis Detected", round(final_score * 100, 2)
    else:
        return "Normal", round((1 - final_score) * 100, 2)
