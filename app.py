import logging
from pathlib import Path

from flask import Flask, render_template, request

from model.predictor import generate_gradcam_for_upload, predict_case, validate_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = Path("static") / "uploads"
if UPLOAD_FOLDER.exists() and not UPLOAD_FOLDER.is_dir():
    UPLOAD_FOLDER.unlink()
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    symptoms = request.form["symptoms"]

    upload_dir = Path(app.config["UPLOAD_FOLDER"])
    if upload_dir.exists() and not upload_dir.is_dir():
        upload_dir.unlink()
    upload_dir.mkdir(parents=True, exist_ok=True)

    path = str(upload_dir / file.filename)
    file.save(path)

    # Validation
    if not validate_image(path):
        return "❌ Invalid Image (QR / selfie / low quality)"

    result, confidence = predict_case(path, symptoms)
    gradcam_path = generate_gradcam_for_upload(path)
    logger.info("Prediction completed: %s (confidence=%s, gradcam=%s)", result, confidence, bool(gradcam_path))

    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        image=path,
        gradcam_image=gradcam_path,
    )

if __name__ == "__main__":
    app.run(debug=True)
