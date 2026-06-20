from flask import Flask, render_template, request
import os
from model.predictor import predict_case, validate_image

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    symptoms = request.form["symptoms"]

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    # Validation
    if not validate_image(path):
        return "❌ Invalid Image (QR / selfie / low quality)"

    result, confidence = predict_case(path, symptoms)

    return render_template("result.html",
                           result=result,
                           confidence=confidence,
                           image=path)

if __name__ == "__main__":
    app.run(debug=True)
