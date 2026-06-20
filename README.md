🧬 EndoNexa AI

🩺 Multimodal AI System for Endometriosis Detection


📌 Overview

EndoNexa AI is a healthtech application that assists in the early detection of endometriosis by combining medical imaging analysis with clinical symptom evaluation. Instead of relying on a single data source, the system fuses image-based and text-based signals to produce a more informed prediction with an associated confidence score.

Endometriosis is a condition that commonly takes years to diagnose due to fragmented evaluation of scans and patient-reported symptoms. EndoNexa AI addresses that gap by processing both inputs together through a single pipeline.


✨ Key Features


🔗 Multimodal Input — accepts both a medical image (MRI/Ultrasound) and a symptom description in the same request
🛡️ Image Validation Layer — automatically rejects invalid uploads such as QR codes, selfies, or low-quality/blank images before they reach the prediction stage
⚖️ Weighted Fusion Logic — combines image-derived and text-derived scores into a single, interpretable confidence output
🖥️ Lightweight Web Interface — simple upload form and result view, no client-side dependencies
⚡ Fast, Local Inference — no external API calls; runs entirely on the Flask backend



🧱 System Architecture

                ┌──────────────────────────┐
                │ 🧍 Patient Input Layer    │
                │  (Image + Symptom Text)   │
                └─────────────┬──────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │ 🛡️ Image Validation       │
                │  (OpenCV — variance &     │
                │   quality checks)         │
                └─────────────┬──────────────┘
                              │  valid image
                              ▼
        ┌─────────────────────────────────────────┐
        │           ⚙️ Feature Scoring               │
        │  ┌───────────────┐   ┌──────────────────┐ │
        │  │ 🖼️ Image Score │   │ 📝 Symptom Keyword │ │
        │  │ (CV-based)    │   │ Matching (NLP)     │ │
        │  └───────┬───────┘   └─────────┬──────────┘ │
        └──────────┼─────────────────────┼────────────┘
                   │                     │
                   ▼                     ▼
              ┌─────────────────────────────┐
              │   🔗 Multimodal Fusion        │
              │  (Weighted Score Combination) │
              └─────────────┬─────────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │ 📊 Prediction Output       │
                │  (Result + Confidence %)   │
                └──────────────────────────┘

Flow:


🧍 The user uploads a medical image and enters their symptoms through the web interface.
🛡️ The image is validated using pixel variance checks to filter out non-medical or low-quality uploads.
⚙️ Two independent scores are generated — one from image analysis, one from symptom keyword matching.
🔗 Both scores are combined using a weighted fusion formula to produce a final prediction and confidence percentage.
📊 The result is rendered back to the user along with the uploaded image.



⚙️ Tech Stack

LayerTechnology🎨 FrontendHTML, CSS🐍 BackendPython (Flask)🖼️ Image ProcessingOpenCV, NumPy📝 Text AnalysisRule-based keyword scoring🔗 Fusion LogicWeighted score combination


📡 API Reference

Endpoint: POST /predict

Request:

FieldTypeDescriptionimagefileMedical scan (MRI/Ultrasound)symptomstextPatient-reported symptoms

Response:

json{
  "status": "success",
  "message": "Endometriosis Detected",
  "confidence": 82.5
}


🚀 Getting Started

bashpip install -r requirements.txt
python app.py

The application will be available at http://localhost:5000 🌐


🔐 Input Validation

To prevent unreliable predictions, every uploaded image passes through a validation check that:


🚫 Rejects QR codes and visually flat/low-variance images
🚫 Filters out unreadable or corrupted files
✅ Ensures only genuine scan-like images proceed to the prediction stage



🌍 Impact


⏱️ Reduces delay in identifying potential endometriosis cases
🏥 Supports clinicians with an additional data-driven reference point
🌐 Demonstrates how multimodal fusion can improve reliability over single-source diagnostics in healthcare AI



📄 License

This project is licensed under the MIT License.


⭐ Star this repository if you found it useful!