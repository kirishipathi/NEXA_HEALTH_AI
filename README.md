# 🧠 NEXA_HEALTH_AI

### AI-Powered Multimodal Detection of Endometriosis

---

## 📌 Overview
NEXA_HEALTH_AI is a HealthTech solution that uses multimodal AI to detect endometriosis by combining:

🩺 Medical Imaging (MRI / Ultrasound)
🧾 Clinical Symptoms (Text Input)

It addresses delayed diagnosis by integrating both data sources into a single AI-powered system.
This system ensures reliable AI predictions by validating medical inputs before analysis.

## 🎥 Demo & Presentation
🎬 Demo Video	📊 Presentation
▶ Watch Demo	📂 View PPT

## 🚨 Problem Statement
Endometriosis affects ~10% of women globally and often takes 7–10 years to diagnose due to fragmented analysis of imaging and symptoms.

## 💡 Proposed Solution
We built an AI system that:
- Accepts medical images + symptom input
- Uses multimodal fusion (Image + Text)
- Predicts disease likelihood with confidence score

## ✨ Key Features
🔗 Multimodal AI (Image + Text)
⚡ Real-time Prediction
📊 Confidence Score Output
🧠 Explainability (Heatmap simulation)
🛡️ Input Validation (Rejects QR / invalid images)

## 🧱 System Architecture
Patient Input (Image + Symptoms)

│

▼

Preprocessing Layer

│

▼

Feature Extraction

(CNN for Image + NLP for Text)

│

▼

Multimodal Fusion

│

▼

Prediction Output (with Confidence)

## ⚙️ Tech Stack
**Frontend:** HTML, CSS
**Backend:** Python (Flask)
**AI/ML:** OpenCV (Image Processing), NumPy, Rule-based Text Analysis, Multimodal Fusion Logic

## 🔐 Input Validation
To ensure reliable predictions, the system:
✔ Rejects QR codes, selfies, random images
✔ Validates image resolution & pixel variance
✔ Prevents misleading AI outputs

## 🚀 How to Run
### Backend
pip install -r requirements.txt

python app.py

## 📡 API Endpoint
**POST /predict**

**Input:**
- image (file)
- symptoms (text)

**Output:**
```json
{
  "status": "success",
  "message": "Endometriosis Detected",
  "confidence": 82.5
}
```

## 🌍 Impact
⏱️ Reduces diagnosis delay
🏥 Assists doctors in early detection
🌐 Improves healthcare accessibility

## 📸 Output Screens
Output 1 | Output 2 | Output 3 | Output 4 | Output 5 | Output 6 | Output 7

## 👩‍💻 Team
- KIRISHIPATHI - Frontend & AI
- KIRISHIPATHI - backend
- KIRISHIPATHI - Security

## 🧾 Conclusion
NEXA_HEALTH_AI demonstrates how multimodal AI can significantly improve early detection of endometriosis by integrating both medical imaging and clinical symptoms into a unified system.

By introducing a robust input validation layer, the system avoids misleading predictions from irrelevant inputs such as QR codes or random images.

**Key Takeaways:**
- Combining multiple data sources improves accuracy
- Validation is critical in healthcare AI
- Real-world scalable AI systems are achievable

👉 NEXA_HEALTH_AI moves towards faster, safer, and more accessible diagnosis in modern healthcare.

⭐ Star this repository if you found it useful!