# 🧠 EndoVision-AI
### AI-Powered Multimodal Detection of Endometriosis

---

## 📌 Overview
#### EndoVision-AI is a HealthTech solution that uses multimodal AI to detect endometriosis by combining:

🩺 Medical Imaging (MRI / Ultrasound)  
🧾 Clinical Symptoms (Text Input)  

#### It addresses delayed diagnosis by integrating both data sources into a single AI-powered system.
#### This system ensures reliable AI predictions by validating medical inputs before analysis.
---

## 🎥 Demo & Presentation

| 🎬 Demo Video | 📊 Presentation |
|--------------|----------------|
| [▶ Watch Demo](https://drive.google.com/file/d/1n4vBPCUzy0_rW94o4BN1DANM8lrIyN8J/view?usp=drive_link) | [📂 View PPT](https://drive.google.com/file/d/1dJ7Jjhvrws9iRyBTfhR5-KXAnkCZTEfs/view?usp=drive_link) |

---
## 🚨 Problem Statement
Endometriosis affects ~10% of women globally and often takes **7–10 years** to diagnose due to fragmented analysis of imaging and symptoms.

---

## 💡 Proposed Solution
We built an AI system that:

```
Accepts medical images + symptom input
Uses multimodal fusion (Image + Text)
Predicts disease likelihood with confidence score
```

---

## ✨ Key Features
```
🔗 Multimodal AI (Image + Text)
⚡ Real-time Prediction
📊 Confidence Score Output
🧠 Explainability (Heatmap simulation)
🛡️ Input Validation (Rejects QR / invalid images)
```

---

## 🧱 System Architecture
```
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
```

---

## ⚙️ Tech Stack


---

## ⚙️ Tech Stack

### Frontend:
HTML, CSS  

### Backend:
Python (Flask)  

### AI/ML:
OpenCV (Image Processing)  
NumPy  
Rule-based Text Analysis  
Multimodal Fusion Logic  

---

## 🔐 Input Validation
To ensure reliable predictions, the system:


```
✔ Rejects QR codes, selfies, random images  
✔ Validates image resolution & pixel variance  
✔ Prevents misleading AI outputs  
```

---

## 🚀 How to Run

### Backend

```bash
pip install -r requirements.txt
python app.py
```


## 📡 API Endpoint

### POST /predict

#### Input:
- image (file)  
- symptoms (text)  

#### Output:
```json
{
  "status": "success",
  "message": "Endometriosis Detected",
  "confidence": 82.5
}
```

---

## 🌍 Impact
```
⏱️ Reduces diagnosis delay  
🏥 Assists doctors in early detection  
🌐 Improves healthcare accessibility  
```

---

## 📸 Output Screens

![Output 1](https://github.com/user-attachments/assets/2543f0b2-065f-4d92-9287-bb7b5cee1735)
![Output 2](https://github.com/user-attachments/assets/74eeb9f4-fffa-4104-899e-d90404059920)
![Output 3](https://github.com/user-attachments/assets/f8d148a6-1edc-415a-992c-e2d0d9abab21)
![Output 4](https://github.com/user-attachments/assets/af0a385c-0b44-4bff-bc9d-9cd89b79f24b)
![Output 5](https://github.com/user-attachments/assets/5efb20e2-1872-4c83-8cff-4bade32a0de7)
![Output 7](https://github.com/user-attachments/assets/0ffb0067-2ec0-4182-9cd2-7c124ccb5849)
![Output 6](https://github.com/user-attachments/assets/2193d682-8268-4c96-9ce6-19fd24d88b55)


---

## 👩‍💻 Team
```
GEDIPUDI DARSHANI - Frontend & AI
MAHASRI P - Backend
OVIYA N - Security
```

---

## 🧾 Conclusion

EndoVision-AI demonstrates how **multimodal AI** can significantly improve early detection of endometriosis by integrating both medical imaging and clinical symptoms into a unified system.

By introducing a **robust input validation layer**, the system avoids misleading predictions from irrelevant inputs such as QR codes or random images.

### Key Takeaways:
- Combining multiple data sources improves accuracy  
- Validation is critical in healthcare AI  
- Real-world scalable AI systems are achievable  

👉 EndoVision-AI moves towards **faster, safer, and more accessible diagnosis** in modern healthcare.

---

⭐ *Star this repository if you found it useful!*
