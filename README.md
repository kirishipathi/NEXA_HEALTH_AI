<div align="center">

# 🧬 EndoNexa AI

### 🩺 Multimodal AI System for Endometriosis Detection

**Fusing medical imaging analysis with clinical symptom evaluation to support earlier, more informed diagnosis.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Computation-013243?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

</div>

<br>

## 📌 Overview

**EndoNexa AI** is a healthtech application that assists in the early detection of endometriosis by combining **medical imaging analysis** with **clinical symptom evaluation**. Instead of relying on a single data source, the system fuses image-based and text-based signals to produce a more informed prediction with an associated confidence score.

Endometriosis is a condition that commonly takes **years to diagnose** due to fragmented evaluation of scans and patient-reported symptoms. EndoNexa AI addresses that gap by processing both inputs together through a single pipeline.

<br>

## ✨ Key Features

| | Feature | Description |
|---|---|---|
| 🔗 | **Multimodal Input** | Accepts both a medical image (MRI/Ultrasound) and a symptom description in the same request |
| 🛡️ | **Image Validation Layer** | Automatically rejects invalid uploads such as QR codes, selfies, or low-quality/blank images before they reach the prediction stage |
| ⚖️ | **Weighted Fusion Logic** | Combines image-derived and text-derived scores into a single, interpretable confidence output |
| 🖥️ | **Lightweight Web Interface** | Simple upload form and result view, no client-side dependencies |
| ⚡ | **Fast, Local Inference** | No external API calls; runs entirely on the Flask backend |

<br>

## 🧱 System Architecture
            ┌──────────────────────────┐
            │  🧍 Patient Input Layer   │
            │  (Image + Symptom Text)   │
            └─────────────┬────────────┘
                          │
                          ▼
            ┌──────────────────────────┐
            │  🛡️ Image Validation      │
            │  (OpenCV — variance &     │
            │   quality checks)         │
            └─────────────┬────────────┘
                          │  valid image
                          ▼
    ┌─────────────────────────────────────────┐
    │             ⚙️ Feature Scoring             │
    │  ┌────────────────┐   ┌─────────────────┐│
    │  │ 🖼️ Image Score  │   │ 📝 Symptom Keyword││
    │  │  (CV-based)     │   │  Matching (NLP)   ││
    │  └────────┬───────┘   └────────┬──────────┘│
    └───────────┼────────────────────┼───────────┘
                 │                    │
                 ▼                    ▼
          ┌────────────────────────────────┐
          │      🔗 Multimodal Fusion        │
          │   (Weighted Score Combination)   │
          └─────────────┬───────────────────┘
                          │
                          ▼
            ┌──────────────────────────┐
            │  📊 Prediction Output      │
            │  (Result + Confidence %)   │
            └──────────────────────────┘

**Flow:**

1. 🧍 The user uploads a medical image and enters their symptoms through the web interface
2. 🛡️ The image is validated using pixel variance checks to filter out non-medical or low-quality uploads
3. ⚙️ Two independent scores are generated — one from image analysis, one from symptom keyword matching
4. 🔗 Both scores are combined using a weighted fusion formula to produce a final prediction and confidence percentage
5. 📊 The result is rendered back to the user along with the uploaded image

<br>

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| 🎨 **Frontend** | HTML, CSS |
| 🐍 **Backend** | Python (Flask) |
| 🖼️ **Image Processing** | OpenCV, NumPy |
| 📝 **Text Analysis** | Rule-based keyword scoring |
| 🔗 **Fusion Logic** | Weighted score combination |

<br>

## 📡 API Reference

### `POST /predict`

**Request**

| Field | Type | Description |
|---|---|---|
| `image` | file | Medical scan (MRI/Ultrasound) |
| `symptoms` | text | Patient-reported symptoms |

**Response**

```json
{
  "status": "success",
  "message": "Endometriosis Detected",
  "confidence": 82.5
}
```

<br>

## 🚀 Getting Started

```bash
pip install -r requirements.txt
python app.py
```

The application will be available at **http://localhost:5000** 🌐

<br>

## 🔐 Input Validation

To prevent unreliable predictions, every uploaded image passes through a validation check that:

- 🚫 Rejects QR codes and visually flat/low-variance images
- 🚫 Filters out unreadable or corrupted files
- ✅ Ensures only genuine scan-like images proceed to the prediction stage

<br>

## 🌍 Impact

- ⏱️ **Reduces delay** in identifying potential endometriosis cases
- 🏥 Supports clinicians with an additional **data-driven reference point**
- 🌐 Demonstrates how **multimodal fusion** can improve reliability over single-source diagnostics in healthcare AI

<br>

## 📜 License

This project is licensed under the **MIT License**.

<br>

<div align="center">

### ⭐ Star this repository if you found it useful!

</div>