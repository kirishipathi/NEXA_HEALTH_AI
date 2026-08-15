# EndoNexa AI

## Multimodal endometriosis screening-support system

EndoNexa AI is a Flask-based research prototype for early endometriosis screening support. It combines a medical image pathway and a symptom-text pathway into one multimodal prediction pipeline. The objective is not to replace medical diagnosis, but to provide a transparent, explainable screening-support tool that can flag potentially suspicious cases for further clinical evaluation.

This project is intentionally framed as a screening-support system, not a clinical diagnostic system. The underlying dataset is a proxy dataset (PCOS pelvic ultrasound proxy mapping), and the model is therefore a research prototype for academic use rather than a validated clinical-grade tool.

## 1. Project overview

Endometriosis is often under-detected because symptoms are heterogeneous and imaging findings may be subtle or fragmented across clinical visits. EndoNexa AI addresses this by fusing two complementary signals:

- image-derived probability from a trained EfficientNet-B0 classifier
- symptom-derived probability from a logistic-regression model built on structured symptom features

The final decision uses a weighted fusion rule:

- image weight = 0.6
- symptom weight = 0.4

The model emits a binary prediction:

- Endometriosis Suspected
- Normal

The app displays confidence in the predicted class, not the raw positive-class probability.

## 2. Architecture

The project pipeline is:

1. User uploads an image and symptom text.
2. Image validation rejects invalid or low-quality uploads.
3. The image branch produces an image probability score.
4. The symptom branch converts text into feature values and outputs a symptom probability.
5. The weighted fusion model produces the final score.
6. The result is rendered in the web interface.
7. Grad-CAM is treated as optional and degrades gracefully when the exported Keras graph is incompatible with gradient tracing.

### High-level flow

```text
User input (image + symptoms)
        ↓
Image validation
        ↓
Image model (EfficientNet-B0)
        ↓
Symptom model (Logistic Regression)
        ↓
Weighted fusion (0.6 image + 0.4 symptoms)
        ↓
Prediction + confidence in predicted class
        ↓
Flask result page
```

## 3. Data and proxy justification

The project uses a proxy dataset for a final-year academic prototype. The repository contains processed splits under `data/processed/train`, `data/processed/val`, and `data/processed/test`, and the training/evaluation pipeline is built around those partitions.

This is not a real clinical endometriosis dataset. Instead, the project explicitly follows a research proxy strategy:

- the data is mapped to a binary screening task
- the project is designed to demonstrate multimodal healthcare AI architecture and evaluation pipelines
- the final system is framed as screening support, not clinical diagnosis

This framing is important for academic integrity. The prototype is valuable as a demonstration of systems design, multimodal fusion, model packaging, and web deployment, but it must not be presented as a clinically validated endometriosis detector.

## 4. Training methodology

### Image branch

- EfficientNet-B0 backbone with ImageNet initialization
- input image size: 224 × 224 × 3
- binary classification head with sigmoid activation
- trained in two stages: base freeze then fine-tuning of later layers
- model artifact saved as `model/efficientnet_b0_model.keras`

### Symptom branch

- structured symptom features generated from text keywords
- features include pain, infertility risk, cycle irregularity, pressure, bleeding, fatigue, and similar clinical descriptors
- logistic regression model trained with class balancing
- artifact saved as `model/symptom_logistic_regression.joblib`

### Fusion logic

The final score is calculated as:

```python
fused_prob = (0.6 * image_prob) + (0.4 * symptom_prob)
```

where:

- `image_prob` = image model probability for the positive class
- `symptom_prob` = symptom model probability for the positive class

The decision threshold is 0.5.

## 5. Phase 4 evaluation results

The evaluation summary in `results/phase4_evaluation_summary.json` reports the following metrics on the test split:

| Branch | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image model | 0.8793 | 1.0000 | 0.7034 | 0.8259 | 0.9889 |
| Symptom model | 0.8034 | 0.7607 | 0.7542 | 0.7574 | 0.8732 |
| Fusion model | 0.8897 | 0.8981 | 0.8220 | 0.8584 | 0.9694 |

These values show a usable screening prototype, but they should be interpreted as a research benchmark rather than clinical performance.

## 6. Confidence semantics and label fix

A genuine issue was identified and fixed: the app previously displayed the raw positive-class probability as “Confidence” even when the predicted class was “Normal”. For example, a raw positive probability of 4.48% was being shown directly as the confidence score for a Normal prediction, which is misleading.

The correct logic is now:

```python
if fused_prob >= 0.5:
    label = "Endometriosis Suspected"
    confidence = fused_prob * 100
else:
    label = "Normal"
    confidence = (1.0 - fused_prob) * 100
```

This means the app reports confidence in the predicted class, which is unambiguous and correct.

## Known Issues

The live prediction pipeline currently shows reduced sensitivity to positive (endometriosis-suspected) cases in manual testing, despite Phase 4 offline evaluation reporting 70% recall. Root cause under investigation — suspected label-orientation mismatch between the image and symptom branches. This does not affect the validity of the reported Phase 3/4 training and evaluation metrics, which were computed correctly on the test set.

## 7. Grad-CAM limitation

The project includes a Grad-CAM explainability path, but it remains optional because the exported EfficientNet model sometimes forms a nested Keras graph that is not compatible with gradient tracing. The runtime error observed was:

```text
ValueError: Output with path 0 is not connected to inputs
```

This is a real limitation of the current exported model structure, not a silent failure. The app keeps working because the pipeline catches this and continues with the prediction result; the overlay simply becomes unavailable for that case.

## 8. Ethical and clinical framing

This project should be used as a research and educational tool, not a medical diagnosis system. It is best positioned as:

- a prototype for screening support
- a demonstration of multimodal healthcare AI
- a final-year project for AI and full-stack application development
- a way to explore explainability and deployment patterns in healthcare AI

## 9. Setup and run instructions

### Environment

```bash
python -m venv .venv
. .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the web app

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## 10. Project structure

```text
NEXA_HEALTH_AI/
├── app.py
├── LICENSE
├── NOTES.md
├── README.md
├── requirements.txt
├── http_check.py
├── verify_live.py
├── verify_runtime_checks.py
├── .venv/
├── data/
│   ├── README.md
│   ├── class_balance_summary.json
│   ├── dataset_manifest.csv
│   ├── download_pcos_dataset.py
│   ├── prepare_pcos_proxy_dataset.py
│   ├── raw/
│   ├── processed/
│   └── symptom_features/
├── model/
│   ├── efficientnet_b0_model.keras
│   ├── efficientnet_b0_stage1.keras
│   ├── efficientnet_b0_finetuned.keras
│   ├── fusion_config.json
│   ├── gradcam.py
│   ├── predictor.py
│   ├── symptom_logistic_regression.joblib
│   ├── train_multimodal.py
│   ├── evaluate.py
│   └── __pycache__/
├── results/
│   ├── image_confusion_matrix.png
│   ├── image_roc_curve.png
│   ├── image_training_curves.png
│   ├── symptom_confusion_matrix.png
│   ├── symptom_roc_curve.png
│   ├── fusion_confusion_matrix.png
│   ├── fusion_roc_curve.png
│   ├── phase3_training_summary.json
│   └── phase4_evaluation_summary.json
├── static/
│   ├── style.css
│   └── uploads/
└── templates/
    ├── index.html
    └── result.html
```

## 11. Limitations and future work

This prototype is intentionally limited by the dataset and explainability constraints. Recommended future work includes:

- replace the proxy dataset with a clinically validated endometriosis dataset
- re-export the image model as a flat functional graph for robust Grad-CAM support
- improve symptom feature extraction with NLP or a clinical text encoder
- add calibration reporting for prediction confidence
- run external validation and broader ablation studies
- extend the app to clinician-facing dashboards and audit logs

## 12. License

This project is distributed under the MIT License.
