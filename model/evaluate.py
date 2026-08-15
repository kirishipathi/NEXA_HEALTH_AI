from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"
SYMPTOM_PATH = DATA_DIR / "symptom_features" / "simulated_symptom_dataset.csv"
CONFIG_PATH = MODEL_DIR / "fusion_config.json"

RESULTS_DIR.mkdir(exist_ok=True)


def load_models():
    image_model = tf.keras.models.load_model(MODEL_DIR / "efficientnet_b0_model.keras")
    symptom_model = joblib.load(MODEL_DIR / "symptom_logistic_regression.joblib")
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        fusion_config = json.load(file)
    return image_model, symptom_model, fusion_config


def get_test_image_dataset():
    split_path = DATA_DIR / "processed" / "test"
    ds = tf.keras.utils.image_dataset_from_directory(
        split_path,
        labels="inferred",
        image_size=(224, 224),
        batch_size=32,
        label_mode="binary",
        shuffle=False,
    )
    return ds


def get_test_symptom_frame():
    summary = json.loads((DATA_DIR / "class_balance_summary.json").read_text(encoding="utf-8"))
    train_count = int(summary["train"]["total"])
    val_count = int(summary["val"]["total"])
    test_count = int(summary["test"]["total"])

    symptom_df = pd.read_csv(SYMPTOM_PATH).sort_values("sample_id").reset_index(drop=True)
    start = train_count + val_count
    end = start + test_count
    return symptom_df.iloc[start:end].reset_index(drop=True)


def get_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": cm.tolist(),
        "threshold": threshold,
    }


def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    tick_labels = ["Normal", "Suspected"]
    plt.xticks([0, 1], tick_labels)
    plt.yticks([0, 1], tick_labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_prob, title, save_path):
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5.5, 5.5))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    image_model, symptom_model, fusion_config = load_models()
    test_ds = get_test_image_dataset()

    image_prob = image_model.predict(test_ds, verbose=0).ravel()
    image_true = np.concatenate([y for _, y in test_ds], axis=0)

    symptom_df = get_test_symptom_frame()
    feature_cols = [
        "pain_level",
        "cycle_irregularity",
        "pain_during_intercourse",
        "pelvic_pressure",
        "heavy_bleeding",
        "infertility_history",
        "fatigue",
    ]
    symptom_X = symptom_df[feature_cols].values.astype(np.float32)
    symptom_true = symptom_df["label"].values.astype(int)
    symptom_prob = symptom_model.predict_proba(symptom_X)[:, 1]

    image_metrics = get_metrics(image_true, image_prob)
    symptom_metrics = get_metrics(symptom_true, symptom_prob)

    fused_prob = (fusion_config["image_weight"] * image_prob) + (fusion_config["symptom_weight"] * symptom_prob)
    fused_metrics = get_metrics(image_true, fused_prob)

    plot_confusion_matrix(np.array(image_metrics["confusion_matrix"]), "Image model confusion matrix", RESULTS_DIR / "image_confusion_matrix.png")
    plot_confusion_matrix(np.array(symptom_metrics["confusion_matrix"]), "Symptom model confusion matrix", RESULTS_DIR / "symptom_confusion_matrix.png")
    plot_confusion_matrix(np.array(fused_metrics["confusion_matrix"]), "Fusion confusion matrix", RESULTS_DIR / "fusion_confusion_matrix.png")
    plot_roc_curve(image_true, image_prob, "Image ROC curve", RESULTS_DIR / "image_roc_curve.png")
    plot_roc_curve(symptom_true, symptom_prob, "Symptom ROC curve", RESULTS_DIR / "symptom_roc_curve.png")
    plot_roc_curve(image_true, fused_prob, "Fusion ROC curve", RESULTS_DIR / "fusion_roc_curve.png")

    warning = None
    if (image_metrics["accuracy"] >= 0.99 or symptom_metrics["accuracy"] >= 0.99 or fused_metrics["accuracy"] >= 0.99):
        warning = (
            "At least one model is at or near 1.0000 accuracy. This is a likely leakage or trivial-feature warning, "
            "not a validation of robust performance. Phase 5 should not continue until this issue is investigated."
        )

    summary = {
        "data_scope": "PCOS pelvic ultrasound proxy dataset mapped to endometriosis-suspected/normal labels for screening support only.",
        "split_used": "test_only",
        "image_branch": image_metrics,
        "symptom_branch": symptom_metrics,
        "fusion_branch": fused_metrics,
        "warning": warning,
        "artifact_paths": {
            "image_confusion_matrix": str(RESULTS_DIR / "image_confusion_matrix.png"),
            "image_roc_curve": str(RESULTS_DIR / "image_roc_curve.png"),
            "symptom_confusion_matrix": str(RESULTS_DIR / "symptom_confusion_matrix.png"),
            "symptom_roc_curve": str(RESULTS_DIR / "symptom_roc_curve.png"),
            "fusion_confusion_matrix": str(RESULTS_DIR / "fusion_confusion_matrix.png"),
            "fusion_roc_curve": str(RESULTS_DIR / "fusion_roc_curve.png"),
        },
    }

    with (RESULTS_DIR / "phase4_evaluation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps({
        "image_accuracy": round(image_metrics["accuracy"], 4),
        "image_precision": round(image_metrics["precision"], 4),
        "image_recall": round(image_metrics["recall"], 4),
        "image_f1": round(image_metrics["f1"], 4),
        "image_roc_auc": round(image_metrics["roc_auc"], 4),
        "symptom_accuracy": round(symptom_metrics["accuracy"], 4),
        "symptom_precision": round(symptom_metrics["precision"], 4),
        "symptom_recall": round(symptom_metrics["recall"], 4),
        "symptom_f1": round(symptom_metrics["f1"], 4),
        "symptom_roc_auc": round(symptom_metrics["roc_auc"], 4),
        "fusion_accuracy": round(fused_metrics["accuracy"], 4),
        "fusion_precision": round(fused_metrics["precision"], 4),
        "fusion_recall": round(fused_metrics["recall"], 4),
        "fusion_f1": round(fused_metrics["f1"], 4),
        "fusion_roc_auc": round(fused_metrics["roc_auc"], 4),
        "warning": warning,
    }, indent=2))


if __name__ == "__main__":
    main()
