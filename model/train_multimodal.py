"""Train the image, symptom, and fusion branches for the EndoNexa AI prototype.

This script follows the project's Phase 3 requirements:
- EfficientNet-B0 image branch with ImageNet pretraining
- 60/40 image/symptom fusion weight
- logistic-regression symptom model
- model artifacts saved under /model
- training curves saved under /results

It uses the prepared PCOS pelvic ultrasound proxy dataset.
The project remains explicit that this is a proxy for endometriosis screening support,
not a clinically verified endometriosis diagnosis.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SYMPTOM_PATH = DATA_DIR / "symptom_features" / "simulated_symptom_dataset.csv"
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"

FEATURE_COLUMNS = [
    "pain_level",
    "cycle_irregularity",
    "pain_during_intercourse",
    "pelvic_pressure",
    "heavy_bleeding",
    "infertility_history",
    "fatigue",
]

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def get_split_dataset(split_name: str):
    """Create an image dataset from the processed split directories."""
    split_path = PROCESSED_DIR / split_name
    if not split_path.exists():
        raise FileNotFoundError(f"Processed data for split '{split_name}' not found at {split_path}")

    ds = tf.keras.utils.image_dataset_from_directory(
        split_path,
        labels="inferred",
        image_size=(224, 224),
        batch_size=32,
        label_mode="binary",
        shuffle=(split_name == "train"),
        seed=42,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def build_image_model() -> Model:
    """Create the EfficientNet-B0 image classifier with frozen backbone stage 1."""
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3), name="image_input")
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = base_model(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="image_output")(x)
    model = Model(inputs, outputs, name="efficientnet_b0_pcos_proxy")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_curves(history_1: dict, history_2: dict, save_path: Path) -> None:
    """Save loss and accuracy curves for both fine-tuning stages."""
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history_1["loss"], label="Stage 1 train loss")
    plt.plot(history_1["val_loss"], label="Stage 1 val loss")
    plt.plot(history_2["loss"], label="Stage 2 train loss")
    plt.plot(history_2["val_loss"], label="Stage 2 val loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_1["accuracy"], label="Stage 1 train acc")
    plt.plot(history_1["val_accuracy"], label="Stage 1 val acc")
    plt.plot(history_2["accuracy"], label="Stage 2 train acc")
    plt.plot(history_2["val_accuracy"], label="Stage 2 val acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_image_branch() -> tuple[Model, dict, dict]:
    """Train the image model in two steps: freeze base, then unfreeze late layers."""
    train_ds = get_split_dataset("train")
    val_ds = get_split_dataset("val")

    model = build_image_model()
    checkpoint_1 = ModelCheckpoint(
        str(MODEL_DIR / "efficientnet_b0_stage1.keras"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
    )
    early_stop_1 = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

    history_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=8,
        callbacks=[checkpoint_1, early_stop_1],
        verbose=1,
    ).history

    # Unfreeze the top layers of EfficientNet while keeping the lower layers frozen.
    efficientnet_layer = model.get_layer("efficientnetb0")
    efficientnet_layer.trainable = True
    for layer in efficientnet_layer.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_2 = ModelCheckpoint(
        str(MODEL_DIR / "efficientnet_b0_finetuned.keras"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
    )
    early_stop_2 = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=[checkpoint_2, early_stop_2],
        verbose=1,
    ).history

    plot_training_curves(history_1, history_2, save_path=RESULTS_DIR / "image_training_curves.png")
    return model, history_1, history_2


def train_symptom_model() -> tuple[Pipeline, dict]:
    """Train a logistic-regression symptom model with a small clinically plausible feature set."""
    summary = json.loads((DATA_DIR / "class_balance_summary.json").read_text(encoding="utf-8"))
    symptom_df = pd.read_csv(SYMPTOM_PATH).sort_values("sample_id").reset_index(drop=True)

    train_count = int(summary["train"]["total"])
    val_count = int(summary["val"]["total"])
    test_count = int(summary["test"]["total"])

    X = symptom_df[FEATURE_COLUMNS].values.astype(np.float32)
    y = symptom_df["label"].values.astype(int)

    train_end = train_count
    val_end = train_count + val_count
    test_end = train_count + val_count + test_count

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:test_end], y[val_end:test_end]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    return model, {
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
    }


def save_artifacts(image_model: Model, symptom_model: Pipeline, fusion_config: dict) -> None:
    """Persist model weights and configuration files for later inference."""
    image_model.save(MODEL_DIR / "efficientnet_b0_model.keras")
    import joblib
    joblib.dump(symptom_model, MODEL_DIR / "symptom_logistic_regression.joblib")

    with (MODEL_DIR / "fusion_config.json").open("w", encoding="utf-8") as file:
        json.dump(fusion_config, file, indent=2)


def main() -> None:
    """Train image and symptom branches, then combine them using 60/40 weighting."""
    image_model, history_1, history_2 = train_image_branch()
    symptom_model, symptom_stats = train_symptom_model()

    test_ds = get_split_dataset("test")
    image_probs = image_model.predict(test_ds, verbose=0).ravel()
    image_labels = np.concatenate([y for _, y in test_ds], axis=0)

    summary = json.loads((DATA_DIR / "class_balance_summary.json").read_text(encoding="utf-8"))
    train_count = int(summary["train"]["total"])
    val_count = int(summary["val"]["total"])
    test_count = int(summary["test"]["total"])

    symptom_df = pd.read_csv(SYMPTOM_PATH).sort_values("sample_id").reset_index(drop=True)
    symptom_test = symptom_df.iloc[train_count + val_count : train_count + val_count + test_count].copy()
    symptom_prob = symptom_model.predict_proba(symptom_test[FEATURE_COLUMNS].values.astype(np.float32))[:, 1]

    fused_prob = 0.6 * image_probs + 0.4 * symptom_prob
    fused_pred = (fused_prob >= 0.5).astype(int)
    fused_accuracy = float(accuracy_score(image_labels.astype(int), fused_pred))

    save_artifacts(
        image_model,
        symptom_model,
        {
            "image_weight": 0.6,
            "symptom_weight": 0.4,
            "threshold": 0.5,
            "fusion_output": "weighted_average",
            "proxy_note": "This fusion is trained on the PCOS pelvic ultrasound proxy dataset and is not a verified endometriosis diagnostic model.",
            "image_model_path": str(MODEL_DIR / "efficientnet_b0_model.keras"),
            "symptom_model_path": str(MODEL_DIR / "symptom_logistic_regression.joblib"),
        },
    )

    phase3_summary = {
        "data_scope": "PCOS pelvic ultrasound proxy dataset mapped to endometriosis-suspected/normal labels for screening support only.",
        "image_branch": {
            "final_train_accuracy": float(history_2["accuracy"][-1]),
            "final_val_accuracy": float(history_2["val_accuracy"][-1]),
            "final_train_loss": float(history_2["loss"][-1]),
            "final_val_loss": float(history_2["val_loss"][-1]),
        },
        "symptom_branch": symptom_stats,
        "fusion_branch": {
            "test_accuracy": fused_accuracy,
            "image_weight": 0.6,
            "symptom_weight": 0.4,
            "threshold": 0.5,
        },
        "artifact_paths": {
            "image_model": str(MODEL_DIR / "efficientnet_b0_model.keras"),
            "symptom_model": str(MODEL_DIR / "symptom_logistic_regression.joblib"),
            "fusion_config": str(MODEL_DIR / "fusion_config.json"),
            "training_plot": str(RESULTS_DIR / "image_training_curves.png"),
        },
    }

    with (RESULTS_DIR / "phase3_training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(phase3_summary, file, indent=2)

    print(json.dumps({
        "image_train_acc": round(float(history_2["accuracy"][-1]), 4),
        "image_val_acc": round(float(history_2["val_accuracy"][-1]), 4),
        "image_train_loss": round(float(history_2["loss"][-1]), 4),
        "image_val_loss": round(float(history_2["val_loss"][-1]), 4),
        "symptom_test_acc": round(float(symptom_stats["test_accuracy"]), 4),
        "fusion_test_acc": round(fused_accuracy, 4),
        "model_dir": str(MODEL_DIR),
        "results_dir": str(RESULTS_DIR),
    }, indent=2))


if __name__ == "__main__":
    main()
