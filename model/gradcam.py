from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"


def preprocess_image_for_model(image_path: str | Path) -> np.ndarray:
    """Load an image and resize it to the EfficientNet input size."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def generate_gradcam_overlay(image_path: str | Path, output_path: str | Path, model_path: str | Path = MODEL_DIR / "efficientnet_b0_model.keras") -> str:
    """Create a Grad-CAM heatmap overlay for a single uploaded ultrasound image."""
    # Future work: the exported EfficientNetB0 image model is a nested submodel, and Keras refuses to
    # build a connected grad_model for it (ValueError: Output with path 0 is not connected to inputs).
    # The workaround is to re-export the image model as a flat functional model whose feature layer is
    # directly connected to the outer input graph before attempting Grad-CAM generation.
    model = tf.keras.models.load_model(model_path)
    input_tensor = model.inputs[0]
    rescale_layer = model.get_layer("rescaling_2")
    base_model = model.get_layer("efficientnetb0")
    nested_conv_layer = base_model.get_layer("top_activation")
    dropout_layer = model.get_layer("dropout")
    output_layer = model.get_layer("image_output")

    x = rescale_layer(input_tensor)
    base_features = base_model(x)
    conv_output = nested_conv_layer(base_features)
    classifier_logits = dropout_layer(base_features)
    predictions = output_layer(classifier_logits)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_output, predictions],
        name="gradcam_model",
    )

    image = preprocess_image_for_model(image_path)
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image)
        score = predictions[:, 0]

    grads = tape.gradient(score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    heatmap = tf.reduce_sum(pooled_grads[..., None] * conv_output, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    heatmap_np = heatmap[0].numpy()
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"Unable to read image at {image_path}")

    resized_heatmap = cv2.resize(heatmap_np, (original.shape[1], original.shape[0]))
    resized_heatmap = (resized_heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.75, heatmap_color, 0.35, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    return str(output_path)


def create_sample_gradcam_outputs(sample_paths: list[str | Path], output_dir: str | Path = RESULTS_DIR) -> list[str]:
    """Generate a few sample Grad-CAM overlays for report-ready visuals."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for i, sample_path in enumerate(sample_paths[:3], start=1):
        out_path = output_dir / f"gradcam_sample_{i}.png"
        generate_gradcam_overlay(sample_path, out_path, MODEL_DIR / "efficientnet_b0_model.keras")
        generated.append(str(out_path))
    return generated


# Phase 5 note:
# The image branch achieved high precision but recall of 0.70 on the test split. Grad-CAM outputs
# for false-negative cases (missed positives) would be a valuable addition to /results to show what
# the model is under-weighting and to guide future improvements on the proxy dataset.
