# Phase 2: PCOS pelvic ultrasound proxy dataset pipeline

This project uses a same-organ-system, same-modality proxy for the image branch:

- Primary target: endometriosis screening support
- Proxy dataset: PCOS pelvic ultrasound images
- Rationale: PCOS and endometriosis are both gynecological conditions commonly imaged via pelvic ultrasound, and they share overlapping symptom profiles such as pelvic pain, cycle irregularity, and discomfort during intercourse.
- Mapping: the dataset labels `infected` and `not_infected` are relabeled to `1 = endometriosis suspected` and `0 = normal` for project framing, with a clear note that these are proxy labels and not verified clinical endometriosis cases.

## Download the PCOS ultrasound dataset from Kaggle

Use one of the following methods.

### Option A: Kaggle API (recommended)

1. Create a Kaggle account and generate an API token.
2. Download the token file `kaggle.json` and place it in `~/.kaggle/` on your machine.
3. Install Kaggle CLI:

```bash
pip install kaggle
```

4. Download the dataset into the project’s expected raw structure:

```bash
mkdir -p data/raw/pcos_ultrasound
kaggle datasets download -d <kaggle-user>/<dataset-name> -p data/raw
```

5. Extract the archive and move or rename the images so the structure matches:

```text
data/raw/pcos_ultrasound/
├── infected/
│   ├── image_001.jpg
│   └── ...
└── not_infected/
    ├── image_010.jpg
    └── ...
```

### Option B: KaggleHub helper

```bash
pip install kagglehub
python - <<'PY'
import kagglehub
path = kagglehub.dataset_download("<kaggle-user>/<dataset-name>")
print(path)
PY
```

Then copy the resulting image folders into:

```text
data/raw/pcos_ultrasound/
```

### Important note

The expected dataset name is the public PCOS ultrasound dataset commonly listed as:

- `PCOS detection using ultrasound images`

The project expects a folder structure with `infected/` and `not_infected/` directories so that the script in [data/prepare_pcos_proxy_dataset.py](data/prepare_pcos_proxy_dataset.py) can map them to the binary endometriosis-suspected/normal labels used in this project.

If the dataset is not yet downloaded, the pipeline will not fabricate results. It will stop with a clear error stating that the raw image folders are missing.

## Required raw folder layout

The pipeline expects the raw images under:

```text
data/raw/pcos_ultrasound/
├── infected/
│   ├── image_001.jpg
│   └── ...
└── not_infected/
    ├── image_010.jpg
    └── ...
```

The project does not claim that these are verified endometriosis cases; they are a same-organ-system, same-modality proxy for a gynecological ultrasound screening prototype.

## Symptom feature schema (simulated for this stage)

The symptom dataset is intentionally simulated and will be replaced by real clinical symptom data if a proper source becomes available.

Fields used:

- pain_level
- cycle_irregularity
- pain_during_intercourse
- pelvic_pressure
- heavy_bleeding
- infertility_history
- fatigue
- label
- is_simulated

This is a minimal, clinically plausible feature set for a final-year multimodal screening prototype.

## Class imbalance handling

The dataset pipeline records counts per class and split so that class imbalance can be tracked explicitly. For later model training, this project will use class-weighting or balanced sampling if the positive class is underrepresented.

## Preprocessing and augmentation

The processing pipeline includes:

- resize to 224x224
- normalize pixel intensities to [0, 1]
- train-only augmentation using rotation, horizontal/vertical flips, and brightness jitter
- stratified 70/15/15 split by label

## Important limitation

This is not a clinically verified endometriosis dataset. It is a defensible gynecological ultrasound proxy chosen for same-organ-system relevance, and the project documents that limitation explicitly.
