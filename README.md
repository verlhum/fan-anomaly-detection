# Fan Audio Anomaly Detection

This project detects abnormal industrial fan sounds using supervised machine learning on audio features extracted from the MIMII Dataset.

The project started as an exploratory notebook analysis and was later refactored into a deployable ML scoring service using a trained model artifact, FastAPI, and Docker.

## Project Overview

- Dataset: MIMII fan 0 dB subset
- Task: Binary classification: normal vs. abnormal fan audio
- Input: 10-second `.wav` audio clips
- Features: MFCC means, zero-crossing rate, and spectral centroid
- Model: Random Forest classifier with SMOTE class balancing
- Deployment: FastAPI service containerized with Docker

## Why This Project Matters

This project demonstrates the full path from model development to deployment:

```text
raw audio files
→ feature extraction
→ model training
→ saved model artifact
→ single-file inference
→ FastAPI prediction endpoint
→ Dockerized deployment
```

The goal is not only to build a model, but to package it so it can be reused as a scoring service.

## Repository Structure

```text
.
├── app/
│   └── main.py                         # FastAPI application
│
├── artifacts/
│   ├── metrics.json                    # Saved model evaluation metrics
│   ├── model_manifest.json             # Model metadata and feature contract
│   └── splits/
│       ├── train_files.csv             # Training split file list
│       └── test_files.csv              # Test split file list
│
├── scripts/
│   ├── build_dataset.py                # Build feature dataset from audio files
│   ├── train_model.py                  # Train model and save artifacts
│   ├── validate_training_pipeline.py   # End-to-end training validation
│   ├── score_file.py                   # Score one audio file
│   └── score_batch.py                  # Score multiple audio files
│
├── src/
│   └── audio_anomaly/
│       ├── config.py
│       ├── data.py                     # Dataset construction
│       ├── evaluation.py               # Classification metrics
│       ├── features.py                 # Audio feature extraction
│       ├── inference.py                # Load model and score audio
│       └── train.py                    # Model training pipeline
│
├── tests/
├── docs/
├── Dockerfile
├── requirements.txt
├── README.md
└── Precision_Audio_Anomaly_Detection_Pipeline.ipynb
```

## Exploratory Notebook

The original notebook is kept in the repository root to document the exploratory analysis, feature development, model comparison, and evaluation process.

Notebook:

```text
Precision_Audio_Anomaly_Detection_Pipeline.ipynb
```

The production-style code lives in `src/`, `scripts/`, and `app/`.

## Model Performance

The selected model is a Random Forest classifier trained with SMOTE balancing.

Recent training run:

| Metric | Value |
|---|---:|
| Accuracy | 0.898 |
| Precision | 0.853 |
| Recall | 0.780 |
| F1 Score | 0.815 |
| ROC AUC | 0.945 |

These metrics are based on a stratified train/test split of the `fan/id_00` subset.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/verlhum/fan-anomaly-detection.git
cd fan-anomaly-detection
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Data Setup

This repository does not include the raw MIMII audio files.

Expected local structure:

```text
fan/
└── id_00/
    ├── normal/
    │   └── *.wav
    └── abnormal/
        └── *.wav
```

The scripts assume this path by default:

```text
fan/id_00
```

## Validate the Training Pipeline

Run the end-to-end validation script:

```bash
PYTHONPATH=src python scripts/validate_training_pipeline.py
```

This checks that the project can:

```text
load raw audio
extract features
build the feature matrix
train the model pipeline
generate predictions
calculate metrics
```

## Train the Model

```bash
PYTHONPATH=src python scripts/train_model.py
```

This creates or updates:

```text
artifacts/model.joblib
artifacts/metrics.json
artifacts/model_manifest.json
artifacts/splits/train_files.csv
artifacts/splits/test_files.csv
```

The model artifact contains the trained scikit-learn / imbalanced-learn pipeline.

## Score One Audio File

```bash
PYTHONPATH=src python scripts/score_file.py fan/id_00/abnormal/00000000.wav
```

Example response:

```json
{
  "file_path": "fan/id_00/abnormal/00000000.wav",
  "prediction": 1,
  "prediction_label": "abnormal",
  "normal_probability": 0.04,
  "abnormal_probability": 0.96
}
```

## Run the FastAPI Service Locally

```bash
PYTHONPATH=src uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

Use the `/predict` endpoint to upload a `.wav` file and receive a prediction.

### Health check

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### Prediction request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@fan/id_00/abnormal/00000000.wav"
```

## Run with Docker

Build the image:

```bash
docker build -t fan-anomaly-api .
```

Run the container:

```bash
docker run -p 8000:8000 fan-anomaly-api
```

If port `8000` is already in use:

```bash
docker run -p 8001:8000 fan-anomaly-api
```

Then test:

```bash
curl http://127.0.0.1:8001/health
```

And score a file:

```bash
curl -X POST "http://127.0.0.1:8001/predict" \
  -F "file=@fan/id_00/abnormal/00000000.wav"
```

## Model Artifact Notes

The trained model is saved as:

```text
artifacts/model.joblib
```

Depending on repository settings, this file may be excluded from Git. If it is not present, recreate it with:

```bash
PYTHONPATH=src python scripts/train_model.py
```

The model manifest records the expected feature columns, class labels, model version, data scope, and train/test split information.

## Limitations

- The current model is trained on the `fan/id_00` subset.
- Performance may not generalize to other fan IDs without recalibration or retraining.
- The model uses summary-level audio features rather than deep audio embeddings.
- The API currently supports `.wav` file uploads only.
- The project is intended as a deployment-focused ML portfolio project, not a production monitoring system.

## Future Improvements

- Add automated tests for feature extraction and inference.
- Add batch scoring endpoint.
- Add model version to API responses.
- Add logging for prediction requests.
- Add monitoring checks for feature drift and prediction distribution shifts.
- Compare generalization across additional MIMII fan IDs.
- Add CI workflow for linting and tests.
