import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from audio_anomaly.features import FEATURE_COLUMNS, extract_features_array


LABEL_MAP = {
    0: "normal",
    1: "abnormal",
}


def load_model(model_path: str | Path = "artifacts/model.joblib") -> Any:
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. "
            "Run scripts/train_model.py first."
        )

    return joblib.load(model_path)


def load_model_manifest(manifest_path: str | Path = "artifacts/model_manifest.json") -> dict:
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Model manifest not found: {manifest_path}. "
            "Run scripts/train_model.py first."
        )

    with manifest_path.open("r") as f:
        return json.load(f)


def predict_file(
    file_path: str | Path,
    model_path: str | Path = "artifacts/model.joblib",
    manifest_path: str | Path = "artifacts/model_manifest.json",
) -> dict:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    model = load_model(model_path)
    manifest = load_model_manifest(manifest_path)

    features = extract_features_array(str(file_path)).reshape(1, -1)
    features_df = pd.DataFrame(features, columns=FEATURE_COLUMNS)

    prediction = int(model.predict(features_df)[0])
    probabilities = model.predict_proba(features_df)[0]

    return {
        "file_path": str(file_path),
        "prediction": prediction,
        "prediction_label": LABEL_MAP[prediction],
        "normal_probability": float(probabilities[0]),
        "abnormal_probability": float(probabilities[1]),
        "model_metadata": {
            "model_name": manifest.get("model_name"),
            "model_version": manifest.get("model_version"),
            "asset_scope": manifest.get("asset_scope"),
            "problem_type": manifest.get("problem_type"),
            "feature_count": len(manifest.get("feature_columns", [])),
            "positive_class": manifest.get("positive_class"),
        },
    }
