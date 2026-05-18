import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from audio_anomaly.data import build_audio_dataset
from audio_anomaly.evaluation import calculate_classification_metrics
from audio_anomaly.features import FEATURE_COLUMNS
from audio_anomaly.train import train_model


def main():
    base_dir = Path("fan/id_00")
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading audio data from: {base_dir}")
    df = build_audio_dataset(base_dir)

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = calculate_classification_metrics(
        name="RandomForest_SMOTE",
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    cm = confusion_matrix(y_test, y_pred).tolist()

    print("Saving model artifact...")
    joblib.dump(model, artifact_dir / "model.joblib")

    print("Saving metrics...")
    metrics_output = {
        "metrics": metrics,
        "confusion_matrix": cm,
    }

    with open(artifact_dir / "metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2)

    print("Saving model manifest...")
    manifest = {
        "model_name": "fan_anomaly_random_forest_smote",
        "model_version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "asset_scope": "fan/id_00",
        "problem_type": "binary_classification",
        "positive_class": {
            "label": 1,
            "name": "abnormal",
        },
        "negative_class": {
            "label": 0,
            "name": "normal",
        },
        "feature_columns": FEATURE_COLUMNS,
        "model_pipeline": [
            "StandardScaler",
            "SMOTE",
            "RandomForestClassifier",
        ],
        "training_data": str(base_dir),
        "random_state": 42,
        "test_size": 0.2,
        "stratified_split": True,
    }

    with open(artifact_dir / "model_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nTraining complete.")
    print(f"Saved model: {artifact_dir / 'model.joblib'}")
    print(f"Saved metrics: {artifact_dir / 'metrics.json'}")
    print(f"Saved manifest: {artifact_dir / 'model_manifest.json'}")
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Confusion matrix: {cm}")


if __name__ == "__main__":
    main()
