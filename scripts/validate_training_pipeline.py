from pathlib import Path

from sklearn.model_selection import train_test_split

from audio_anomaly.data import build_audio_dataset
from audio_anomaly.features import FEATURE_COLUMNS
from audio_anomaly.train import train_model
from audio_anomaly.evaluation import calculate_classification_metrics


def main():
    base_dir = Path("fan/id_00") 

    print(f"Loading audio data from: {base_dir}")
    df = build_audio_dataset(base_dir)

    print("\nDataset loaded.")
    print(f"Rows: {len(df)}")
    print("Class counts:")
    print(df["label_name"].value_counts())

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing expected feature columns: {missing_features}")

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    print("\nFeature matrix:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print("\nTraining model...")
    model = train_model(X_train, y_train)

    print("Generating predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = calculate_classification_metrics(
        name="RandomForest_SMOTE",
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    print("\nSmoke test passed.")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
