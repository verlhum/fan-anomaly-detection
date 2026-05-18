from pathlib import Path

from audio_anomaly.data import build_audio_dataset


def main():
    base_dir = Path("data/raw/fan/id_00")
    output_path = Path("data/processed/fan_id_00_features.csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_audio_dataset(base_dir)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")
    print(df["label_name"].value_counts())


if __name__ == "__main__":
    main()
