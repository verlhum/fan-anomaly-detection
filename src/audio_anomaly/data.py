from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from audio_anomaly.features import extract_features_dict


LABEL_MAP = {
    "normal": 0,
    "abnormal": 1,
}


def build_audio_dataset(
    base_dir: str | Path,
    categories: Iterable[str] = ("normal", "abnormal"),
) -> pd.DataFrame:
    """
    Build a labeled feature dataframe from a MIMII-style directory.

    Expected directory structure:
        base_dir/
            normal/
                *.wav
            abnormal/
                *.wav

    Returns one row per audio file with:
        file_path, file_name, label_name, label, features...
    """
    base_dir = Path(base_dir)
    rows = []
    errors = []

    for label_name in categories:
        folder = base_dir / label_name

        if label_name not in LABEL_MAP:
            raise ValueError(f"Unknown label category: {label_name}")

        if not folder.exists():
            raise FileNotFoundError(f"Expected folder not found: {folder}")

        wav_files = sorted(folder.glob("*.wav"))

        if not wav_files:
            raise FileNotFoundError(f"No .wav files found in {folder}")

        for file_path in tqdm(wav_files, desc=f"Processing {label_name}"):
            try:
                features = extract_features_dict(str(file_path))

                rows.append({
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "label_name": label_name,
                    "label": LABEL_MAP[label_name],
                    **features,
                })

            except Exception as exc:
                errors.append({
                    "file_path": str(file_path),
                    "error": str(exc),
                })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No valid audio files were processed.")

    if errors:
        print(f"Skipped {len(errors)} files due to feature extraction errors.")

    return df
