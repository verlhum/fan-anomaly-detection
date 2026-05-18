import librosa
import numpy as np

FEATURE_COLUMNS = [
    *[f"mfcc_{i}_mean" for i in range(1, 14)],
    "zero_crossing_rate_mean",
    "spectral_centroid_mean",
]


def extract_features_array(file_path: str) -> np.ndarray:
    """
    Extract features from a single audio file as a NumPy array.
    Feature order matches FEATURE_COLUMNS.
    """
    y, sr = librosa.load(file_path, sr=None)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    return np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(zcr),
        np.mean(centroid),
    ])

def extract_features_dict(file_path: str) -> dict:
    """
    Extract features from a single audio file as a dictionary.
    Useful for logging, debugging, and dataframe construction.
    """
    features = extract_features_array(file_path)
    return dict(zip(FEATURE_COLUMNS, features))
