from pathlib import Path

import numpy as np

from audio_anomaly.features import FEATURE_COLUMNS, extract_features_array, extract_features_dict


def test_extract_features_array_returns_expected_shape():
    file_path = Path("fan/id_00/normal/00000000.wav")

    features = extract_features_array(file_path)

    assert isinstance(features, np.ndarray)
    assert features.shape == (len(FEATURE_COLUMNS),)
    assert np.all(np.isfinite(features))


def test_extract_features_dict_returns_expected_keys():
    file_path = Path("fan/id_00/normal/00000000.wav")

    features = extract_features_dict(file_path)

    assert set(features.keys()) == set(FEATURE_COLUMNS)
    assert all(np.isfinite(value) for value in features.values())
