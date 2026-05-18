from pathlib import Path

from audio_anomaly.inference import predict_file


def test_predict_file_returns_expected_fields():
    file_path = Path("fan/id_00/abnormal/00000000.wav")

    result = predict_file(file_path)

    expected_keys = {
        "file_path",
        "prediction",
        "prediction_label",
        "normal_probability",
        "abnormal_probability",
    }

    assert expected_keys.issubset(result.keys())
    assert result["prediction"] in [0, 1]
    assert result["prediction_label"] in ["normal", "abnormal"]
    assert 0.0 <= result["normal_probability"] <= 1.0
    assert 0.0 <= result["abnormal_probability"] <= 1.0
