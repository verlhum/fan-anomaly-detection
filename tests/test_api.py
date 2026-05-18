from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_accepts_wav_file():
    file_path = Path("fan/id_00/abnormal/00000000.wav")

    with file_path.open("rb") as audio_file:
        response = client.post(
            "/predict",
            files={"file": ("00000000.wav", audio_file, "audio/wav")},
        )

    assert response.status_code == 200

    result = response.json()

    expected_keys = {
        "file_path",
        "prediction",
        "prediction_label",
        "normal_probability",
        "abnormal_probability",
        "model_metadata",
        "file_name",
        "latency_seconds",
    }

    assert expected_keys.issubset(result.keys())
    assert result["prediction"] in [0, 1]
    assert result["prediction_label"] in ["normal", "abnormal"]
    assert 0.0 <= result["normal_probability"] <= 1.0
    assert 0.0 <= result["abnormal_probability"] <= 1.0
    assert result["file_name"] == "00000000.wav"

    metadata = result["model_metadata"]

    assert metadata["model_name"] is not None
    assert metadata["model_version"] is not None
    assert metadata["asset_scope"] is not None
    assert metadata["problem_type"] == "binary_classification"
    assert metadata["feature_count"] > 0
    
    assert result["latency_seconds"] >= 0


def test_predict_endpoint_rejects_non_wav_file():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not audio", "text/plain")},
    )

    assert response.status_code == 400
