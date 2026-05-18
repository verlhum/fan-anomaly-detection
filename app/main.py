import logging
import time
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile

from audio_anomaly.inference import predict_file


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Fan Audio Anomaly Detection API",
    description="Scores fan audio files as normal or abnormal using a trained ML model.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    """
    Score one uploaded .wav file.
    """
    start_time = time.perf_counter()

    if not file.filename.lower().endswith(".wav"):
        latency_seconds = time.perf_counter() - start_time

        logger.warning(
            "prediction_request status=rejected reason=unsupported_file_type "
            "file_name=%s latency_seconds=%.4f",
            file.filename,
            latency_seconds,
        )

        raise HTTPException(
            status_code=400,
            detail="Only .wav files are supported.",
        )

    try:
        with NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file.flush()

            prediction = predict_file(temp_file.name)

        latency_seconds = time.perf_counter() - start_time

        prediction["file_name"] = file.filename
        prediction["latency_seconds"] = round(latency_seconds, 4)

        model_metadata = prediction.get("model_metadata", {})

        logger.info(
            "prediction_request status=success file_name=%s prediction_label=%s "
            "abnormal_probability=%.4f model_version=%s asset_scope=%s "
            "latency_seconds=%.4f",
            file.filename,
            prediction.get("prediction_label"),
            prediction.get("abnormal_probability"),
            model_metadata.get("model_version"),
            model_metadata.get("asset_scope"),
            latency_seconds,
        )

        return prediction

    except FileNotFoundError as exc:
        latency_seconds = time.perf_counter() - start_time

        logger.exception(
            "prediction_request status=error error_type=file_not_found "
            "file_name=%s latency_seconds=%.4f",
            file.filename,
            latency_seconds,
        )

        raise HTTPException(status_code=500, detail=str(exc))

    except Exception as exc:
        latency_seconds = time.perf_counter() - start_time

        logger.exception(
            "prediction_request status=error error_type=prediction_failed "
            "file_name=%s latency_seconds=%.4f",
            file.filename,
            latency_seconds,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )
