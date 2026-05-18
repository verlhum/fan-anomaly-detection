from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile

from audio_anomaly.inference import predict_file


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
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Only .wav files are supported.",
        )

    try:
        # Save uploaded file temporarily so librosa can read it from disk.
        with NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file.flush()

            prediction = predict_file(temp_file.name)

        # Preserve original uploaded filename in API response.
        prediction["file_name"] = file.filename

        return prediction

    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )
