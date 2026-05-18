import argparse
import json

from audio_anomaly.inference import predict_file


def main():
    parser = argparse.ArgumentParser(description="Score one audio file with the trained anomaly model.")
    parser.add_argument("file_path", help="Path to a .wav file")
    parser.add_argument(
        "--model-path",
        default="artifacts/model.joblib",
        help="Path to trained model artifact",
    )

    args = parser.parse_args()

    result = predict_file(
        file_path=args.file_path,
        model_path=args.model_path,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
