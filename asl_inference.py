"""Load the submitted ASL model for image or webcam inference without training."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv
import keras
import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / "models" / "cnn_for_asl_grayscale.h5"
LABELS = [
    "A", "B", "C", "D", "del", "E", "F", "G", "H", "I", "J", "K", "L",
    "M", "N", "nothing", "O", "P", "Q", "R", "S", "space", "T", "U", "V",
    "W", "X", "Y", "Z",
]
IMAGE_SIZE = (200, 200)


def load_submitted_model(model_path: Path = MODEL_PATH):
    """Load the published model without compiling or training it."""
    model = keras.models.load_model(model_path, compile=False)
    if model.input_shape != (None, 200, 200, 1) or model.output_shape != (None, 29):
        raise ValueError(
            f"Unexpected model shapes: input={model.input_shape}, output={model.output_shape}"
        )
    return model


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Match the notebook: grayscale uint8, 200x200, one channel, no normalization."""
    if frame is None:
        raise ValueError("The image could not be read")
    if frame.ndim == 3:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.resize(frame, IMAGE_SIZE)
    return frame.astype(np.uint8).reshape(1, 200, 200, 1)


def predict_label(model, frame: np.ndarray) -> tuple[str, np.ndarray]:
    probabilities = model.predict(preprocess_frame(frame), verbose=0)[0]
    return LABELS[int(np.argmax(probabilities))], probabilities


def run_webcam(model) -> None:
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("No webcam could be opened")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The webcam stopped returning frames")
            height, width = frame.shape[:2]
            x1, y1 = max(0, width // 2 - 100), max(0, height // 2 - 100)
            roi = frame[y1 : y1 + 200, x1 : x1 + 200]
            label, _ = predict_label(model, roi)
            cv.rectangle(frame, (x1, y1), (x1 + 200, y1 + 200), (252, 19, 3), 2)
            cv.putText(frame, f"Predicted label: {label}", (10, height - 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv.imshow("ASL model", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=Path, help="Predict one image")
    mode.add_argument("--webcam", action="store_true", help="Start webcam inference")
    mode.add_argument("--smoke-test", action="store_true", help="Run one zero-image prediction")
    args = parser.parse_args()
    model = load_submitted_model()
    if args.webcam:
        run_webcam(model)
    else:
        image = np.zeros(IMAGE_SIZE, dtype=np.uint8) if args.smoke_test else cv.imread(str(args.image))
        label, probabilities = predict_label(model, image)
        print(f"label={label} output_shape={probabilities.shape}")


if __name__ == "__main__":
    main()
