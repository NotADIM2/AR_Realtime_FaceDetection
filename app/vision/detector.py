"""Wrapper around the MediaPipe Face Landmarker."""

import hashlib

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from app import config


class ModelError(RuntimeError):
    """The face model file is missing or damaged."""


def verify_model(path=config.MODEL_PATH, expected_sha256=config.MODEL_SHA256):
    if not path.is_file():
        raise ModelError(f"Face model not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ModelError(f"Face model failed its checksum and may be damaged: {path}")


class FaceDetector:
    """Finds faces in RGB frames. Must be used from a single thread."""

    def __init__(self, model_path=config.MODEL_PATH, max_faces=config.MAX_FACES):
        verify_model(model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp = -1

    def detect(self, rgb, timestamp_ms):
        """Return one (478, 3) array of normalized landmarks per detected face."""
        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(int(timestamp_ms), self._last_timestamp + 1)
        self._last_timestamp = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        return [
            np.array([(p.x, p.y, p.z) for p in face], dtype=np.float32)
            for face in result.face_landmarks
        ]

    def close(self):
        self._landmarker.close()
