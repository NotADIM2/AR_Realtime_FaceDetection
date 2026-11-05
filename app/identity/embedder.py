"""Converts a face into a numeric template using the SFace model.

Recognition has two image-processing stages:

1. Alignment: a similarity transform (rotation, uniform scale and translation) maps
   five facial keypoints onto fixed positions in a 112x112 image. This removes most
   differences caused by head position, tilt and distance from the camera.
2. Embedding: a neural network turns the aligned image into 128 numbers. Images of
   the same person produce vectors that point in similar directions.

Two templates are compared with cosine similarity. For unit vectors, this is simply
their dot product:

    similarity = (a . b) / (|a| |b|) = a . b
"""

import cv2
import numpy as np

from app import config
from app.vision import landmarks as lm
from app.vision.detector import verify_model

KEYPOINTS = (lm.IRIS_IMAGE_LEFT, lm.IRIS_IMAGE_RIGHT, lm.NOSE_TIP, lm.MOUTH_IMAGE_LEFT, lm.MOUTH_IMAGE_RIGHT)


def normalize(vector):
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


class FaceEmbedder:
    def __init__(self, model_path=config.RECOGNITION_MODEL_PATH, expected_sha256=config.RECOGNITION_MODEL_SHA256):
        verify_model(model_path, expected_sha256)
        self._recognizer = cv2.FaceRecognizerSF.create(str(model_path), "")

    def align(self, frame, points):
        """Return the 112x112 aligned face image."""
        # OpenCV reads the keypoints from columns 4-13 of a face detection row, ordered
        # image-left eye, image-right eye, nose, image-left mouth, image-right mouth.
        face_row = np.zeros((1, 15), np.float32)
        face_row[0, 4:14] = points[list(KEYPOINTS), :2].ravel()
        return self._recognizer.alignCrop(frame, face_row)

    def embed(self, frame, points):
        """Return the face's template as a 128-value unit vector."""
        return normalize(self._recognizer.feature(self.align(frame, points)).ravel().astype(np.float64))
