"""Landmark indices and mesh connections from the MediaPipe face model."""

import numpy as np
from mediapipe.tasks.python.vision import FaceLandmarksConnections as _Connections

LANDMARK_COUNT = 478

EYE_OUTER_IMAGE_LEFT = 33
EYE_OUTER_IMAGE_RIGHT = 263
FOREHEAD_TOP = 10
CHIN = 152
NOSE_BRIDGE = 168
NOSE_TIP = 1
PHILTRUM = 164
IRIS_IMAGE_LEFT = 468
IRIS_IMAGE_RIGHT = 473
MOUTH_IMAGE_LEFT = 61
MOUTH_IMAGE_RIGHT = 291


def _unique_segments(connections):
    """Convert connections to an (N, 2) index array, dropping repeated edges."""
    pairs = {tuple(sorted((c.start, c.end))) for c in connections}
    return np.array(sorted(pairs), dtype=np.int32)


MESH_SEGMENTS = _unique_segments(_Connections.FACE_LANDMARKS_TESSELATION)
# The contours set already includes the face oval, eyes, eyebrows and lips.
OUTLINE_SEGMENTS = _unique_segments(_Connections.FACE_LANDMARKS_CONTOURS)
