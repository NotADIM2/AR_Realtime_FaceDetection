"""Primary-face selection and temporal smoothing.

Multi-face policy: the filter is applied to one primary face. The largest face
is chosen first and stays primary while it is visible, so the filter doesn't
jump between people. It only moves if another face becomes clearly larger
(config.PRIMARY_SWITCH_RATIO) or the primary face leaves.
"""

import numpy as np

from app import config
from app.vision.smoothing import OneEuroFilter


def _center_and_area(points):
    low = points[:, :2].min(axis=0)
    high = points[:, :2].max(axis=0)
    return (low + high) / 2.0, float(np.prod(high - low))


def choose_primary(faces, previous_center):
    """Return the index of the primary face. faces holds pixel landmark arrays."""
    measures = [_center_and_area(face) for face in faces]
    largest = max(range(len(faces)), key=lambda i: measures[i][1])
    if previous_center is None:
        return largest

    # The face nearest the previous position is treated as the same person.
    current = min(
        range(len(faces)), key=lambda i: np.linalg.norm(measures[i][0] - previous_center)
    )
    if measures[largest][1] > measures[current][1] * config.PRIMARY_SWITCH_RATIO:
        return largest
    return current


class FaceTracker:
    """Turns raw detections into one smoothed primary face per frame."""

    def __init__(self):
        self._filter = OneEuroFilter(
            config.SMOOTH_MIN_CUTOFF, config.SMOOTH_BETA, config.SMOOTH_D_CUTOFF
        )
        self.reset()

    def reset(self):
        self._filter.reset()
        self._center = None
        self._last_points = None
        self._missed = 0

    def update(self, faces, time_s):
        """Return smoothed (478, 3) pixel landmarks, or None when no face is available."""
        if not faces:
            self._missed += 1
            if self._last_points is not None and self._missed <= config.FACE_HOLD_FRAMES:
                return self._last_points
            self.reset()
            return None

        index = choose_primary(faces, self._center)
        points = faces[index]
        center, area = _center_and_area(points)

        # A large jump means a different face, so old positions must not be blended in.
        if self._center is not None and np.linalg.norm(center - self._center) > np.sqrt(area):
            self._filter.reset()

        self._center = center
        self._missed = 0
        self._last_points = self._filter(points, time_s)
        return self._last_points
