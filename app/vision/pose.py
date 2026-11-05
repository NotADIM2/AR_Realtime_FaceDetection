"""Builds a coordinate frame attached to the face.

Three orthonormal axes are derived from the 3D landmarks:

    X = normalize(outer eye right - outer eye left)
    Y = normalize(v - (v . X) X)    where v = chin - forehead (Gram-Schmidt)
    Z = X x Y

Together these form the rotation matrix R = [X Y Z]. The distance between the
outer eye corners gives the face scale in pixels. Because it is measured in 3D
it barely changes as the head turns, whereas the 2D distance would shrink.
"""

from dataclasses import dataclass

import numpy as np

from app import config
from app.vision import landmarks as lm


@dataclass
class FacePose:
    points: np.ndarray  # (478, 3) landmarks in pixels
    rotation: np.ndarray  # 3x3 matrix whose columns are the face axes X, Y, Z
    scale: float  # Distance between the outer eye corners, in pixels
    yaw: float  # Degrees
    pitch: float  # Degrees
    roll: float  # Degrees

    def project(self, anchor_index, offsets):
        """Project face-space offsets to image coordinates.

        offsets is (N, 3) in eye-width units. A point's depth along the camera
        axis shrinks or enlarges it, which produces perspective foreshortening:

            P     = s * R * offset
            k     = d / (d + P_z / s)    where d = PERSPECTIVE_DISTANCE
            p_2d  = A_2d + k * P_xy
        """
        anchor = self.points[anchor_index]
        relative = np.asarray(offsets, dtype=np.float32) @ self.rotation.T.astype(np.float32)
        relative *= self.scale
        depth = relative[:, 2] / self.scale
        perspective = config.PERSPECTIVE_DISTANCE / (config.PERSPECTIVE_DISTANCE + depth)
        return anchor[:2] + relative[:, :2] * perspective[:, None]


def _normalize(vector):
    return vector / max(float(np.linalg.norm(vector)), 1e-6)


def estimate_pose(points):
    """Compute a FacePose from (478, 3) pixel landmarks."""
    eye_vector = points[lm.EYE_OUTER_IMAGE_RIGHT] - points[lm.EYE_OUTER_IMAGE_LEFT]
    scale = float(np.linalg.norm(eye_vector))

    x_axis = _normalize(eye_vector)
    down = points[lm.CHIN] - points[lm.FOREHEAD_TOP]
    y_axis = _normalize(down - np.dot(down, x_axis) * x_axis)
    z_axis = np.cross(x_axis, y_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))

    roll = np.degrees(np.arctan2(x_axis[1], x_axis[0]))
    yaw = np.degrees(np.arcsin(np.clip(x_axis[2], -1.0, 1.0)))
    pitch = np.degrees(np.arcsin(np.clip(y_axis[2], -1.0, 1.0)))
    return FacePose(points, rotation, scale, float(yaw), float(pitch), float(roll))
