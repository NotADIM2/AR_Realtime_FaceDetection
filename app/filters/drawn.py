"""Filters and debug layers drawn directly from landmark data."""

import cv2
import numpy as np

from app.imaging.transforms import clipped_bounds
from app.vision import landmarks as lm

_SHIFT = 4  # OpenCV fixed-point bits, giving sub-pixel line positions.
_FIXED = 1 << _SHIFT


def _segments_in_roi(pose, segments, origin):
    points = np.round((pose.points[:, :2] - origin) * _FIXED).astype(np.int32)
    return list(points[segments])


def _face_roi(frame, pose, padding):
    height, width = frame.shape[:2]
    return clipped_bounds(pose.points, width, height, padding=padding)


class NormalFilter:
    """Plain camera, with no effect."""

    key = "normal"
    label = "Normal"
    needs_face = False
    problem = None

    def apply(self, frame, pose):
        pass


class MeshFilter:
    """Draws the triangulated face mesh, semi-transparent over the face."""

    key = "mesh"
    label = "Face Mesh"
    needs_face = True
    problem = None

    def apply(self, frame, pose):
        if pose is None:
            return
        bounds = _face_roi(frame, pose, padding=2)
        if bounds is None:
            return
        x0, y0, x1, y1 = bounds
        roi = frame[y0:y1, x0:x1]
        layer = roi.copy()
        segments = _segments_in_roi(pose, lm.MESH_SEGMENTS, (x0, y0))
        cv2.polylines(layer, segments, False, (235, 220, 90), 1, cv2.LINE_AA, _SHIFT)
        cv2.addWeighted(layer, 0.85, roi, 0.15, 0, dst=roi)


class OutlineFilter:
    """Draws glowing contours for the face, eyes, eyebrows and lips."""

    key = "outline"
    label = "Face Outline"
    needs_face = True
    problem = None

    def apply(self, frame, pose):
        if pose is None:
            return
        bounds = _face_roi(frame, pose, padding=24)
        if bounds is None:
            return
        x0, y0, x1, y1 = bounds
        roi = frame[y0:y1, x0:x1]
        contours = _segments_in_roi(pose, lm.OUTLINE_SEGMENTS, (x0, y0))

        # Glow: blur a copy of the lines, then add a fraction of it to the image.
        glow = np.zeros_like(roi)
        cv2.polylines(glow, contours, False, (255, 215, 60), 3, cv2.LINE_AA, _SHIFT)
        glow = cv2.GaussianBlur(glow, (0, 0), max(pose.scale * 0.025, 1.5))
        cv2.addWeighted(roi, 1.0, glow, 0.8, 0, dst=roi)
        cv2.polylines(roi, contours, False, (255, 235, 150), 1, cv2.LINE_AA, _SHIFT)


def draw_landmarks(frame, pose):
    """Draw every landmark as a small dot."""
    points = np.round(pose.points[:, :2] * _FIXED).astype(np.int32)
    for point in points:
        cv2.circle(frame, tuple(int(v) for v in point), 1 * _FIXED, (120, 255, 120), -1, cv2.LINE_AA, _SHIFT)


def draw_axes(frame, pose):
    """Draw the face coordinate axes at the nose tip: X red, Y green, Z blue."""
    ends = pose.project(lm.NOSE_TIP, np.array([(0.5, 0, 0), (0, 0.5, 0), (0, 0, -0.5)], np.float32))
    origin = tuple(int(v) for v in pose.points[lm.NOSE_TIP][:2])
    for end, color in zip(ends, ((60, 60, 255), (90, 230, 90), (255, 150, 60))):
        cv2.arrowedLine(frame, origin, tuple(int(v) for v in end), color, 2, cv2.LINE_AA, tipLength=0.15)
