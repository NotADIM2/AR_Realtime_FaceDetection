"""The per-frame image processing pipeline."""

from dataclasses import dataclass

import cv2

from app.filters.drawn import draw_axes, draw_landmarks
from app.imaging.transforms import normalized_to_pixels
from app.vision.pose import estimate_pose
from app.vision.tracker import FaceTracker


@dataclass
class FrameInfo:
    faces: int = 0
    tracking: bool = False
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


class FrameProcessor:
    """Runs one frame through detection, tracking, pose estimation and filtering."""

    def __init__(self, detector):
        self._detector = detector
        self._tracker = FaceTracker()

    def process(self, frame, time_s, active_filter, show_landmarks=False, show_axes=False, identity_session=None):
        # Mirror first, like a selfie camera. Detecting on the mirrored frame means
        # the landmarks and the displayed image always share the same coordinates.
        output = cv2.flip(frame, 1)
        height, width = output.shape[:2]

        # MediaPipe expects RGB, while OpenCV stores frames as BGR.
        rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        detections = self._detector.detect(rgb, time_s * 1000.0)
        faces = [normalized_to_pixels(face, width, height) for face in detections]

        points = self._tracker.update(faces, time_s)
        pose = estimate_pose(points) if points is not None else None

        # Recognition must see the real face, so it runs before any filter is drawn.
        if identity_session is not None:
            identity_session.offer(output, pose, len(faces), time_s)

        active_filter.apply(output, pose)
        if pose is not None:
            if show_landmarks:
                draw_landmarks(output, pose)
            if show_axes:
                draw_axes(output, pose)

        info = FrameInfo(faces=len(faces), tracking=pose is not None)
        if pose is not None:
            info.yaw, info.pitch, info.roll = pose.yaw, pose.pitch, pose.roll
        return output, info
