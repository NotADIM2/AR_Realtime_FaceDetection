"""A single Register or Scan operation, which collects face samples over a few seconds.

Several samples are averaged because one frame can be affected by blur, expression or
lighting. Sessions start only when the user clicks a button, so the app never tries to
identify anyone in the background.
"""

import cv2
import numpy as np

from app import config
from app.identity.embedder import KEYPOINTS, normalize
from app.identity.store import DuplicateProfileError

ENROLL = "enroll"
SCAN = "scan"

# Slight head turns give the saved template more than one viewing angle.
_ENROLL_PROMPTS = (
    "Look straight at the camera",
    "Look straight at the camera",
    "Turn your head slightly to one side",
    "Turn your head slightly to one side",
    "Turn your head slightly to the other side",
    "Turn your head slightly to the other side",
    "Look straight at the camera",
    "Look straight at the camera",
)
_SCAN_PROMPT = "Look at the camera and hold still"


def sample_problem(mode, frame, pose, face_count):
    """Explain why this frame can't be used as a sample. Returns None if it can."""
    if pose is None or face_count == 0:
        return "No face in view"
    if mode == ENROLL and face_count > 1:
        return "Only one person should be in view"
    if pose.scale < config.MIN_SAMPLE_EYE_WIDTH_PX:
        return "Move closer to the camera"
    if max(abs(pose.yaw), abs(pose.pitch)) > config.MAX_SAMPLE_ANGLE_DEG:
        return "Face the camera more directly"
    height, width = frame.shape[:2]
    keypoints = pose.points[list(KEYPOINTS), :2]
    if ((keypoints < 0) | (keypoints >= (width, height))).any():
        return "Move your whole face into the frame"
    return None


class CaptureSession:
    def __init__(self, mode, embedder, now, name=None):
        self.mode = mode
        self.name = name
        self.total = config.ENROLL_SAMPLES if mode == ENROLL else config.SCAN_SAMPLES
        self._embedder = embedder
        self._samples = []
        self._deadline = now + config.SESSION_TIMEOUT_S
        self._next_sample_time = now
        self._problem = None

    def _prompt(self):
        if self._problem:
            return self._problem
        if self.mode == SCAN:
            return _SCAN_PROMPT
        return _ENROLL_PROMPTS[min(len(self._samples), len(_ENROLL_PROMPTS) - 1)]

    def offer(self, frame, pose, face_count, now):
        """Consider a frame. It must not have a filter drawn on it yet."""
        if self.is_done(now) or now < self._next_sample_time:
            return
        self._problem = sample_problem(self.mode, frame, pose, face_count)
        if self._problem:
            return
        try:
            self._samples.append(self._embedder.embed(frame, pose.points))
        except cv2.error:
            self._problem = "Could not read that frame. Trying again"
            return
        self._next_sample_time = now + config.SAMPLE_INTERVAL_S

    def is_done(self, now):
        return len(self._samples) >= self.total or now >= self._deadline

    def describe(self):
        return {
            "mode": self.mode,
            "name": self.name,
            "collected": len(self._samples),
            "total": self.total,
            "hint": self._prompt(),
        }

    def _timed_out_result(self):
        reason = self._problem or "The camera couldn't get a clear view of your face"
        action = "Registration" if self.mode == ENROLL else "Scan"
        return {"kind": "failed", "message": f"{action} timed out. Last problem: {reason}."}

    def finish(self, store):
        """Turn the collected samples into a result, saving the profile if registering."""
        if len(self._samples) < self.total:
            return self._timed_out_result()

        samples = np.array(self._samples)
        template = normalize(samples.mean(axis=0))
        if self.mode == ENROLL:
            return self._finish_enroll(store, samples, template)
        return self._finish_scan(store, template)

    def _finish_enroll(self, store, samples, template):
        # A low score means the samples probably don't all show the same face.
        if float((samples @ template).min()) < config.MIN_ENROLL_CONSISTENCY:
            return {
                "kind": "failed",
                "message": "The samples didn't match each other. Make sure only you are in view, then try again.",
            }
        try:
            store.add(self.name, template)
        except DuplicateProfileError as exc:
            return {"kind": "failed", "message": str(exc)}
        except OSError as exc:
            return {"kind": "failed", "message": f"Could not save your face profile: {exc}"}
        return {"kind": "enrolled", "name": self.name, "message": f"{self.name} is registered. Try Scan Face."}

    def _finish_scan(self, store, template):
        match = store.best_match(template)
        if match is None:
            return {"kind": "failed", "message": "No faces are registered yet."}
        name, score = match
        threshold = config.FACE_MATCH_THRESHOLD
        if score >= threshold:
            return {"kind": "recognized", "name": name, "score": round(score, 2), "threshold": threshold, "message": f"Welcome back, {name}!"}
        return {"kind": "unknown", "score": round(score, 2), "threshold": threshold, "message": "Face not recognised."}
