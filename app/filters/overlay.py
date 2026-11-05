"""Filters that warp a transparent PNG onto the face."""

from dataclasses import dataclass

import cv2
import numpy as np

from app.imaging.blending import composite_premultiplied, premultiply
from app.imaging.transforms import clipped_bounds, homography_to_roi

_MIN_FACE_SCALE_PX = 8.0


@dataclass(frozen=True)
class OverlaySpec:
    """Where a sprite sits on the face. Distances are in eye-width units."""

    key: str
    label: str
    asset: str
    landmark: int  # Landmark the sprite is attached to
    width: float  # Sprite width
    pivot: tuple = (0.5, 0.5)  # Point on the sprite (0..1) placed at the landmark
    offset: tuple = (0.0, 0.0, 0.0)  # Shift from the landmark along X, Y, Z
    lean: float = 0.0  # Depth change per unit of sprite height, tilts it backwards


class Sprite:
    """A premultiplied BGRA image with a mipmap pyramid.

    Warping a large sprite straight down to a small size skips source pixels and
    causes shimmering. Each pyramid level is half the size of the one before, and
    warping starts from the smallest level still larger than the target.
    """

    def __init__(self, bgra):
        self.levels = [premultiply(bgra)]
        while min(self.levels[-1].shape[:2]) > 32:
            self.levels.append(cv2.pyrDown(self.levels[-1]))
        height, width = bgra.shape[:2]
        self.aspect = height / width

    def level_for(self, target_width):
        for level in reversed(self.levels):
            if level.shape[1] >= target_width:
                return level
        return self.levels[0]


class OverlayFilter:
    needs_face = True

    def __init__(self, spec, path):
        self.spec = spec
        self.key = spec.key
        self.label = spec.label
        self.sprite = None
        self.problem = None

        bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
            self.problem = f"Missing or invalid asset: {spec.asset}"
            return
        self.sprite = Sprite(bgra)
        self._face_corners = self._build_face_corners()

    def _build_face_corners(self):
        """Compute the sprite's four corners in face space."""
        spec = self.spec
        height = spec.width * self.sprite.aspect
        corners = []
        for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
            down = (v - spec.pivot[1]) * height
            corners.append(
                (
                    spec.offset[0] + (u - spec.pivot[0]) * spec.width,
                    spec.offset[1] + down,
                    spec.offset[2] - down * spec.lean,
                )
            )
        return np.array(corners, dtype=np.float32)

    def apply(self, frame, pose):
        if pose is None or self.sprite is None or pose.scale < _MIN_FACE_SCALE_PX:
            return

        # Geometric transform: face-space corners -> image corners.
        target = pose.project(self.spec.landmark, self._face_corners)
        if not np.isfinite(target).all():
            return

        # ROI extraction: only the pixels the sprite can touch are processed.
        height, width = frame.shape[:2]
        bounds = clipped_bounds(target, width, height, padding=1)
        if bounds is None:
            return
        x0, y0, x1, y1 = bounds

        target_width = float(np.linalg.norm(target[1] - target[0]))
        level = self.sprite.level_for(target_width)
        source_h, source_w = level.shape[:2]
        source = [(0, 0), (source_w, 0), (source_w, source_h), (0, source_h)]

        # Resize, rotate and align in one step with a perspective warp.
        homography = homography_to_roi(source, target, (x0, y0))
        warped = cv2.warpPerspective(
            level,
            homography,
            (x1 - x0, y1 - y0),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        composite_premultiplied(frame[y0:y1, x0:x1], warped)
