"""Coordinate mapping and geometric transformations."""

import cv2
import numpy as np


def normalized_to_pixels(landmarks, width, height):
    """Map MediaPipe's normalized landmarks to pixel coordinates.

    MediaPipe returns x and y in the range 0..1, relative to image width and
    height. Its z uses roughly the same scale as x, so it is multiplied by the
    width to keep all three axes in the same unit:

        x_px = x * W      y_px = y * H      z_px = z * W
    """
    return np.asarray(landmarks, dtype=np.float32) * np.array([width, height, width], np.float32)


def clipped_bounds(points, width, height, padding=0):
    """Return the bounding box of points, clipped to the image, or None if off-screen.

    This defines the Region of Interest (ROI). All per-pixel work is limited to
    it, and clipping keeps array slices valid when a face reaches the frame edge.
    """
    points = np.asarray(points)
    x0 = max(int(np.floor(points[:, 0].min())) - padding, 0)
    y0 = max(int(np.floor(points[:, 1].min())) - padding, 0)
    x1 = min(int(np.ceil(points[:, 0].max())) + padding + 1, width)
    y1 = min(int(np.ceil(points[:, 1].max())) + padding + 1, height)
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def homography_to_roi(source_quad, target_quad, roi_origin):
    """Compute the 3x3 homography from sprite corners to ROI coordinates.

    A homography maps (x, y, 1) to (x', y', w), and the final position is
    (x'/w, y'/w). Four point pairs determine its 8 unknowns exactly, and unlike
    an affine transform it can represent perspective foreshortening.

    Warping into the ROI alone needs a translation applied after the homography:

        H_roi = T(-x0, -y0) * H
    """
    homography = cv2.getPerspectiveTransform(
        np.asarray(source_quad, np.float32), np.asarray(target_quad, np.float32)
    )
    translation = np.array(
        [[1, 0, -roi_origin[0]], [0, 1, -roi_origin[1]], [0, 0, 1]], dtype=np.float64
    )
    return translation @ homography
