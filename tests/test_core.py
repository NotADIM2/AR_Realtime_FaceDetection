"""Tests for the image-processing core. These need no camera or face model.

Run from the project root:  python -m unittest discover tests
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.filters.drawn import MeshFilter, NormalFilter, OutlineFilter
from app.filters.manager import FilterManager
from app.filters.overlay import OverlayFilter, OverlaySpec
from app.imaging.blending import composite_premultiplied, premultiply
from app.imaging.transforms import clipped_bounds, homography_to_roi, normalized_to_pixels
from app.vision import landmarks as lm
from app.vision.pose import estimate_pose
from app.vision.smoothing import OneEuroFilter
from app.vision.tracker import FaceTracker, choose_primary


def synthetic_face(center=(640.0, 360.0), eye_width=200.0, yaw_deg=0.0, roll_deg=0.0):
    """Builds pixel landmarks with the key points placed at known positions."""
    yaw, roll = np.radians(yaw_deg), np.radians(roll_deg)
    rotate_yaw = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rotate_roll = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    rotation = rotate_roll @ rotate_yaw

    # Unnamed landmarks sit at the face centre so they don't affect the bounding box.
    points = np.full((lm.LANDMARK_COUNT, 3), (center[0], center[1], 0), np.float32)
    local = {
        lm.EYE_OUTER_IMAGE_LEFT: (-0.5, 0, 0),
        lm.EYE_OUTER_IMAGE_RIGHT: (0.5, 0, 0),
        lm.FOREHEAD_TOP: (0, -0.8, 0),
        lm.CHIN: (0, 1.1, 0),
        lm.NOSE_BRIDGE: (0, 0, 0),
        lm.NOSE_TIP: (0, 0.4, -0.3),
        lm.PHILTRUM: (0, 0.6, -0.2),
        lm.IRIS_IMAGE_LEFT: (-0.3, 0, 0),
        lm.IRIS_IMAGE_RIGHT: (0.3, 0, 0),
        lm.MOUTH_IMAGE_LEFT: (-0.25, 0.75, 0),
        lm.MOUTH_IMAGE_RIGHT: (0.25, 0.75, 0),
    }
    for index, offset in local.items():
        points[index] = (rotation @ (np.array(offset) * eye_width)) + (center[0], center[1], 0)
    return points


class TransformTests(unittest.TestCase):
    def test_normalized_to_pixels_scales_each_axis(self):
        result = normalized_to_pixels([[0.5, 0.25, -0.1]], 1280, 720)
        np.testing.assert_allclose(result, [[640, 180, -128]], rtol=1e-6)

    def test_bounds_are_clipped_to_the_image(self):
        self.assertEqual(clipped_bounds([[-50, -20], [30, 40]], 100, 100), (0, 0, 31, 41))

    def test_bounds_are_none_when_fully_off_screen(self):
        self.assertIsNone(clipped_bounds([[200, 200], [300, 300]], 100, 100))

    def test_homography_maps_corners_into_roi(self):
        source = [(0, 0), (100, 0), (100, 50), (0, 50)]
        target = [(110, 220), (210, 220), (210, 270), (110, 270)]
        matrix = homography_to_roi(source, target, (100, 200))
        mapped = matrix @ np.array([100, 50, 1.0])
        np.testing.assert_allclose(mapped[:2] / mapped[2], (110, 70), atol=1e-6)


class BlendingTests(unittest.TestCase):
    def test_premultiply(self):
        bgra = np.array([[[200, 100, 50, 128]]], np.uint8)
        np.testing.assert_array_equal(premultiply(bgra), [[[100, 50, 25, 128]]])

    def test_composite_matches_the_alpha_formula(self):
        background = np.full((2, 2, 3), 100, np.uint8)
        foreground = np.full((2, 2, 4), 200, np.uint8)
        foreground[..., 3] = 128
        composite_premultiplied(background, premultiply(foreground))
        expected = round((128 / 255) * 200 + (1 - 128 / 255) * 100)
        self.assertLessEqual(abs(int(background[0, 0, 0]) - expected), 1)

    def test_transparent_overlay_changes_nothing(self):
        background = np.random.default_rng(1).integers(0, 255, (4, 4, 3), dtype=np.uint8)
        original = background.copy()
        composite_premultiplied(background, np.zeros((4, 4, 4), np.uint8))
        np.testing.assert_array_equal(background, original)

    def test_opaque_overlay_replaces_background(self):
        background = np.zeros((1, 1, 3), np.uint8)
        composite_premultiplied(background, np.array([[[10, 20, 30, 255]]], np.uint8))
        np.testing.assert_array_equal(background, [[[10, 20, 30]]])


class PoseTests(unittest.TestCase):
    def test_frontal_face(self):
        pose = estimate_pose(synthetic_face())
        self.assertAlmostEqual(pose.scale, 200, places=2)
        self.assertAlmostEqual(pose.roll, 0, places=2)
        self.assertAlmostEqual(pose.yaw, 0, places=2)

    def test_roll_is_measured(self):
        self.assertAlmostEqual(estimate_pose(synthetic_face(roll_deg=30)).roll, 30, places=1)

    def test_yaw_is_measured_and_scale_stays_constant(self):
        pose = estimate_pose(synthetic_face(yaw_deg=40))
        self.assertAlmostEqual(abs(pose.yaw), 40, places=1)
        self.assertAlmostEqual(pose.scale, 200, places=2)

    def test_rotation_is_orthonormal(self):
        pose = estimate_pose(synthetic_face(yaw_deg=25, roll_deg=-15))
        np.testing.assert_allclose(pose.rotation.T @ pose.rotation, np.eye(3), atol=1e-5)

    def test_turning_foreshortens_the_projected_width(self):
        offsets = np.array([(-0.5, 0, 0), (0.5, 0, 0)], np.float32)
        def width(yaw):
            ends = estimate_pose(synthetic_face(yaw_deg=yaw)).project(lm.NOSE_BRIDGE, offsets)
            return abs(ends[1, 0] - ends[0, 0])
        self.assertAlmostEqual(width(0), 200, places=1)
        self.assertLess(width(50), width(0) * 0.8)

    def test_turning_makes_the_near_side_larger(self):
        pose = estimate_pose(synthetic_face(yaw_deg=40))
        offsets = np.array([(-0.5, -0.2, 0), (-0.5, 0.2, 0), (0.5, -0.2, 0), (0.5, 0.2, 0)], np.float32)
        ends = pose.project(lm.NOSE_BRIDGE, offsets)
        depth = (offsets @ pose.rotation.T)[:, 2]
        near, far = (0, 2) if depth[0] < depth[2] else (2, 0)
        self.assertGreater(ends[near + 1, 1] - ends[near, 1], ends[far + 1, 1] - ends[far, 1])


class SmoothingTests(unittest.TestCase):
    def setUp(self):
        self.filter = OneEuroFilter(min_cutoff=1.2, beta=0.02)

    def test_first_value_passes_through(self):
        value = np.array([5.0, 7.0], np.float32)
        np.testing.assert_array_equal(self.filter(value, 0.0), value)

    def test_jitter_on_a_still_point_is_reduced(self):
        rng = np.random.default_rng(3)
        noise = rng.normal(0, 1.5, 300)
        output = [self.filter(np.array([100.0 + n]), i / 30)[0] for i, n in enumerate(noise)]
        self.assertLess(np.std(output[30:]), np.std(noise[30:]) * 0.5)

    def test_lag_is_small_during_fast_motion(self):
        output = None
        for i in range(60):
            output = self.filter(np.array([i * 20.0]), i / 30)
        self.assertLess(abs(59 * 20.0 - output[0]), 40.0)

    def test_filter_handles_repeated_timestamps(self):
        self.filter(np.array([1.0]), 1.0)
        self.assertTrue(np.isfinite(self.filter(np.array([2.0]), 1.0)).all())


class TrackerTests(unittest.TestCase):
    def test_largest_face_is_chosen_first(self):
        faces = [synthetic_face(eye_width=100), synthetic_face(eye_width=250)]
        self.assertEqual(choose_primary(faces, None), 1)

    def test_primary_face_is_kept_when_sizes_are_similar(self):
        faces = [synthetic_face((300, 360), 200), synthetic_face((900, 360), 215)]
        self.assertEqual(choose_primary(faces, np.array((300.0, 360.0))), 0)

    def test_primary_moves_when_another_face_is_clearly_larger(self):
        faces = [synthetic_face((300, 360), 100), synthetic_face((900, 360), 260)]
        self.assertEqual(choose_primary(faces, np.array((300.0, 360.0))), 1)

    def test_brief_dropout_is_held_then_released(self):
        tracker = FaceTracker()
        self.assertIsNotNone(tracker.update([synthetic_face()], 0.0))
        self.assertIsNotNone(tracker.update([], 0.03))
        results = [tracker.update([], 0.06 + i * 0.03) for i in range(4)]
        self.assertIsNone(results[-1])


class FilterTests(unittest.TestCase):
    def make_overlay(self, directory):
        import cv2

        path = Path(directory) / "sprite.png"
        sprite = np.zeros((100, 200, 4), np.uint8)
        sprite[...] = (0, 0, 255, 255)
        cv2.imwrite(str(path), sprite)
        spec = OverlaySpec("test", "Test", "sprite.png", lm.NOSE_BRIDGE, width=1.0)
        return OverlayFilter(spec, path)

    def test_overlay_is_drawn_at_the_landmark(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = self.make_overlay(directory)
        frame = np.zeros((720, 1280, 3), np.uint8)
        overlay.apply(frame, estimate_pose(synthetic_face()))
        np.testing.assert_array_equal(frame[360, 640], (0, 0, 255))
        np.testing.assert_array_equal(frame[100, 100], (0, 0, 0))
        # Width is 1 eye width = 200 px, centred on x = 640.
        self.assertGreater(frame[360, 545, 2], 200)
        self.assertEqual(frame[360, 760, 2], 0)

    def test_overlay_near_every_edge_does_not_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            overlay = self.make_overlay(directory)
        for center in ((0, 0), (1280, 0), (0, 720), (1280, 720), (5000, 5000), (-5000, 10)):
            frame = np.zeros((720, 1280, 3), np.uint8)
            overlay.apply(frame, estimate_pose(synthetic_face(center)))

    def test_every_filter_handles_a_missing_face(self):
        frame = np.zeros((10, 10, 3), np.uint8)
        for item in (NormalFilter(), MeshFilter(), OutlineFilter()):
            item.apply(frame, None)
        self.assertEqual(frame.sum(), 0)

    def test_missing_assets_disable_filters_without_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FilterManager(Path(directory))
        described = {f["key"]: f["available"] for f in manager.describe()}
        self.assertFalse(described["crown"])
        self.assertTrue(described["normal"])
        self.assertTrue(described["mesh"])
        self.assertFalse(manager.is_usable("crown"))
        frame = np.zeros((720, 1280, 3), np.uint8)
        manager.get("crown").apply(frame, estimate_pose(synthetic_face()))
        self.assertEqual(frame.sum(), 0)

    def test_project_assets_are_all_present(self):
        self.assertTrue(all(f["available"] for f in FilterManager().describe()))


if __name__ == "__main__":
    unittest.main()
