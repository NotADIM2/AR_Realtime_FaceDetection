"""Tests for registering and recognising faces. These use fake templates, not real faces.

Run from the project root:  python -m unittest discover tests
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from app import config
from app.identity.embedder import normalize
from app.identity.session import ENROLL, SCAN, CaptureSession, sample_problem
from app.identity.store import DuplicateProfileError, FaceStore
from app.pipeline.engine import Engine, IdentityError
from app.vision.pose import estimate_pose
from app.web.server import create_app
from tests.test_core import synthetic_face


def template(seed):
    return normalize(np.random.default_rng(seed).normal(size=128))


def nearby(base, seed, noise=0.02):
    return normalize(base + np.random.default_rng(seed).normal(size=128) * noise)


class FakeEmbedder:
    """Returns the queued templates in order, so tests control what the 'camera' sees."""

    def __init__(self, templates):
        self.templates = list(templates)

    def embed(self, frame, points):
        return self.templates.pop(0)


FRAME = np.zeros((720, 1280, 3), np.uint8)
GOOD_POSE = estimate_pose(synthetic_face())


def run_session(session, now=0.0):
    while not session.is_done(now):
        session.offer(FRAME, GOOD_POSE, 1, now)
        now += config.SAMPLE_INTERVAL_S
    return now


class StoreCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "nested" / "profiles.json"
        self.store = FaceStore(self.path)


class StoreTests(StoreCase):
    def test_profiles_survive_a_restart(self):
        saved = template(1)
        self.store.add("Asha", saved)
        reloaded = FaceStore(self.path)
        self.assertEqual(reloaded.names(), ["Asha"])
        self.assertAlmostEqual(reloaded.best_match(saved)[1], 1.0, places=6)

    def test_best_match_chooses_the_closest_profile(self):
        first, second = template(1), template(2)
        self.store.add("One", first)
        self.store.add("Two", second)
        self.assertEqual(self.store.best_match(nearby(second, 9))[0], "Two")

    def test_best_match_with_no_profiles(self):
        self.assertIsNone(self.store.best_match(template(1)))

    def test_a_name_can_only_be_registered_once(self):
        self.store.add("Asha", template(1))
        with self.assertRaisesRegex(DuplicateProfileError, "name Asha is already registered"):
            self.store.add("ASHA", template(2))
        self.assertEqual(self.store.names(), ["Asha"])
        self.assertTrue(self.store.has_name("asha"))

    def test_a_face_can_only_be_registered_once(self):
        base = template(1)
        self.store.add("Asha", base)
        with self.assertRaisesRegex(DuplicateProfileError, "already registered as Asha"):
            self.store.add("Someone Else", nearby(base, 5))
        self.assertEqual(FaceStore(self.path).names(), ["Asha"])

    def test_a_different_face_can_register(self):
        self.store.add("Asha", template(1))
        self.store.add("Ravi", template(2))
        self.assertEqual(self.store.names(), ["Asha", "Ravi"])

    def test_deleting_a_profile_allows_registering_again(self):
        base = template(1)
        self.store.add("Asha", base)
        self.store.remove("Asha")
        self.store.add("Asha", nearby(base, 5))
        self.assertEqual(self.store.names(), ["Asha"])

    def test_deleted_profile_is_gone_after_a_restart(self):
        self.store.add("Asha", template(1))
        self.assertTrue(self.store.remove("asha"))
        self.assertEqual(FaceStore(self.path).names(), [])
        self.assertFalse(self.store.remove("asha"))

    def test_damaged_file_is_reported_and_kept_aside(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{not json", encoding="utf-8")
        store = FaceStore(self.path)
        self.assertEqual(store.names(), [])
        self.assertIn("could not be read", store.problem)
        self.assertTrue(self.path.with_name("profiles.json.damaged").exists())

    def test_file_holds_only_names_and_numbers(self):
        self.store.add("Asha", template(1))
        text = self.path.read_text(encoding="utf-8")
        self.assertNotIn("image", text)
        self.assertLess(len(text), 8000)


class SampleQualityTests(unittest.TestCase):
    def test_good_frame_passes(self):
        self.assertIsNone(sample_problem(ENROLL, FRAME, GOOD_POSE, 1))

    def test_no_face(self):
        self.assertIn("No face", sample_problem(SCAN, FRAME, None, 0))
        # A pose held over from an earlier frame isn't a real sample.
        self.assertIn("No face", sample_problem(SCAN, FRAME, GOOD_POSE, 0))

    def test_second_person_blocks_registration_only(self):
        self.assertIn("one person", sample_problem(ENROLL, FRAME, GOOD_POSE, 2))
        self.assertIsNone(sample_problem(SCAN, FRAME, GOOD_POSE, 2))

    def test_small_face(self):
        pose = estimate_pose(synthetic_face(eye_width=20))
        self.assertIn("closer", sample_problem(SCAN, FRAME, pose, 1))

    def test_turned_face(self):
        pose = estimate_pose(synthetic_face(yaw_deg=60))
        self.assertIn("directly", sample_problem(SCAN, FRAME, pose, 1))

    def test_face_partly_out_of_frame(self):
        pose = estimate_pose(synthetic_face(center=(1270, 360)))
        self.assertIn("frame", sample_problem(SCAN, FRAME, pose, 1))


class SessionTests(StoreCase):
    def enroll(self, name="Asha", base=None):
        base = template(1) if base is None else base
        samples = [nearby(base, i) for i in range(config.ENROLL_SAMPLES)]
        session = CaptureSession(ENROLL, FakeEmbedder(samples), 0.0, name)
        end = run_session(session)
        return session.finish(self.store), end

    def scan(self, samples):
        session = CaptureSession(SCAN, FakeEmbedder(samples), 0.0)
        run_session(session)
        return session.finish(self.store)

    def test_registering_saves_a_profile(self):
        result, _ = self.enroll()
        self.assertEqual(result["kind"], "enrolled")
        self.assertEqual(self.store.names(), ["Asha"])

    def test_same_face_is_recognised(self):
        base = template(1)
        self.enroll(base=base)
        result = self.scan([nearby(base, 100 + i) for i in range(config.SCAN_SAMPLES)])
        self.assertEqual(result["kind"], "recognized")
        self.assertEqual(result["name"], "Asha")
        self.assertGreater(result["score"], config.FACE_MATCH_THRESHOLD)

    def test_different_face_is_not_recognised(self):
        self.enroll(base=template(1))
        other = template(2)
        result = self.scan([nearby(other, 100 + i) for i in range(config.SCAN_SAMPLES)])
        self.assertEqual(result["kind"], "unknown")
        self.assertNotIn("name", result)

    def test_registered_face_cannot_register_under_another_name(self):
        base = template(1)
        self.enroll("Asha", base)
        result, _ = self.enroll("Priya", nearby(base, 50))
        self.assertEqual(result["kind"], "failed")
        self.assertIn("already registered as Asha", result["message"])
        self.assertEqual(self.store.names(), ["Asha"])

    def test_mixed_people_are_rejected_during_registration(self):
        samples = [template(i) for i in range(config.ENROLL_SAMPLES)]
        session = CaptureSession(ENROLL, FakeEmbedder(samples), 0.0, "Asha")
        run_session(session)
        self.assertEqual(session.finish(self.store)["kind"], "failed")
        self.assertEqual(self.store.names(), [])

    def test_timeout_explains_the_problem_and_saves_nothing(self):
        session = CaptureSession(ENROLL, FakeEmbedder([]), 0.0, "Asha")
        session.offer(FRAME, None, 0, 1.0)
        self.assertFalse(session.is_done(1.0))
        self.assertTrue(session.is_done(config.SESSION_TIMEOUT_S))
        result = session.finish(self.store)
        self.assertEqual(result["kind"], "failed")
        self.assertIn("No face", result["message"])
        self.assertEqual(self.store.names(), [])

    def test_samples_are_spaced_apart(self):
        session = CaptureSession(SCAN, FakeEmbedder([template(1), template(2)]), 0.0)
        session.offer(FRAME, GOOD_POSE, 1, 0.0)
        session.offer(FRAME, GOOD_POSE, 1, 0.01)
        self.assertEqual(session.describe()["collected"], 1)


class EngineIdentityTests(StoreCase):
    def setUp(self):
        super().setUp()
        self.engine = Engine(store=self.store, embedder_factory=lambda: FakeEmbedder([]))

    def test_register_needs_a_running_camera(self):
        with self.assertRaisesRegex(IdentityError, "Start the camera"):
            self.engine.start_enroll("Asha")

    def test_scan_needs_a_registered_face(self):
        with self.assertRaisesRegex(IdentityError, "No faces are registered"):
            self.engine.start_scan()

    def test_only_one_session_runs_at_a_time(self):
        with self.engine._condition:
            self.engine._state = "running"
        self.engine.start_enroll("Asha")
        with self.assertRaisesRegex(IdentityError, "wait"):
            self.engine.start_enroll("Asha")
        self.engine.cancel_identity_session()
        self.engine.start_enroll("Asha")

    def test_taken_name_is_refused_before_scanning(self):
        self.store.add("Asha", template(1))
        with self.engine._condition:
            self.engine._state = "running"
        with self.assertRaisesRegex(IdentityError, "already registered"):
            self.engine.start_enroll("asha")
        self.assertIsNone(self.engine.status()["identity"]["session"])

    def test_status_reports_profiles(self):
        self.store.add("Asha", template(1))
        self.assertEqual(self.engine.status()["identity"]["profiles"], ["Asha"])


class IdentityApiTests(StoreCase):
    BASE = "http://127.0.0.1:8765"

    def setUp(self):
        super().setUp()
        self.engine = Engine(store=self.store, embedder_factory=lambda: FakeEmbedder([]))
        self.client = create_app(self.engine).test_client()

    def post(self, path, payload=None):
        return self.client.post(path, base_url=self.BASE, json=payload or {})

    def test_blank_or_long_names_are_rejected(self):
        for name in ("", "   ", "x" * 41, 5, None, "bad\x00name"):
            self.assertEqual(self.post("/api/identity/enroll", {"name": name}).status_code, 400, name)

    def test_register_without_camera_gives_a_clear_message(self):
        response = self.post("/api/identity/enroll", {"name": "Asha"})
        self.assertEqual(response.status_code, 409)
        self.assertIn("Start the camera", response.get_json()["error"])

    def test_delete_removes_a_profile(self):
        self.store.add("Asha", template(1))
        self.assertEqual(self.post("/api/identity/delete", {"name": "Asha"}).status_code, 200)
        self.assertEqual(self.store.names(), [])
        self.assertEqual(self.post("/api/identity/delete", {"name": "Asha"}).status_code, 404)

    def test_other_sites_cannot_use_recognition(self):
        response = self.client.post(
            "/api/identity/scan", base_url=self.BASE, json={}, headers={"Origin": "https://evil.example"}
        )
        self.assertEqual(response.status_code, 403)

    def test_find_cameras_is_refused_while_the_camera_is_running(self):
        with self.engine._condition:
            self.engine._state = "running"
        response = self.post("/api/cameras/find")
        self.assertEqual(response.status_code, 409)
        self.assertIn("Stop the camera", response.get_json()["error"])

    def test_find_cameras_updates_the_list(self):
        with mock.patch("app.web.server.scan_cameras", return_value=[0, 2]):
            self.assertEqual(self.post("/api/cameras/find").get_json()["cameras"], [0, 2])
        self.assertEqual(self.engine.status()["cameras"], [0, 2])


if __name__ == "__main__":
    unittest.main()
