"""Failure-handling tests for the camera session. These use fake cameras.

Run from the project root:  python -m unittest discover tests
"""

import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from app import config
from app.camera.capture import CameraError, CameraSource, FrameSource
from app.pipeline import engine as engine_module
from app.pipeline.engine import Engine
from app.vision.detector import ModelError, verify_model
from app.web.server import create_app


class FakeSource(FrameSource):
    """Delivers blank frames, then optionally fails like an unplugged camera."""

    instances = []
    fail_after = None
    fail_on_open = False

    def __init__(self, index):
        super().__init__()
        self.description = f"Fake {index}"
        self.released = False
        FakeSource.instances.append(self)

    def _open(self):
        if FakeSource.fail_on_open:
            raise CameraError("Could not open camera.")

    def _close(self):
        self.released = True

    def _run(self):
        count = 0
        while not self._stop_event.wait(0.01):
            count += 1
            if FakeSource.fail_after is not None and count > FakeSource.fail_after:
                raise CameraError("Camera stopped responding.")
            self._publish(np.zeros((120, 160, 3), np.uint8))


def wait_until(condition, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


class EngineTests(unittest.TestCase):
    def setUp(self):
        FakeSource.instances = []
        FakeSource.fail_after = None
        FakeSource.fail_on_open = False
        patcher = mock.patch.object(engine_module, "CameraSource", FakeSource)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.engine = Engine()
        self.addCleanup(self.engine.stop)

    def keep_viewer(self):
        with self.engine._condition:
            self.engine._viewers = 1

    def test_stop_releases_the_camera(self):
        self.engine.start(0)
        self.keep_viewer()
        self.assertTrue(wait_until(lambda: self.engine.status()["state"] == "running"))
        self.engine.stop()
        self.assertTrue(FakeSource.instances[0].released)
        status = self.engine.status()
        self.assertEqual(status["state"], "stopped")
        self.assertIsNone(self.engine.capture_png())

    def test_failure_to_open_reports_an_error(self):
        FakeSource.fail_on_open = True
        with self.assertRaises(CameraError):
            self.engine.start(0)
        self.assertEqual(self.engine.status()["state"], "error")

    def test_disconnect_during_a_session_is_reported_and_releases_the_camera(self):
        FakeSource.fail_after = 3
        self.engine.start(0)
        self.keep_viewer()
        self.assertTrue(wait_until(lambda: self.engine.status()["state"] == "error"))
        self.assertIn("stopped responding", self.engine.status()["message"])
        self.assertTrue(FakeSource.instances[0].released)

    def test_camera_can_restart_after_an_error(self):
        FakeSource.fail_after = 1
        self.engine.start(0)
        self.keep_viewer()
        self.assertTrue(wait_until(lambda: self.engine.status()["state"] == "error"))
        FakeSource.fail_after = None
        self.engine.start(0)
        self.assertTrue(wait_until(lambda: self.engine.status()["state"] == "running"))

    def test_camera_turns_off_when_no_browser_is_watching(self):
        with mock.patch.object(config, "NO_VIEWER_TIMEOUT_S", 0.2):
            self.engine.start(0)
            self.assertTrue(wait_until(lambda: self.engine.status()["state"] == "stopped"))
        self.assertTrue(FakeSource.instances[0].released)
        status = self.engine.status()
        self.assertEqual(status["state"], "stopped")
        self.assertIn("no browser", status["message"])

    def test_no_face_frames_are_processed_without_error(self):
        self.engine.start(0)
        self.keep_viewer()
        self.assertTrue(wait_until(lambda: self.engine.status()["fps"] > 0))
        status = self.engine.status()
        self.assertEqual(status["faces"], 0)
        self.assertFalse(status["tracking"])
        self.assertIsNone(status["pose"])

    def test_capture_returns_a_png_only_while_running(self):
        self.assertIsNone(self.engine.capture_png())
        self.engine.start(0)
        self.keep_viewer()
        self.assertTrue(wait_until(lambda: self.engine.capture_png() is not None))
        self.assertTrue(self.engine.capture_png().startswith(b"\x89PNG"))

    def test_unknown_filter_is_rejected(self):
        self.assertFalse(self.engine.set_filter("does_not_exist"))
        self.assertEqual(self.engine.status()["filter"], "normal")


class ModelTests(unittest.TestCase):
    def test_missing_model_is_reported(self):
        with self.assertRaises(ModelError):
            verify_model(Path("does_not_exist.task"))

    def test_damaged_model_is_reported(self):
        with self.assertRaises(ModelError):
            verify_model(config.MODEL_PATH, "0" * 64)

    def test_project_model_passes_its_checksum(self):
        verify_model()


class ServerSecurityTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app(Engine()).test_client()

    def test_local_request_is_allowed(self):
        self.assertEqual(self.client.get("/api/state", base_url="http://127.0.0.1:8765").status_code, 200)

    def test_other_host_names_are_blocked(self):
        self.assertEqual(self.client.get("/api/state", base_url="http://evil.example").status_code, 403)

    def test_other_sites_cannot_read_the_stream(self):
        response = self.client.get(
            "/stream", base_url="http://127.0.0.1:8765", headers={"Sec-Fetch-Site": "cross-site"}
        )
        self.assertEqual(response.status_code, 403)

    def test_other_sites_cannot_start_the_camera(self):
        response = self.client.post(
            "/api/start", base_url="http://127.0.0.1:8765", json={},
            headers={"Origin": "https://evil.example"},
        )
        self.assertEqual(response.status_code, 403)

    def test_capture_without_a_session_saves_nothing(self):
        response = self.client.post("/api/capture", base_url="http://127.0.0.1:8765", json={})
        self.assertEqual(response.status_code, 409)

    def test_invalid_camera_index_is_rejected(self):
        response = self.client.post("/api/start", base_url="http://127.0.0.1:8765", json={"camera": "x"})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
