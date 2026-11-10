"""Runs the camera session and shares its results with the web server.

Frames live in memory only. Nothing is written to disk, and the most recent frame
is cleared when the session ends. Photos leave this class only when the user
clicks Capture.
"""

import threading
import time
import traceback

import cv2

from app import config
from app.camera.capture import CameraError, CameraSource, ImageSource
from app.filters.manager import FilterManager
from app.identity.embedder import FaceEmbedder
from app.identity.session import ENROLL, SCAN, CaptureSession
from app.identity.store import FaceStore
from app.pipeline.processor import FrameProcessor
from app.utils.fps import FpsCounter
from app.vision.detector import FaceDetector, ModelError, verify_model


class IdentityError(RuntimeError):
    """A Register or Scan request can't run right now. The message is shown to the user."""


class Engine:
    def __init__(self, demo_image=None, filters=None, store=None, embedder_factory=FaceEmbedder):
        self.demo_image = demo_image
        self.filters = filters or FilterManager()
        self.store = store or FaceStore()
        self._embedder_factory = embedder_factory
        self._embedder = None
        self._embedder_lock = threading.Lock()
        self._session = None
        self._identity_result = None
        self._identity_result_id = 0
        self._lifecycle_lock = threading.Lock()  # Serializes start and stop
        self._condition = threading.Condition()  # Guards every field below
        self._state = "stopped"
        self._message = None
        self._source_name = None
        self._filter_key = "normal"
        self._show_landmarks = False
        self._show_axes = False
        self._info = None
        self._fps = 0.0
        self._process_ms = 0.0
        self._resolution = None
        self._cameras = [0]
        self._camera_index = 0
        self._jpeg = None
        self._jpeg_sequence = 0
        self._last_frame = None
        self._viewers = 0
        self._last_viewer_time = 0.0
        self._thread = None
        self._stop_event = None

    def start(self, camera_index=0):
        """Open the camera and begin processing. Raises CameraError or ModelError."""
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            verify_model()
            source = ImageSource(self.demo_image) if self.demo_image else CameraSource(camera_index)
            with self._condition:
                self._state = "starting"
                self._message = None
            try:
                source.start()
            except CameraError as exc:
                self._finish("error", str(exc))
                raise
            with self._condition:
                self._camera_index = camera_index
                self._source_name = source.description
                self._last_viewer_time = time.monotonic()
                self._stop_event = threading.Event()
                self._thread = threading.Thread(
                    target=self._run, args=(source, self._stop_event), name="processing", daemon=True
                )
                self._thread.start()

    def stop(self):
        """Stop processing and release the camera."""
        with self._lifecycle_lock:
            thread, event = self._thread, self._stop_event
            if thread is None:
                return
            event.set()
            thread.join(timeout=5.0)
            self._thread = None

    def _finish(self, state, message=None):
        with self._condition:
            self._state = state
            self._message = message
            self._source_name = None
            self._info = None
            self._fps = 0.0
            self._process_ms = 0.0
            self._resolution = None
            self._jpeg = None
            self._last_frame = None
            self._session = None
            self._identity_result = None
            self._condition.notify_all()

    def _run(self, source, stop_event):
        """The processing thread. Whatever happens, it releases the camera on exit."""
        final_state, final_message = "stopped", None
        detector = None
        try:
            detector = FaceDetector()
            processor = FrameProcessor(detector)
            counter = FpsCounter()
            sequence = 0
            with self._condition:
                self._state = "running"
                self._condition.notify_all()

            while not stop_event.is_set():
                if source.error:
                    final_state, final_message = "error", source.error
                    break
                if self._viewers_gone():
                    final_message = "Camera turned off because no browser window was watching."
                    break
                item = source.wait_frame(sequence)
                if item is None:
                    continue
                sequence, frame = item
                self._process_frame(processor, frame, counter)
        except (CameraError, ModelError) as exc:
            final_state, final_message = "error", str(exc)
        except Exception:
            traceback.print_exc()
            final_state, final_message = "error", "Unexpected processing error. See the terminal for details."
        finally:
            source.stop()
            if detector is not None:
                detector.close()
            self._finish(final_state, final_message)

    def _viewers_gone(self):
        with self._condition:
            if self._viewers > 0:
                return False
            return time.monotonic() - self._last_viewer_time > config.NO_VIEWER_TIMEOUT_S

    def _process_frame(self, processor, frame, counter):
        with self._condition:
            active = self.filters.get(self._filter_key)
            show_landmarks = self._show_landmarks
            show_axes = self._show_axes
            session = self._session

        started = time.perf_counter()
        now = time.monotonic()
        output, info = processor.process(frame, now, active, show_landmarks, show_axes, session)
        if session is not None and session.is_done(now):
            self._complete_session(session)
        ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        counter.tick()
        if not ok:
            return

        with self._condition:
            self._info = info
            self._fps = counter.fps
            # Exponential moving average keeps the displayed number readable.
            self._process_ms = 0.9 * self._process_ms + 0.1 * elapsed_ms if self._process_ms else elapsed_ms
            self._resolution = (output.shape[1], output.shape[0])
            self._last_frame = output
            self._jpeg = encoded.tobytes()
            self._jpeg_sequence += 1
            self._condition.notify_all()

    def _complete_session(self, session):
        result = session.finish(self.store)
        with self._condition:
            # The user may have cancelled while the result was being computed.
            if self._session is session:
                self._session = None
                self._identity_result_id += 1
                self._identity_result = result | {"id": self._identity_result_id}

    def _get_embedder(self):
        """Load the recognition model the first time it's needed."""
        with self._embedder_lock:
            if self._embedder is None:
                self._embedder = self._embedder_factory()
            return self._embedder

    def _begin_session(self, mode, name=None):
        # Check first, so a clear reason is given before the model is loaded.
        self._check_can_begin()
        embedder = self._get_embedder()
        with self._condition:
            self._check_can_begin()
            self._identity_result = None
            self._session = CaptureSession(mode, embedder, time.monotonic(), name)

    def _check_can_begin(self):
        with self._condition:
            if self._state != "running":
                raise IdentityError("Start the camera first.")
            if self._session is not None:
                raise IdentityError("Please wait for the current face scan to finish.")

    def start_enroll(self, name):
        """Begin registering a face. Raises IdentityError or ModelError."""
        # The store checks again when saving, but this answers before any samples are taken.
        if self.store.has_name(name):
            raise IdentityError(f"The name {name} is already registered. Choose a different name.")
        self._begin_session(ENROLL, name)

    def start_scan(self):
        """Begin a recognition scan. Raises IdentityError or ModelError."""
        if not self.store.names():
            raise IdentityError("No faces are registered yet. Click Register Face first.")
        self._begin_session(SCAN)

    def cancel_identity_session(self):
        with self._condition:
            self._session = None

    def delete_profile(self, name):
        removed = self.store.remove(name)
        if removed:
            with self._condition:
                self._identity_result = None
        return removed

    def stream(self):
        """Yield multipart JPEG chunks for as long as the session runs."""
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
        with self._condition:
            self._viewers += 1
            last = self._jpeg_sequence
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._jpeg_sequence != last or self._state not in ("starting", "running"),
                        timeout=1.0,
                    )
                    if self._state not in ("starting", "running"):
                        return
                    if self._jpeg_sequence == last or self._jpeg is None:
                        continue
                    last = self._jpeg_sequence
                    jpeg = self._jpeg
                yield boundary + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
        finally:
            with self._condition:
                self._viewers -= 1
                self._last_viewer_time = time.monotonic()

    def capture_png(self):
        """Encode the current frame on an explicit request. Returns None if there is none."""
        with self._condition:
            frame = self._last_frame
        if frame is None:
            return None
        ok, encoded = cv2.imencode(".png", frame)
        return encoded.tobytes() if ok else None

    def set_filter(self, key):
        if not self.filters.is_usable(key):
            return False
        with self._condition:
            self._filter_key = key
        return True

    def set_options(self, landmarks=None, axes=None):
        with self._condition:
            if landmarks is not None:
                self._show_landmarks = bool(landmarks)
            if axes is not None:
                self._show_axes = bool(axes)

    def reset(self):
        """Return to the plain camera view with all debug layers off."""
        with self._condition:
            self._filter_key = "normal"
            self._show_landmarks = False
            self._show_axes = False
            self._identity_result = None

    def set_cameras(self, indices):
        with self._condition:
            self._cameras = indices or [self._camera_index]

    def status(self):
        profiles = self.store.names()
        with self._condition:
            info = self._info
            return {
                "identity": {
                    "profiles": profiles,
                    "session": self._session.describe() if self._session else None,
                    "result": self._identity_result,
                    "problem": self.store.problem,
                },
                "state": self._state,
                "message": self._message,
                "source": self._source_name,
                "filter": self._filter_key,
                "landmarks": self._show_landmarks,
                "axes": self._show_axes,
                "fps": round(self._fps, 1),
                "process_ms": round(self._process_ms, 1),
                "resolution": self._resolution,
                "cameras": self._cameras,
                "camera": self._camera_index,
                "demo": bool(self.demo_image),
                "faces": info.faces if info else 0,
                "tracking": info.tracking if info else False,
                "pose": {"yaw": round(info.yaw, 1), "pitch": round(info.pitch, 1), "roll": round(info.roll, 1)}
                if info and info.tracking
                else None,
            }
