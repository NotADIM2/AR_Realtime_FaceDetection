"""Frame sources that run on a background thread.

Each source keeps only the newest frame (a "mailbox"). If processing is slower
than the camera, old frames are dropped instead of queueing up, so the preview
never falls behind real time.
"""

import sys
import threading
import time

import cv2

from app import config

# Quieten OpenCV's console warnings when probing camera indices that don't exist.
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)

_MAX_READ_FAILURES = 30
_OPEN_READ_ATTEMPTS = 20


class CameraError(RuntimeError):
    """The camera could not be opened or stopped delivering frames."""


class FrameSource:
    """Base class: a thread that publishes the newest frame to a mailbox."""

    description = "source"

    def __init__(self):
        self._condition = threading.Condition()
        self._frame = None
        self._sequence = 0
        self._stop_event = threading.Event()
        self._thread = None
        self.error = None

    def start(self):
        self._open()
        self._thread = threading.Thread(target=self._guarded_run, name="capture", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._close()

    def wait_frame(self, last_sequence, timeout=0.5):
        """Return (sequence, frame) once a newer frame exists, otherwise None."""
        def ready():
            return self._sequence != last_sequence or self._stop_event.is_set() or self.error

        with self._condition:
            self._condition.wait_for(ready, timeout)
            if self._sequence == last_sequence or self._frame is None:
                return None
            return self._sequence, self._frame

    def _publish(self, frame):
        with self._condition:
            self._frame = frame
            self._sequence += 1
            self._condition.notify_all()

    def _guarded_run(self):
        try:
            self._run()
        except CameraError as exc:
            self.error = str(exc)
            with self._condition:
                self._condition.notify_all()

    def _open(self):
        pass

    def _close(self):
        pass

    def _run(self):
        raise NotImplementedError


def _preferred_backends():
    """Return (backend, request_mjpg) pairs in the order they should be tried."""
    if sys.platform == "win32":
        # On a USB 2.0 HD webcam at 720p, Media Foundation delivered 30 FPS and
        # opened faster, while DirectShow delivered only 15 FPS.
        return [(cv2.CAP_MSMF, False), (cv2.CAP_DSHOW, False), (cv2.CAP_ANY, False)]
    if sys.platform == "darwin":
        return [(cv2.CAP_AVFOUNDATION, False), (cv2.CAP_ANY, False)]
    # On Linux, many USB 2.0 webcams need MJPG to reach 720p at 30 FPS.
    return [(cv2.CAP_V4L2, True), (cv2.CAP_ANY, False)]


def _open_device(index, backend, request_mjpg):
    """Open a camera and confirm it delivers a frame. Returns None on failure."""
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        capture.release()
        return None

    if request_mjpg:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, config.CAPTURE_FPS)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(_OPEN_READ_ATTEMPTS):
        ok, frame = capture.read()
        if ok and frame is not None:
            return capture
        time.sleep(0.05)
    capture.release()
    return None


class CameraSource(FrameSource):
    """A webcam read through OpenCV."""

    def __init__(self, index):
        super().__init__()
        self.index = index
        self.description = f"Camera {index}"
        self._capture = None

    def _open(self):
        for backend, request_mjpg in _preferred_backends():
            self._capture = _open_device(self.index, backend, request_mjpg)
            if self._capture is not None:
                return
        raise CameraError(
            f"Could not open camera {self.index}. Check that it is connected, that no "
            "other app is using it, and that camera access is allowed in your system "
            "privacy settings."
        )

    def _close(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _run(self):
        failures = 0
        while not self._stop_event.is_set():
            ok, frame = self._capture.read()
            if ok and frame is not None:
                failures = 0
                self._publish(frame)
                continue
            failures += 1
            if failures >= _MAX_READ_FAILURES:
                raise CameraError(f"Camera {self.index} stopped responding. Was it disconnected?")
            time.sleep(0.02)


class ImageSource(FrameSource):
    """Replays a still image as if it were a camera, for demos without a webcam."""

    def __init__(self, path):
        super().__init__()
        self.path = str(path)
        self.description = "Demo image"
        self._image = None

    def _open(self):
        self._image = cv2.imread(self.path)
        if self._image is None:
            raise CameraError(f"Could not read demo image: {self.path}")

    def _run(self):
        interval = 1.0 / config.CAPTURE_FPS
        while not self._stop_event.wait(interval):
            self._publish(self._image.copy())


def scan_cameras():
    """Return the indices of working cameras. This briefly activates each camera."""
    found = []
    backend, request_mjpg = _preferred_backends()[0]
    for index in range(config.MAX_CAMERA_INDEX + 1):
        capture = _open_device(index, backend, request_mjpg)
        if capture is not None:
            found.append(index)
            capture.release()
    return found
