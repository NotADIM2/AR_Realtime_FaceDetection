"""Central settings for the application."""

import os
import sys
from pathlib import Path


def _user_data_dir():
    """Per-user app data folder, kept outside the project and cloud-sync folders."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "ar-face-filters"


ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = ROOT / "assets"
FILTER_DIR = ASSET_DIR / "filters"
MODEL_PATH = ASSET_DIR / "models" / "face_landmarker.task"
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
RECOGNITION_MODEL_PATH = ASSET_DIR / "models" / "face_recognition_sface_2021dec.onnx"
RECOGNITION_MODEL_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
FACE_STORE_PATH = _user_data_dir() / "face_profiles.json"

# The server only ever binds to the loopback interface, so nothing is
# reachable from another device on the network.
HOST = "127.0.0.1"
PORT = 8765

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 30
MAX_CAMERA_INDEX = 4  # Indices 0..4 are probed by "Scan for cameras".

# Detect two faces so the largest one can be chosen as the primary face.
MAX_FACES = 2
# Another face must be this much larger before the filter moves to it.
PRIMARY_SWITCH_RATIO = 1.35
# Keep the last pose for this many frames when detection briefly drops out.
FACE_HOLD_FRAMES = 3

# One Euro filter settings, in pixel units.
SMOOTH_MIN_CUTOFF = 1.2
SMOOTH_BETA = 0.02
SMOOTH_D_CUTOFF = 1.0

# Assumed camera distance in eye-widths, used for perspective foreshortening.
PERSPECTIVE_DISTANCE = 6.5

# Face recognition. Scores are cosine similarity, from -1 to 1.
FACE_MATCH_THRESHOLD = 0.363  # OpenCV's published threshold for the SFace model
MAX_NAME_LENGTH = 40
ENROLL_SAMPLES = 8
SCAN_SAMPLES = 5
SAMPLE_INTERVAL_S = 0.35
SESSION_TIMEOUT_S = 20.0
MIN_SAMPLE_EYE_WIDTH_PX = 45.0
MAX_SAMPLE_ANGLE_DEG = 32.0
# During registration, every sample must be this similar to the average of all samples.
MIN_ENROLL_CONSISTENCY = 0.45

JPEG_QUALITY = 82
# Stop the camera when no browser has been watching for this many seconds.
NO_VIEWER_TIMEOUT_S = 8.0
