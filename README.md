# Real-Time AR Face Filters

A local web app that detects facial landmarks from your webcam, applies AR filters that follow your face, and can register your face and recognise you the next time you scan. Python does all the processing, and your browser shows the interface.

It runs entirely on your computer. There are no API keys, accounts or cloud services, and video never leaves your machine.

## Features

- **Live webcam:** start and stop, camera selection, FPS and processing-time display
- **Face tracking:** 478 landmarks from MediaPipe, with live yaw, pitch and roll
- **Face recognition:** register your face once, then Scan Face recognises you, even after restarting the app
- **8 modes:** Normal, AR Glasses, Sunglasses, Moustache, Bunny Ears, Crown, Face Mesh, Face Outline
- **Visualization:** optional landmark points and head pose axes
- **Photos:** saved only when you click Capture Photo or press `C`
- **Shortcuts:** press `1`–`8` to switch filters

## Run it

You need Python 3.12 or 3.13.

**Windows:** double-click `run.bat`.

**Linux or macOS:**

```sh
sh run.sh
```

The first run installs the dependencies, which needs internet access and about 400 MB of disk space. Your browser then opens at <http://127.0.0.1:8765>. To quit, press `Ctrl+C` in the terminal window.

### Manual setup

```sh
python -m venv .venv
# Windows:        .venv\Scripts\activate
# Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

On Windows, if the project is in a OneDrive folder, create the virtual environment somewhere else so it doesn't get synced. `run.bat` does this for you.

### Options

| Option | Purpose |
|---|---|
| `--port 9000` | Use a different local port |
| `--image photo.jpg` | Use a still image instead of a webcam |
| `--no-browser` | Don't open the browser automatically |

## Using face recognition

1. Click **Start Camera**.
2. Click **Register Face**, enter your name and follow the prompts. It takes about 3 seconds.
3. Next time, click **Start Camera**, then **Scan Face**.

To remove your data, click **Delete** next to your name.

Registering again with the same name replaces the saved template. Good lighting and a similar setup to when you registered give the best results.

### Measured accuracy

This project's recognition pipeline scored **99.17%** on the LFW benchmark, using OpenCV's published threshold of 0.363 without tuning it.

| Measure | Result |
|---|---|
| Pairs evaluated | 5,997 of 6,000 (no face found in 3 pairs) |
| Same person wrongly rejected | 1.53% |
| Different person wrongly accepted | 0.13% |

LFW compares single, mostly well-lit photos of public figures. The app compares averages of several webcam frames from a smaller group of people, so results in everyday use may differ.

**This isn't a security system.** It has no liveness check, so a photo of a registered person can pass. Don't use it to protect anything, and only register people who have agreed to it.

**Model scope:** the app uses MediaPipe's models to find and align the face, and the separate SFace model produces the template used for recognition. MediaPipe's model cards list identity recognition and surveillance as out of scope for their models, so this combination goes beyond the use MediaPipe intended. This project is a personal demo and must not be used for surveillance.

## Screenshots

Screenshots are coming soon.

## How it works

```text
Camera thread          Processing thread                      Browser
capture frame  ──►  mirror → detect → choose face → smooth  ──►  video stream
(keeps newest          → estimate pose → warp and blend          controls
 frame only)           → encode JPEG                             status
```

| Stage | Technique | Code |
|---|---|---|
| Landmark detection | MediaPipe Face Landmarker | `app/vision/detector.py` |
| Coordinate mapping | `x_px = x·W`, `y_px = y·H`, `z_px = z·W` | `app/imaging/transforms.py` |
| Temporal smoothing | One Euro filter | `app/vision/smoothing.py` |
| Head pose | Gram-Schmidt axes, rotation matrix `R = [X Y Z]` | `app/vision/pose.py` |
| Perspective | Depth scaling `k = d / (d + z)` | `app/vision/pose.py` |
| Resize, rotate, align | Homography with `cv2.warpPerspective` | `app/filters/overlay.py` |
| Downscaling quality | Mipmap pyramid | `app/filters/overlay.py` |
| ROI extraction | Clipped bounding box | `app/imaging/transforms.py` |
| Alpha blending | `C = F' + (1 − α)·B` with premultiplied alpha | `app/imaging/blending.py` |
| Face alignment | Similarity transform from 5 keypoints to a 112×112 image | `app/identity/embedder.py` |
| Face template | SFace network, 128 numbers per face | `app/identity/embedder.py` |
| Face comparison | Cosine similarity `a · b` against a threshold | `app/identity/store.py` |

### Project layout

```text
main.py                 Entry point
app/camera/             Webcam capture on a background thread
app/vision/             Detection, tracking, smoothing and head pose
app/imaging/            Coordinate mapping, geometry and alpha blending
app/identity/           Face templates, registration and recognition scans
app/filters/            AR filters and their placement on the face
app/pipeline/           Per-frame pipeline and camera session
app/web/                Local server and browser interface
app/utils/              FPS measurement
assets/                 Face model and generated filter artwork
tools/                  Script that draws the filter artwork
tests/                  Automated tests (no camera needed)
```

### Multiple faces

Filters apply to one primary face. The largest face is chosen first and stays primary while visible, so the filter doesn't jump between people. It moves only if another face becomes clearly larger or the primary face leaves. The app detects up to two faces.

## Privacy

- The server binds to `127.0.0.1`, so other devices can't reach it. It also rejects requests from other websites open in your browser.
- Frames stay in memory. The server never writes images or landmark points to disk.
- Photos are created only when you click Capture Photo, and your browser saves them to its download location. If that folder syncs to the cloud, your photos will sync too.
- Recognition runs only when you click Register Face or Scan Face, never in the background.
- Registering saves your name and a 128-number template, not a photo. The file is `face_profiles.json` in `%LOCALAPPDATA%r-face-filters` on Windows, `~/Library/Application Support/ar-face-filters` on macOS, or `~/.local/share/ar-face-filters` on Linux. It's outside the project folder, so it can't be committed by accident.
- The template file isn't encrypted, so anyone with access to your user account could copy it. A face template is biometric data, so treat it as sensitive.
- The camera turns off when you click Stop, quit the app, or close the browser tab (after about 8 seconds).

## Tests

```sh
python -m unittest discover tests
```

To redraw the filter artwork, run `python -m tools.generate_assets`.

## Troubleshooting

| Problem | Try this |
|---|---|
| "Could not open camera" | Close other apps using the camera, such as Zoom or Teams, then check camera access in your system privacy settings |
| Wrong camera | Stop the camera, click Find Cameras, and choose another one |
| Face not recognised | Check your lighting and face the camera, or click Register Face again with the same name |
| Low FPS | Close other heavy apps and improve your lighting |
| Filters don't appear | Face the camera with your whole face in view, in good light |
| Port already in use | The app may already be running. Otherwise, use `--port 9000` |
| `mp.solutions` errors | You're using old tutorial code. This project uses the newer Tasks API |
| OpenCV import conflicts | Run `pip uninstall opencv-python` so that only `opencv-contrib-python` remains |

## Platform support

| Platform | Status |
|---|---|
| Windows 11, Python 3.13 | Tested |
| Linux x86-64 or ARM64 | Dependencies are available, not yet tested |
| macOS on Apple Silicon | Dependencies are available, not yet tested |
| macOS on Intel | Not supported: MediaPipe 1.2.2 has no Intel Mac build |

## Packaging as an executable

The app could later be packaged with PyInstaller, including `assets/` and `app/web/static/` as data files and MediaPipe's native libraries. This hasn't been set up or tested yet.

## License

The code is under the [MIT License](LICENSE). The model files have their own terms: see [assets/models/NOTICE.md](assets/models/NOTICE.md).

## Credits

- Face model and landmark detection: [MediaPipe](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
- Face recognition model: [SFace from the OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) (Apache 2.0), by Yaoyao Zhong and others
- Filter artwork: generated by `tools/generate_assets.py`
- Smoothing: Casiez, Roussel and Vogel, "1€ Filter", CHI 2012
