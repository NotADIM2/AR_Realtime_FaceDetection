"""The local web server. It binds to 127.0.0.1 only and has no outside connections."""

from urllib.parse import urlsplit

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from app import config
from app.camera.capture import CameraError, scan_cameras
from app.pipeline.engine import IdentityError
from app.vision.detector import ModelError

_LOCAL_HOSTS = {"127.0.0.1", "localhost"}


def create_app(engine):
    app = Flask(__name__, static_folder="static", static_url_path="/static")

    @app.before_request
    def block_other_sites():
        """Stop other websites open in the browser from reaching the camera.

        Loopback binding blocks other devices, but any page in a browser on this
        computer can still send requests to 127.0.0.1. These checks reject those
        requests, including DNS rebinding attempts that use a different Host name.
        """
        if urlsplit(f"//{request.host}").hostname not in _LOCAL_HOSTS:
            abort(403)
        if request.headers.get("Sec-Fetch-Site") not in (None, "same-origin", "none"):
            abort(403)
        origin = request.headers.get("Origin")
        if origin and urlsplit(origin).netloc != request.host:
            abort(403)

    @app.after_request
    def add_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'"
        )
        if request.path.startswith(("/api/", "/stream")):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/assets/filters/<path:name>")
    def filter_asset(name):
        return send_from_directory(config.FILTER_DIR, name)

    @app.get("/api/state")
    def state():
        return jsonify(engine.status() | {"filters": engine.filters.describe()})

    @app.post("/api/start")
    def start():
        data = request.get_json(silent=True) or {}
        index = data.get("camera", 0)
        if not isinstance(index, int) or not 0 <= index <= config.MAX_CAMERA_INDEX:
            return jsonify(error="Invalid camera."), 400
        try:
            engine.start(index)
        except (CameraError, ModelError) as exc:
            return jsonify(error=str(exc)), 503
        return jsonify(ok=True)

    @app.post("/api/stop")
    def stop():
        engine.stop()
        return jsonify(ok=True)

    @app.post("/api/filter")
    def set_filter():
        data = request.get_json(silent=True) or {}
        if not engine.set_filter(data.get("key")):
            return jsonify(error="That filter is unavailable."), 400
        return jsonify(ok=True)

    @app.post("/api/options")
    def set_options():
        data = request.get_json(silent=True) or {}
        engine.set_options(landmarks=data.get("landmarks"), axes=data.get("axes"))
        return jsonify(ok=True)

    @app.post("/api/reset")
    def reset():
        engine.reset()
        return jsonify(ok=True)

    @app.post("/api/cameras/find")
    def find_cameras():
        if engine.demo_image:
            return jsonify(error="Demo mode uses an image instead of a camera."), 409
        # Opening a camera that is in use would fail, so searching needs it stopped.
        if engine.status()["state"] in ("starting", "running"):
            return jsonify(error="Stop the camera before searching for other cameras."), 409
        cameras = scan_cameras()
        engine.set_cameras(cameras)
        return jsonify(cameras=cameras)

    def run_identity_action(action):
        try:
            action()
        except IdentityError as exc:
            return jsonify(error=str(exc)), 409
        except ModelError as exc:
            return jsonify(error=str(exc)), 503
        return jsonify(ok=True)

    @app.post("/api/identity/enroll")
    def identity_enroll():
        data = request.get_json(silent=True) or {}
        name = data.get("name")
        name = " ".join(name.split()) if isinstance(name, str) else ""
        if not name or len(name) > config.MAX_NAME_LENGTH or not name.isprintable():
            return jsonify(error=f"Enter a name of up to {config.MAX_NAME_LENGTH} characters."), 400
        return run_identity_action(lambda: engine.start_enroll(name))

    @app.post("/api/identity/scan")
    def identity_scan():
        return run_identity_action(engine.start_scan)

    @app.post("/api/identity/cancel")
    def identity_cancel():
        engine.cancel_identity_session()
        return jsonify(ok=True)

    @app.post("/api/identity/delete")
    def identity_delete():
        data = request.get_json(silent=True) or {}
        name = data.get("name")
        if not isinstance(name, str) or not engine.delete_profile(name):
            return jsonify(error="That face profile doesn't exist."), 404
        return jsonify(ok=True)

    @app.post("/api/capture")
    def capture():
        png = engine.capture_png()
        if png is None:
            return jsonify(error="Start the camera before capturing a photo."), 409
        return Response(png, mimetype="image/png")

    @app.get("/stream")
    def stream():
        return Response(engine.stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app
