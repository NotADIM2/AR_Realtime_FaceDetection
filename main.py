"""Application entry point: starts the local server and opens the browser."""

import argparse
import logging
import os
import sys
import threading
import webbrowser

# Hide MediaPipe's informational console output. Must be set before it is imported.
os.environ.setdefault("GLOG_minloglevel", "2")

from werkzeug.serving import make_server

from app import config
from app.pipeline.engine import Engine
from app.web.server import create_app


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time AR Face Filters")
    parser.add_argument("--port", type=int, default=config.PORT, help="local port to serve on")
    parser.add_argument("--image", help="use a still image instead of a webcam")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    return parser.parse_args()


def main():
    args = parse_args()
    # The page polls for status twice a second, so per-request logs would flood the console.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    engine = Engine(demo_image=args.image)

    try:
        server = make_server(config.HOST, args.port, create_app(engine), threaded=True)
    except OSError:
        print(f"Port {args.port} is already in use. Is the app already running?")
        print(f"Open http://{config.HOST}:{args.port} or start with a different --port.")
        return 1

    url = f"http://{config.HOST}:{args.port}"
    print(f"AR Face Filters is running at {url}")
    print("It is only available on this computer. Press Ctrl+C to quit.", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # Always release the webcam on the way out.
        engine.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
