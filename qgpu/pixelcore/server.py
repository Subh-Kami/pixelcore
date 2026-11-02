"""Loopback-only playable PixelCore UI. No host game simulation supplies model outputs."""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from qgpu.pixelcore.runtime import Game, verify

UI = Path(__file__).with_name("ui")


def serve(game, port, recording=None):
    lock = threading.Lock()
    replay = json.loads(recording.read_text()) if recording and recording.exists() else None
    if replay:
        verify(replay)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def local(self):
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            origin = self.headers.get("Origin")
            return self.headers.get("Host") in hosts and (not origin or origin in {f"http://{h}" for h in hosts})

        def send(self, code, body, mime="application/json"):
            data = json.dumps(body).encode() if mime == "application/json" else body
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; "
                             "script-src 'self'; connect-src 'self'; img-src 'self' blob: data:")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self.local():
                return self.send(403, {"error": "Use the local game URL"})
            path = urlparse(self.path).path
            files = {"/": ("index.html", "text/html; charset=utf-8"),
                     "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                     "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path in files:
                name, mime = files[path]
                return self.send(200, (UI / name).read_bytes(), mime)
            if path == "/api/status":
                return self.send(200, {**game.status(), "has_recording": bool(replay)})
            if path == "/api/recording":
                return self.send(200, replay) if replay else self.send(404, {"error": "No recording"})
            if path == "/api/trace":
                if lock.locked():
                    return self.send(409, {"error": "Wait for this tick to finish"})
                return self.send(200, game.export())
            self.send(404, {"error": "Not found"})

        def do_POST(self):
            if not self.local():
                return self.send(403, {"error": "Use the local game URL"})
            path = urlparse(self.path).path
            if path not in ("/api/tick", "/api/restart"):
                return self.send(404, {"error": "Not found"})
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 1024:
                    raise ValueError("Expected a small JSON body")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("Expected an object")
                action = body.get("action")
                if action not in (None, "left", "right", "stay", "reset"):
                    raise ValueError("Unknown button")
            except (ValueError, TypeError) as e:
                return self.send(400, {"error": str(e)})
            if not lock.acquire(blocking=False):
                return self.send(409, {"error": "A tick is already executing"})
            try:
                if path == "/api/restart":
                    game.reset()
                    return self.send(200, game.status())
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                connected = True

                def emit(event):
                    nonlocal connected
                    if connected:
                        try:
                            self.wfile.write(json.dumps(event).encode() + b"\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            connected = False
                try:
                    for event in game.step(action):
                        emit(event)
                except Exception as e:
                    emit({"kind": "error", "data": {"message": str(e)}})
            finally:
                lock.release()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"PixelCore / {game.meta['kind']} / {game.device}: http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--recording", type=Path)
    args = parser.parse_args()
    print("Loading PixelCore…", flush=True)
    serve(Game(args.checkpoint), args.port, args.recording)


if __name__ == "__main__":
    main()
