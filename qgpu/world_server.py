"""Loopback-only viewer and recording CLI for the WeightWorld experiment."""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from qgpu.world import MAX_FRAMES, World, verify_trace

UI = Path(__file__).with_name("world_ui")


def serve(world, port=8765):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code, payload, mime="application/json"):
            data = json.dumps(payload).encode() if mime == "application/json" else payload
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; "
                             "script-src 'self'; connect-src 'self'; img-src 'self' blob: data:")
            self.end_headers()
            self.wfile.write(data)

        def local(self):
            host = self.headers.get("Host", "")
            allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
            origin = self.headers.get("Origin")
            return host in allowed and (not origin or origin in {f"http://{h}" for h in allowed})

        def do_GET(self):
            if not self.local():
                return self.send(403, {"error": "Use the local viewer URL"})
            path = urlparse(self.path).path
            assets = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path in assets:
                name, mime = assets[path]
                return self.send(200, (UI / name).read_bytes(), mime)
            if path in ("/api/status", "/api/trace"):
                if not lock.acquire(blocking=False):
                    return self.send(409, {"error": "A step is in progress"})
                try:
                    return self.send(200, world.status() if path == "/api/status" else world.export())
                finally:
                    lock.release()
            self.send(404, {"error": "Not found"})

        def do_POST(self):
            if not self.local():
                return self.send(403, {"error": "Use the local viewer URL"})
            path = urlparse(self.path).path
            if path not in ("/api/step", "/api/reset", "/api/verify"):
                return self.send(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= (4_000_000 if path == "/api/verify" else 4096):
                    raise ValueError("Expected a small JSON body")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                if path == "/api/verify":
                    return self.send(200, verify_trace(body))
                seed = body.get("seed", world.seed)
                if type(seed) is not int or not 0 <= seed <= 1000000:
                    raise ValueError("Seed must be an integer from 0 to 1000000")
            except (ValueError, TypeError, KeyError, IndexError, AttributeError) as e:
                return self.send(400, {"error": str(e)})
            if not lock.acquire(blocking=False):
                return self.send(409, {"error": "A step is already running"})
            try:
                if path == "/api/reset":
                    world.reset(seed)
                    return self.send(200, world.status())
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
                            # Finish the current step even if a tab closes; keep state coherent.
                            connected = False

                try:
                    for event in world.step():
                        emit(event)
                except Exception as e:
                    emit({"kind": "error", "data": {"message": str(e)}})
            finally:
                lock.release()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"WeightWorld · {world.backend} · http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "record"):
        p = sub.add_parser(name)
        p.add_argument("--backend", choices=("reference", "model"), default="reference")
        p.add_argument("--merged", type=Path)
        p.add_argument("--seed", type=int, default=7)
        if name == "serve":
            p.add_argument("--port", type=int, default=8765)
        else:
            p.add_argument("--frames", type=int, default=4)
            p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("verify")
    p.add_argument("trace", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify_trace(json.loads(args.trace.read_text())), indent=2))
        return
    if args.command == "record" and not 1 <= args.frames <= MAX_FRAMES:
        parser.error(f"--frames must be 1..{MAX_FRAMES}")
    if not 0 <= args.seed <= 1000000:
        parser.error("--seed must be 0..1000000")
    print(f"Loading {args.backend} backend…", flush=True)
    world = World(args.backend, args.merged, args.seed)
    if args.command == "serve":
        serve(world, args.port)
    else:
        for _ in range(args.frames):
            for event in world.step():
                if event["kind"] == "phase":
                    print(f"Frame {world.frame + 1} / {event['data']['name']}", flush=True)
                if event["kind"] == "phase_done":
                    d = event["data"]
                    print(f"  verified {d['generated_tokens']} tokens in {d['seconds']:.2f}s", flush=True)
                if event["kind"] == "frame":
                    print(f"Frame {world.frame}: {world.last['generated_tokens']} execution tokens; "
                          f"{world.last['host_memory_words']} host memory words; {world.backend}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(world.export(), indent=2))


if __name__ == "__main__":
    main()
