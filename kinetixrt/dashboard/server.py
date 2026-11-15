"""Lightweight live dashboard (standard library HTTP server, no extra deps).

Every value it shows is read from the runtime's telemetry; nothing is
generated for display.
"""

from __future__ import annotations

import argparse
import json
import struct
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from kinetixrt.dashboard.demo import LiveDemo

INDEX = Path(__file__).parent / "index.html"


def png_bytes(img: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 array as PNG (zlib only)."""
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 3)) + chunk(b"IEND", b"")


def build_state(demo: LiveDemo) -> dict:
    rt = demo.runtime
    state: dict = {"status": demo.status, "controller": demo.controller_name, "presets": list(demo.presets), "episode": demo.episode, "aggregates": demo.aggregates, "kind": demo.kind, "dynamic": demo.dynamic}
    if rt is None:
        return state
    snap = rt.telemetry.snapshot(n_ticks=300, n_inferences=40, n_events=30)
    keep = ("tick", "source", "chunk_age", "remaining", "phase", "decision", "request", "steps", "horizon", "reasons", "surprise", "disagreement", "step_wall_ms", "planned_speed", "ticks_to_event")
    state.update(
        meta=snap["meta"],
        counters=snap["counters"],
        ticks=[{k: t.get(k) for k in keep} for t in snap["ticks"]],
        inferences=snap["inferences"],
        events=snap["events"],
        predictor={str(k): v for k, v in rt.predictor.snapshot().items()},
        surprise_threshold=getattr(getattr(rt.controller, "cfg", None), "surprise_threshold", None),
    )
    return state


def make_handler(demo: LiveDemo) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # noqa: ANN002
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send(200, json.dumps(build_state(demo), default=str).encode(), "application/json")
            elif path == "/frame.png":
                frame = demo.frame
                if frame is None:
                    self._send(404, b"", "text/plain")
                else:
                    self._send(200, png_bytes(np.ascontiguousarray(frame)), "image/png")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/api/controller":
                    demo.request_controller(str(body["name"]))
                elif self.path == "/api/scenario":
                    demo.dynamic = bool(body["dynamic"])
                else:
                    self._send(404, b"not found", "text/plain")
                    return
                self._send(200, b'{"ok": true}', "application/json")
            except (KeyError, ValueError) as exc:
                self._send(400, json.dumps({"error": repr(exc)}).encode(), "application/json")

    return Handler


def serve(demo: LiveDemo, host: str = "127.0.0.1", port: int = 8050) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(demo))
    threading.Thread(target=server.serve_forever, name="krt-dashboard", daemon=True).start()
    return server


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="krt-dashboard", description=__doc__)
    p.add_argument("--demo", choices=["toy", "libero"], default="toy", help="toy = synthetic CPU demo; libero = real SmolVLA + LIBERO")
    p.add_argument("--controller", default="physical_aware")
    p.add_argument("--static", action="store_true", help="disable the dynamic-scene disturbance")
    p.add_argument("--device", default="auto")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8050)
    args = p.parse_args(argv)
    demo = LiveDemo(args.demo, args.controller, dynamic=not args.static, device=args.device)
    demo.start()
    server = serve(demo, args.host, args.port)
    print(f"KinetixRT dashboard: http://{args.host}:{args.port}  (demo={args.demo}; Ctrl-C to stop)")
    try:
        demo.join()
    except KeyboardInterrupt:
        pass
    finally:
        demo.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
