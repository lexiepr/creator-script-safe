from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIB_PATH = PROJECT_ROOT / "lib"
sys.path.insert(0, str(LIB_PATH))

from creator_safety.creator_safety_pipeline import decide  # noqa: E402


MAX_BODY_BYTES = 64 * 1024


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > MAX_BODY_BYTES:
            self.write_json(413, {"error": "invalid_body_size"})
            return

        try:
            body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.write_json(400, {"error": "invalid_json"})
            return

        script = str(payload.get("script", "")).strip()
        if not script:
            self.write_json(400, {"error": "script_required"})
            return

        metadata = payload.get("metadata") or {}
        force_full_review = bool(payload.get("force_full_review", False))
        auto_rewrite = bool(payload.get("auto_rewrite", True))

        result = decide(
            script,
            metadata=metadata,
            force_full_review=force_full_review,
            auto_rewrite=auto_rewrite,
        )
        self.write_json(200, result)

    def send_cors_headers(self):
        origin = self.headers.get("Origin", "")
        allowed_origins = {
            "https://www.creatorscriptsafe.xyz",
            "https://creatorscriptsafe.xyz",
        }
        if origin in allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
        else:
            self.send_header("Access-Control-Allow-Origin", "https://www.creatorscriptsafe.xyz")
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def write_json(self, status: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

