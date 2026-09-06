from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        upstash_url = os.getenv("UPSTASH_REDIS_REST_URL")
        upstash_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
        openai_key = os.getenv("OPENAI_API_KEY")
        self.write_json(
            200,
            {
                "has_upstash_url": bool(upstash_url),
                "has_upstash_token": bool(upstash_token),
                "has_openai_key": bool(openai_key),
                "upstash_url_length": len(upstash_url or ""),
                "upstash_token_length": len(upstash_token or ""),
                "openai_key_length": len(openai_key or ""),
                "environment_hint": os.getenv("VERCEL_ENV", "unknown"),
            },
        )

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

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
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def write_json(self, status: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
