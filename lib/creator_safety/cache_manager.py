#!/usr/bin/env python3
"""
Optional cache layer for Creator Script Safe.

Uses Upstash Redis REST when env vars are present:
- UPSTASH_REDIS_REST_URL
- UPSTASH_REDIS_REST_TOKEN

Falls back to best-effort in-memory cache for local/MVP use.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any


PIPELINE_CACHE_VERSION = "creator-safety-v4-ai-routing"
DEFAULT_TTL_SECONDS = 24 * 60 * 60
_MEMORY_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def stable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "platform",
        "surface",
        "market",
        "language",
        "category",
        "commerce",
        "sponsorship",
        "claim_evidence",
        "audience_age",
        "off_platform_conversion",
    ]
    return {key: metadata.get(key) for key in keys if key in metadata}


def build_cache_key(
    text: str,
    metadata: dict[str, Any] | None = None,
    force_full_review: bool = False,
    auto_rewrite: bool = True,
) -> str:
    payload = {
        "version": PIPELINE_CACHE_VERSION,
        "text": normalize_text(text),
        "metadata": stable_metadata(metadata or {}),
        "force_full_review": force_full_review,
        "auto_rewrite": auto_rewrite,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"css:decision:{digest}"


def ttl_for_decision(decision: str | None) -> int:
    if decision == "refuse_or_redirect":
        return 30 * 24 * 60 * 60
    if decision in {"call_creator_script_safe", "rewrite_locally_first", "ask_for_context"}:
        return 7 * 24 * 60 * 60
    return DEFAULT_TTL_SECONDS


def upstash_configured() -> bool:
    return bool(os.getenv("UPSTASH_REDIS_REST_URL") and os.getenv("UPSTASH_REDIS_REST_TOKEN"))


def _upstash_command(command: list[Any], timeout: float = 2.0) -> Any:
    url = os.environ["UPSTASH_REDIS_REST_URL"].rstrip("/")
    token = os.environ["UPSTASH_REDIS_REST_TOKEN"]
    request = urllib.request.Request(
        url,
        data=json.dumps(command, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    if "error" in data:
        raise RuntimeError(data["error"])
    return data.get("result")


def _memory_get(key: str) -> dict[str, Any] | None:
    item = _MEMORY_CACHE.get(key)
    if not item:
        return None
    expires_at, value = item
    if expires_at <= time.time():
        _MEMORY_CACHE.pop(key, None)
        return None
    return json.loads(json.dumps(value, ensure_ascii=False))


def _memory_set(key: str, value: dict[str, Any], ttl_seconds: int) -> None:
    cloned = json.loads(json.dumps(value, ensure_ascii=False))
    _MEMORY_CACHE[key] = (time.time() + ttl_seconds, cloned)


def get_cached_result(key: str) -> tuple[dict[str, Any] | None, str]:
    if upstash_configured():
        try:
            cached = _upstash_command(["GET", key])
            if cached:
                return json.loads(cached), "upstash"
            return None, "upstash"
        except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError, OSError):
            pass

    cached = _memory_get(key)
    return cached, "memory"


def set_cached_result(key: str, value: dict[str, Any], ttl_seconds: int | None = None) -> str:
    ttl = ttl_seconds or ttl_for_decision(value.get("decision"))
    cached_value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    if upstash_configured():
        try:
            _upstash_command(["SET", key, cached_value, "EX", ttl])
            return "upstash"
        except (urllib.error.URLError, TimeoutError, RuntimeError, OSError):
            pass

    _memory_set(key, value, ttl)
    return "memory"


def cache_status(hit: bool, key: str, backend: str) -> dict[str, Any]:
    return {
        "hit": hit,
        "backend": backend,
        "key": key,
        "version": PIPELINE_CACHE_VERSION,
    }
