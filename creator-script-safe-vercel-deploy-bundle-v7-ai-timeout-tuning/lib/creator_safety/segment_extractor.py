#!/usr/bin/env python3
"""
Flagged segment extractor for Creator Script Safe.

No LLM calls. Uses Layer 1 findings to extract only the relevant
sentence/paragraph windows for downstream AI review.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


MAX_SEGMENTS = 8
DEFAULT_NEIGHBOR_SENTENCES = 1


PRIVACY_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone_cn_or_us": r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|1[3-9]\d{9})(?!\d)",
    "id_card_cn": r"(?<!\d)\d{17}[\dXx](?!\d)",
    "credit_card_like": r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)",
    "url": r"https?://[^\s]+|www\.[^\s]+",
    "off_platform_contact": r"(加我|私信|微信|VX|WhatsApp|Telegram|Line)[:：]?\s*[A-Za-z0-9_.-]{4,}",
}


def sentence_boundaries(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[。！？!?;\n]+", text):
        end = match.end()
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def expand_to_context(
    text: str,
    start: int,
    end: int,
    neighbor_sentences: int = DEFAULT_NEIGHBOR_SENTENCES,
) -> tuple[int, int]:
    spans = sentence_boundaries(text)
    for index, (sent_start, sent_end) in enumerate(spans):
        if sent_start <= start < sent_end:
            context_start_index = max(0, index - neighbor_sentences)
            context_end_index = min(len(spans) - 1, index + neighbor_sentences)
            return spans[context_start_index][0], spans[context_end_index][1]
    return max(0, start - 40), min(len(text), end + 40)


def find_match_spans(text: str, finding: dict[str, Any]) -> list[tuple[int, int]]:
    matched = str(finding.get("matched", "") or "")
    rule_id = str(finding.get("rule_id", "") or "")
    spans: list[tuple[int, int]] = []

    if matched and "***" not in matched:
        start = 0
        while True:
            index = text.find(matched, start)
            if index < 0:
                break
            spans.append((index, index + len(matched)))
            start = index + max(1, len(matched))

    if spans:
        return spans

    pattern = PRIVACY_PATTERNS.get(rule_id)
    if pattern:
        spans.extend((m.start(), m.end()) for m in re.finditer(pattern, text, flags=re.IGNORECASE))
    return spans


def merge_overlapping(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda item: (item["start"], item["end"]))
    merged: list[dict[str, Any]] = [ordered[0]]
    for segment in ordered[1:]:
        current = merged[-1]
        if segment["start"] <= current["end"]:
            current["end"] = max(current["end"], segment["end"])
            current["segment"] = segment["source_text"][current["start"] : current["end"]].strip()
            current["findings"].extend(segment["findings"])
        else:
            merged.append(segment)
    for segment in merged:
        segment.pop("source_text", None)
        deduped = []
        seen = set()
        for finding in segment["findings"]:
            key = (finding.get("rule_id"), finding.get("matched"))
            if key not in seen:
                seen.add(key)
                deduped.append(finding)
        segment["findings"] = deduped
    return merged[:MAX_SEGMENTS]


def extract_flagged_segments(
    text: str,
    layer1: dict[str, Any],
    neighbor_sentences: int = DEFAULT_NEIGHBOR_SENTENCES,
) -> dict[str, Any]:
    raw_segments: list[dict[str, Any]] = []
    for finding in layer1.get("findings", []):
        for match_start, match_end in find_match_spans(text, finding):
            start, end = expand_to_context(text, match_start, match_end, neighbor_sentences)
            raw_segments.append(
                {
                    "start": start,
                    "end": end,
                    "segment": text[start:end].strip(),
                    "match_start": match_start,
                    "match_end": match_end,
                    "findings": [finding],
                    "source_text": text,
                }
            )

    segments = merge_overlapping(raw_segments)
    original_chars = len(text)
    segment_chars = sum(len(segment["segment"]) for segment in segments)
    savings_ratio = 0.0
    if original_chars:
        savings_ratio = max(0.0, 1 - (segment_chars / original_chars))

    return {
        "mode": "flagged_segments_only" if segments else "full_script_fallback",
        "segments": segments,
        "segment_count": len(segments),
        "original_chars": original_chars,
        "segment_chars": segment_chars,
        "estimated_char_savings_ratio": round(savings_ratio, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Layer 1 flagged segments.")
    parser.add_argument("file", help="Text file to extract from.")
    parser.add_argument("--layer1-json", required=True, help="Layer 1 JSON string or path to JSON file.")
    args = parser.parse_args()

    text = Path(args.file).read_text(encoding="utf-8")
    layer1_path = Path(args.layer1_json)
    if layer1_path.exists():
        layer1 = json.loads(layer1_path.read_text(encoding="utf-8"))
    else:
        layer1 = json.loads(args.layer1_json)
    print(json.dumps(extract_flagged_segments(text, layer1), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
