#!/usr/bin/env python3
"""
Local input type classifier for Creator Script Safe.

No LLM calls. Classifies user input as:
- idea: rough concept that should go straight to AI generation
- script: draft content that should go through Layer 1 / Layer 2 checks
- unclear: ambiguous input; handle conservatively as a script for local screening
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


IDEA = "idea"
SCRIPT = "script"
UNCLEAR = "unclear"


IDEA_PATTERNS = [
    r"帮我写",
    r"帮我生成",
    r"想做一期",
    r"想拍",
    r"想分享",
    r"主题是",
    r"idea",
    r"concept",
    r"write a script",
    r"make a video about",
    r"turn.*into.*script",
]


SCRIPT_PATTERNS = [
    r"大家好",
    r"今天给大家",
    r"欢迎来到",
    r"点击",
    r"下单",
    r"评论区",
    r"关注我",
    r"link in bio",
    r"comment below",
    r"use code",
    r"buy now",
    r"100%",
    r"保证",
    r"治愈",
    r"根治",
    r"稳赚",
    r"零风险",
    r"guarantee",
    r"risk[- ]?free",
]


STRUCTURE_PATTERNS = [
    r"hook[:：]",
    r"body[:：]",
    r"cta[:：]",
    r"开场[:：]",
    r"正文[:：]",
    r"结尾[:：]",
    r"镜头\s*\d+",
    r"第\s*\d+\s*段",
]


@dataclass
class Signal:
    name: str
    value: float
    note: str


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def count_sentences(text: str) -> int:
    parts = re.split(r"[。！？!?.\n]+", text)
    return len([part for part in parts if part.strip()])


def pattern_hits(text: str, patterns: list[str]) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)]


def classify(text: str) -> dict:
    stripped = text.strip()
    normalized = normalize(stripped)
    char_count = len(stripped)
    sentence_count = count_sentences(stripped)
    line_count = len([line for line in stripped.splitlines() if line.strip()])

    idea_hits = pattern_hits(normalized, IDEA_PATTERNS)
    script_hits = pattern_hits(normalized, SCRIPT_PATTERNS)
    structure_hits = pattern_hits(normalized, STRUCTURE_PATTERNS)

    idea_score = 0.0
    script_score = 0.0
    signals: list[Signal] = []

    if char_count <= 80:
        idea_score += 0.18
        signals.append(Signal("short_input", 0.18, "Short input can mean a rough idea."))
    elif char_count >= 180:
        script_score += 0.35
        signals.append(Signal("long_input", 0.35, "Long input often means a draft script."))

    if sentence_count <= 2:
        idea_score += 0.1
        signals.append(Signal("few_sentences", 0.1, "Few sentences can suggest an idea."))
    elif sentence_count >= 4:
        script_score += 0.25
        signals.append(Signal("many_sentences", 0.25, "Many sentences suggest a script."))

    if line_count >= 3:
        script_score += 0.25
        signals.append(Signal("multi_line", 0.25, "Multiple lines suggest script structure."))

    if idea_hits:
        idea_score += min(0.45, 0.18 * len(idea_hits))
        signals.append(Signal("idea_patterns", min(0.45, 0.18 * len(idea_hits)), ", ".join(idea_hits)))

    if script_hits:
        script_score += min(0.45, 0.15 * len(script_hits))
        signals.append(Signal("script_patterns", min(0.45, 0.15 * len(script_hits)), ", ".join(script_hits)))

    if structure_hits:
        script_score += min(0.5, 0.2 * len(structure_hits))
        signals.append(Signal("script_structure", min(0.5, 0.2 * len(structure_hits)), ", ".join(structure_hits)))

    margin = abs(idea_score - script_score)
    has_explicit_idea_intent = bool(idea_hits)
    if has_explicit_idea_intent and idea_score >= 0.45 and idea_score > script_score + 0.15:
        input_type = IDEA
        confidence = min(0.98, 0.55 + margin)
    elif script_score >= 0.5 and script_score > idea_score + 0.15:
        input_type = SCRIPT
        confidence = min(0.98, 0.55 + margin)
    else:
        input_type = UNCLEAR
        confidence = max(0.35, min(0.65, 0.5 + margin / 2))

    return {
        "input_type": input_type,
        "confidence": round(confidence, 3),
        "idea_score": round(idea_score, 3),
        "script_score": round(script_score, 3),
        "signals": [asdict(signal) for signal in signals],
        "routing_note": {
            IDEA: "Skip Layer 1/2 and use AI generation with the Creator Script Safe policy prefix.",
            SCRIPT: "Run Layer 1 and Layer 2 before any LLM call.",
            UNCLEAR: "Treat conservatively as a script and run local screening first.",
        }[input_type],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify input as idea, script, or unclear.")
    parser.add_argument("file", nargs="?", help="Text file to classify. Reads stdin if omitted.")
    args = parser.parse_args()

    text = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    print(json.dumps(classify(text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
