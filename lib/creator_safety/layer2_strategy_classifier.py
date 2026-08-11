#!/usr/bin/env python3
"""
Local Layer 2 strategy and classification filter for creator scripts.

No LLM calls. Combines Layer 1 output, text features, content metadata,
account profile, and device signals into a calibrated-looking risk score.

It can run as:
- a heuristic scorer with built-in feature weights
- a configurable lightweight classifier by passing a JSON weights file
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


ROUTE_FULL_REVIEW = "Full review"
ROUTE_CHEAP_PASS = "Cheap pass"
ROUTE_REWRITE_FIRST = "Rewrite first"
ROUTE_REFUSE = "Refuse/redirect"
ROUTE_NEEDS_CONTEXT = "Needs more context"


SENSITIVE_TOPICS = {
    "health": ["减肥", "祛痘", "脱发", "睡眠", "焦虑", "抑郁", "药", "治疗", "症状", "doctor", "cure", "weight loss"],
    "finance": ["赚钱", "收益", "投资", "贷款", "副业", "稳赚", "回报", "crypto", "loan", "income", "profit"],
    "legal": ["赔偿", "起诉", "合同", "律师", "移民", "签证", "lawsuit", "visa", "legal"],
    "minors": ["学生", "未成年", "孩子", "儿童", "校园", "teen", "kid", "school"],
    "adult": ["裸", "性感", "私密", "约会", "成人", "sexy", "nude", "adult"],
    "tragedy": ["死亡", "自杀", "灾难", "事故", "血", "suicide", "death", "disaster"],
    "regulated_goods": ["电子烟", "烟", "酒", "处方药", "枪", "赌博", "vape", "alcohol", "weapon", "gambling"],
}


CLAIM_PATTERNS = {
    "guarantee": r"保证|一定|100%|永久|零风险|稳赚|guarantee|always|never|risk[- ]?free",
    "medical": r"治愈|根治|治疗|临床|医生推荐|处方|cure|treat|clinical|doctor recommended",
    "income": r"月入|日赚|躺赚|收益翻倍|财富自由|make \$?\d+|earn \$?\d+|passive income",
    "scarcity": r"最后\d+个|仅限今天|马上下架|倒计时|last chance|only today|limited stock",
    "testimonial": r"客户反馈|真实案例|买家秀|before and after|testimonial|review says",
    "sponsorship_missing": r"合作|赞助|佣金|affiliate|sponsored|ad\b",
}


DEFAULT_WEIGHTS = {
    "bias": -2.15,
    "layer1_review": 0.65,
    "layer1_rewrite": 1.15,
    "layer1_refuse": 3.0,
    "topic_health": 0.65,
    "topic_finance": 0.75,
    "topic_legal": 0.7,
    "topic_minors": 0.75,
    "topic_adult": 0.8,
    "topic_tragedy": 0.85,
    "topic_regulated_goods": 0.95,
    "claim_guarantee": 0.85,
    "claim_medical": 0.9,
    "claim_income": 0.85,
    "claim_scarcity": 0.45,
    "claim_testimonial": 0.4,
    "claim_sponsorship_missing": 0.25,
    "commerce": 0.35,
    "minor_audience": 0.7,
    "prior_warning": 0.8,
    "high_post_volume": 0.45,
    "new_account": 0.35,
    "duplicate_device": 0.55,
    "vpn_or_proxy": 0.35,
    "many_accounts_same_device": 0.75,
    "unknown_platform": 0.3,
    "missing_product_evidence": 0.55,
    "off_platform_conversion": 0.65,
}


REQUIRED_CONTEXT_FOR_HIGH_STAKES = {
    "health": ["claim_evidence"],
    "finance": ["claim_evidence"],
    "regulated_goods": ["market"],
    "minors": ["audience_age"],
}


@dataclass
class Signal:
    name: str
    value: float
    weight: float
    contribution: float
    note: str


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def load_json_arg(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def has_any(text: str, terms: list[str]) -> bool:
    lowered = normalize(text)
    return any(term.lower() in lowered for term in terms)


def regex_hit(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def add_signal(
    signals: list[Signal],
    weights: dict[str, float],
    name: str,
    value: float,
    note: str,
) -> None:
    if value <= 0:
        return
    weight = float(weights.get(name, 0.0))
    signals.append(Signal(name=name, value=value, weight=weight, contribution=value * weight, note=note))


def extract_signals(
    text: str,
    layer1: dict[str, Any],
    metadata: dict[str, Any],
    weights: dict[str, float],
) -> list[Signal]:
    signals: list[Signal] = []
    layer1_result = layer1.get("result")

    add_signal(signals, weights, "layer1_review", 1 if layer1_result == "Review needed" else 0, "Layer 1 requested review.")
    add_signal(signals, weights, "layer1_rewrite", 1 if layer1_result == "Immediate rewrite" else 0, "Layer 1 requested immediate rewrite.")
    add_signal(signals, weights, "layer1_refuse", 1 if layer1_result == "Refuse/redirect" else 0, "Layer 1 found refusal-level risk.")

    for topic, terms in SENSITIVE_TOPICS.items():
        add_signal(
            signals,
            weights,
            f"topic_{topic}",
            1 if has_any(text, terms) or metadata.get("category") == topic else 0,
            f"Sensitive topic signal: {topic}.",
        )

    for claim, pattern in CLAIM_PATTERNS.items():
        if claim == "sponsorship_missing" and metadata.get("sponsorship") is True:
            continue
        add_signal(
            signals,
            weights,
            f"claim_{claim}",
            1 if regex_hit(text, pattern) else 0,
            f"Claim pattern signal: {claim}.",
        )

    add_signal(signals, weights, "commerce", 1 if metadata.get("commerce") or metadata.get("sponsorship") else 0, "Commercial or sponsored content.")
    add_signal(signals, weights, "minor_audience", 1 if str(metadata.get("audience_age", "")).lower() in {"minor", "teen", "under_18", "children"} else 0, "Audience may include minors.")
    add_signal(signals, weights, "unknown_platform", 1 if not metadata.get("platform") else 0, "Platform is unknown.")
    add_signal(signals, weights, "missing_product_evidence", 1 if metadata.get("commerce") and not metadata.get("claim_evidence") else 0, "Commercial claim lacks evidence metadata.")
    add_signal(signals, weights, "off_platform_conversion", 1 if metadata.get("off_platform_conversion") else 0, "Potential off-platform conversion.")

    account = metadata.get("account", {}) or {}
    add_signal(signals, weights, "prior_warning", min(float(account.get("prior_warnings", 0)), 3) / 3, "Account has prior warning history.")
    add_signal(signals, weights, "high_post_volume", 1 if float(account.get("posts_last_24h", 0)) >= 20 else 0, "High posting volume.")
    add_signal(signals, weights, "new_account", 1 if float(account.get("account_age_days", 9999)) <= 14 else 0, "New account.")

    device = metadata.get("device", {}) or {}
    add_signal(signals, weights, "duplicate_device", 1 if device.get("duplicate_device") else 0, "Device fingerprint appears duplicated.")
    add_signal(signals, weights, "vpn_or_proxy", 1 if device.get("vpn_or_proxy") else 0, "VPN/proxy signal.")
    add_signal(signals, weights, "many_accounts_same_device", 1 if float(device.get("accounts_on_device", 1)) >= 4 else 0, "Many accounts share this device.")

    return signals


def missing_context(text: str, metadata: dict[str, Any]) -> list[str]:
    missing: set[str] = set()
    for topic, terms in SENSITIVE_TOPICS.items():
        topic_present = has_any(text, terms) or metadata.get("category") == topic
        if not topic_present:
            continue
        for key in REQUIRED_CONTEXT_FOR_HIGH_STAKES.get(topic, []):
            value = metadata.get(key)
            if value is None or value == "" or value == []:
                missing.add(key)
    if metadata.get("commerce") and metadata.get("sponsorship") is None:
        missing.add("sponsorship")
    return sorted(missing)


def classify(score: float, layer1_result: str | None, missing: list[str]) -> tuple[str, str]:
    if layer1_result == "Refuse/redirect":
        return "High", ROUTE_REFUSE
    if missing and score >= 0.45:
        return "Needs more context", ROUTE_NEEDS_CONTEXT
    if score >= 0.78:
        return "High", ROUTE_FULL_REVIEW
    if score >= 0.48:
        return "Medium", ROUTE_FULL_REVIEW
    if layer1_result == "Immediate rewrite":
        return "Medium", ROUTE_REWRITE_FIRST
    if layer1_result == "Review needed":
        return "Medium", ROUTE_FULL_REVIEW
    return "Low", ROUTE_CHEAP_PASS


def check(
    text: str,
    layer1: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    layer1 = layer1 or {}
    metadata = metadata or {}
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    signals = extract_signals(text, layer1, metadata, weights)
    raw_score = float(weights.get("bias", 0.0)) + sum(signal.contribution for signal in signals)
    probability = round(sigmoid(raw_score), 4)
    missing = missing_context(text, metadata)
    risk_level, routing = classify(probability, layer1.get("result"), missing)

    top_signals = sorted(signals, key=lambda s: s.contribution, reverse=True)[:8]
    return {
        "layer": "Layer 2: Strategy & Classification",
        "risk_probability": probability,
        "risk_level": risk_level,
        "routing": routing,
        "missing_context": missing,
        "top_signals": [asdict(signal) for signal in top_signals],
        "all_signal_count": len(signals),
        "next_step": {
            ROUTE_CHEAP_PASS: "Return fast local result unless the user requested full creator-safety review.",
            ROUTE_FULL_REVIEW: "Send Layer 1 + Layer 2 results into creator-script-safe.",
            ROUTE_REWRITE_FIRST: "Rewrite flagged wording, re-run Layer 1 and Layer 2, then decide whether full review is needed.",
            ROUTE_REFUSE: "Do not process the unsafe request as written; redirect to a safer alternative.",
            ROUTE_NEEDS_CONTEXT: "Ask only for the missing context fields before deciding whether full review is needed.",
        }[routing],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Layer 2 strategy/classification filter.")
    parser.add_argument("file", nargs="?", help="Text file to scan. Reads stdin if omitted.")
    parser.add_argument("--layer1-json", help="Layer 1 JSON string or path to JSON file.")
    parser.add_argument("--metadata", help="Metadata JSON string or path to JSON file.")
    parser.add_argument("--weights", help="Optional weights JSON string or path to JSON file.")
    args = parser.parse_args()

    text = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    layer1 = load_json_arg(args.layer1_json)
    metadata = load_json_arg(args.metadata)
    weights = load_json_arg(args.weights)

    print(json.dumps(check(text, layer1=layer1, metadata=metadata, weights=weights), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
