#!/usr/bin/env python3
"""
Deterministic Layer 1 machine filter for creator scripts.

No LLM calls. Uses keyword rules, regex patterns, privacy detection,
spam/repetition checks, and simple score routing.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Iterable


ROUTE_PASS = "Pass"
ROUTE_REVIEW = "Review needed"
ROUTE_REWRITE = "Immediate rewrite"
ROUTE_REFUSE = "Refuse/redirect"


@dataclass
class Finding:
    kind: str
    severity: str
    rule_id: str
    matched: str
    action: str
    note: str


KEYWORD_RULES = [
    {
        "id": "platform_evasion",
        "severity": "block",
        "terms": [
            "绕过审核",
            "规避审核",
            "躲过风控",
            "不被封号",
            "ban-proof",
            "avoid moderation",
            "evade moderation",
        ],
        "note": "请求可能涉及绕过平台审核或风控。",
    },
    {
        "id": "fake_claims",
        "severity": "rewrite",
        "terms": [
            "100%有效",
            "永久有效",
            "保证瘦",
            "保证赚钱",
            "稳赚不赔",
            "零风险",
            "治愈",
            "根治",
            "guaranteed",
            "risk-free",
        ],
        "note": "绝对化、治疗、收益或保证类表达需要改写。",
    },
    {
        "id": "fake_social_proof",
        "severity": "rewrite",
        "terms": [
            "假装客户",
            "伪造评论",
            "刷好评",
            "虚假案例",
            "fake review",
            "fake testimonial",
        ],
        "note": "虚假评价、虚假案例或伪造社会证明不可使用。",
    },
    {
        "id": "harassment",
        "severity": "block",
        "terms": [
            "人肉",
            "网暴",
            "去骂他",
            "曝光他家",
            "dox",
            "harass",
        ],
        "note": "可能鼓励骚扰、网暴或公开个人信息。",
    },
    {
        "id": "regulated_goods",
        "severity": "review",
        "terms": [
            "代购烟",
            "电子烟",
            "处方药",
            "减肥药",
            "贷款",
            "网赌",
            "枪",
            "cbd",
            "vape",
            "prescription",
        ],
        "note": "涉及监管品类，需要进入更严格审查。",
    },
    {
        "id": "engagement_bait",
        "severity": "review",
        "terms": [
            "不点赞就",
            "评论区打1",
            "疯狂刷屏",
            "点关注马上",
            "comment 1",
            "like or else",
        ],
        "note": "可能是互动诱导、低质量或垃圾信息信号。",
    },
]


REGEX_RULES = [
    {
        "id": "email",
        "severity": "rewrite",
        "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "note": "检测到邮箱，发布前建议隐藏或确认已获授权。",
    },
    {
        "id": "phone_cn_or_us",
        "severity": "rewrite",
        "pattern": r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|1[3-9]\d{9})(?!\d)",
        "note": "检测到手机号，发布前建议打码或删除。",
    },
    {
        "id": "id_card_cn",
        "severity": "block",
        "pattern": r"(?<!\d)\d{17}[\dXx](?!\d)",
        "note": "检测到疑似身份证号，属于高风险隐私信息。",
    },
    {
        "id": "credit_card_like",
        "severity": "block",
        "pattern": r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)",
        "note": "检测到疑似银行卡/信用卡号，属于高风险隐私信息。",
    },
    {
        "id": "url",
        "severity": "review",
        "pattern": r"https?://[^\s]+|www\.[^\s]+",
        "note": "检测到外链，需要确认是否允许以及是否涉及站外交易/引流。",
    },
    {
        "id": "off_platform_contact",
        "severity": "review",
        "pattern": r"(加我|私信|微信|VX|WhatsApp|Telegram|Line)[:：]?\s*[A-Za-z0-9_.-]{4,}",
        "note": "疑似站外联系方式或导流表达。",
    },
]


REWRITE_HINTS = {
    "100%有效": "对我来说有帮助，效果因人而异",
    "保证瘦": "配合饮食和运动管理，可能帮助改善体态",
    "保证赚钱": "分享我的经验，不代表收益承诺",
    "稳赚不赔": "投资有风险，需要自行判断",
    "零风险": "风险较低，但仍需根据自身情况判断",
    "治愈": "帮助缓解或支持改善，不能替代专业建议",
    "根治": "可能帮助改善，具体情况建议咨询专业人士",
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def keyword_findings(text: str) -> list[Finding]:
    lowered = normalize(text)
    findings: list[Finding] = []
    for rule in KEYWORD_RULES:
        for term in rule["terms"]:
            if term.lower() in lowered:
                action = "refuse" if rule["severity"] == "block" else rule["severity"]
                findings.append(
                    Finding(
                        kind="keyword",
                        severity=rule["severity"],
                        rule_id=rule["id"],
                        matched=term,
                        action=action,
                        note=rule["note"],
                    )
                )
    return findings


def regex_findings(text: str) -> list[Finding]:
    findings: list[Finding] = []
    for rule in REGEX_RULES:
        for match in re.finditer(rule["pattern"], text, flags=re.IGNORECASE):
            value = match.group(0)
            findings.append(
                Finding(
                    kind="regex",
                    severity=rule["severity"],
                    rule_id=rule["id"],
                    matched=mask_sensitive(value, rule["id"]),
                    action="refuse" if rule["severity"] == "block" else rule["severity"],
                    note=rule["note"],
                )
            )
    return findings


def mask_sensitive(value: str, rule_id: str) -> str:
    if rule_id in {"email", "phone_cn_or_us", "id_card_cn", "credit_card_like"}:
        if len(value) <= 6:
            return "***"
        return value[:3] + "***" + value[-2:]
    return value[:80]


def line_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def repetition_findings(text: str) -> list[Finding]:
    findings: list[Finding] = []
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 8]

    exact_seen: set[str] = set()
    for line in lines:
        key = normalize(line)
        if key in exact_seen:
            findings.append(
                Finding(
                    kind="duplicate",
                    severity="review",
                    rule_id="exact_repeated_line",
                    matched=line[:80],
                    action="review",
                    note="检测到重复句子，可能是垃圾信息或低原创度信号。",
                )
            )
            break
        exact_seen.add(key)

    for i, left in enumerate(lines):
        for right in lines[i + 1 :]:
            if line_similarity(left, right) >= 0.88:
                findings.append(
                    Finding(
                        kind="duplicate",
                        severity="review",
                        rule_id="near_duplicate_line",
                        matched=left[:80],
                        action="review",
                        note="检测到高度相似句子，建议去重或增加真实上下文。",
                    )
                )
                return findings

    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    if len(tokens) >= 40:
        most_common_count = max(tokens.count(t) for t in set(tokens))
        if most_common_count / len(tokens) >= 0.12:
            findings.append(
                Finding(
                    kind="spam",
                    severity="review",
                    rule_id="keyword_stuffing",
                    matched="repeated token ratio >= 12%",
                    action="review",
                    note="检测到词语堆砌，可能触发垃圾信息或低质量内容风险。",
                )
            )
    return findings


def route(findings: Iterable[Finding]) -> str:
    severities = {finding.severity for finding in findings}
    if "block" in severities:
        return ROUTE_REFUSE
    if "rewrite" in severities:
        return ROUTE_REWRITE
    if "review" in severities:
        return ROUTE_REVIEW
    return ROUTE_PASS


def rewrite_suggestions(text: str) -> list[dict[str, str]]:
    suggestions = []
    for risky, safer in REWRITE_HINTS.items():
        if risky in text:
            suggestions.append({"original": risky, "safer": safer})
    return suggestions


def check(text: str) -> dict:
    findings = keyword_findings(text) + regex_findings(text) + repetition_findings(text)
    result_route = route(findings)
    return {
        "layer": "Layer 1: Machine / Rule Filter",
        "result": result_route,
        "finding_count": len(findings),
        "findings": [asdict(finding) for finding in findings],
        "rewrite_suggestions": rewrite_suggestions(text),
        "next_step": {
            ROUTE_PASS: "Proceed to Layer 2 or full review.",
            ROUTE_REVIEW: "Send to semantic review/classifier before publishing.",
            ROUTE_REWRITE: "Rewrite flagged wording, then re-run Layer 1.",
            ROUTE_REFUSE: "Do not process as requested; redirect to a safer alternative.",
        }[result_route],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Layer 1 creator-script filter.")
    parser.add_argument("file", nargs="?", help="Text file to scan. Reads stdin if omitted.")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        import sys

        text = sys.stdin.read()

    print(json.dumps(check(text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
