#!/usr/bin/env python3
"""
AI routing and execution for Creator Script Safe.

This module keeps model decisions, prompt shape, and structured output
separate from Layer 1 / Layer 2 so the low-cost local path stays simple.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


LOW_PASS_MAX = float(os.getenv("CSS_AI_LOW_PASS_MAX", "0.35"))
HIGH_REFUSE_MIN = float(os.getenv("CSS_AI_HIGH_REFUSE_MIN", "0.70"))
FAST_MODEL = os.getenv("CSS_FAST_MODEL", "gpt-5-mini")
STRONG_MODEL = os.getenv("CSS_STRONG_MODEL", "gpt-5.1")
ALLOW_STRONG_MODEL = os.getenv("CSS_ALLOW_STRONG_MODEL", "false").lower() == "true"


STRUCTURED_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "overall_risk",
        "category_checks",
        "risk_findings",
        "safer_rewrites",
        "final_safer_script",
        "live_brief",
        "notes",
        "confidence",
        "needs_strong_model",
    ],
    "properties": {
        "overall_risk": {"type": "string"},
        "category_checks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["category", "status", "note"],
                "properties": {
                    "category": {"type": "string"},
                    "status": {"type": "string"},
                    "note": {"type": "string"},
                },
            },
        },
        "risk_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["severity", "issue", "evidence", "fix"],
                "properties": {
                    "severity": {"type": "string"},
                    "issue": {"type": "string"},
                    "evidence": {"type": "string"},
                    "fix": {"type": "string"},
                },
            },
        },
        "safer_rewrites": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["original", "rewrite", "reason"],
                "properties": {
                    "original": {"type": "string"},
                    "rewrite": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
        "final_safer_script": {"type": "string"},
        "live_brief": {"type": "string"},
        "notes": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "needs_strong_model": {"type": "boolean"},
    },
}


POLICY_PREFIX = """Creator Script Safe policy prefix.

Goal:
- Help creators turn ideas or draft scripts into safer platform-ready wording.
- Reduce risky claims, privacy exposure, spam signals, misleading urgency, undisclosed sponsorship, unsafe regulated-good promotion, and evasion language.
- Never promise platform approval, reach, monetization, account safety, warning removal, medical outcomes, financial outcomes, or legal outcomes.

Output exactly these sections as structured JSON:
- Overall risk
- Category checks
- Risk findings
- Safer rewrites
- Final safer script
- Live brief
- Notes

Style:
- Be practical, creator-friendly, and concise.
- Preserve the creator's intent when it is safe.
- Rewrite unsafe language into cautious, evidence-based, transparent wording.
- If the request is unsafe as written, redirect to a safer educational, transparent, privacy-preserving, or non-commercial alternative.
"""


def ai_enabled() -> bool:
    if os.getenv("CSS_AI_ENABLED", "").lower() in {"0", "false", "no"}:
        return False
    return bool(os.getenv("OPENAI_API_KEY"))


def target_from_segments(original_text: str, flagged_segments: dict[str, Any]) -> tuple[str, str, float]:
    segments = flagged_segments.get("segments") or []
    if not segments:
        return "full_script", original_text, 0.0

    lines = []
    for index, item in enumerate(segments, start=1):
        reasons = ", ".join(
            finding.get("rule_id", "unknown") for finding in item.get("findings", [])
        )
        lines.append(f"[Segment {index}; reasons: {reasons}]\n{item.get('segment', '')}")

    target_text = "\n\n".join(lines)
    savings = flagged_segments.get("estimated_char_savings_ratio", 0.0)
    return "flagged_segments", target_text, float(savings)


def build_ai_plan(
    *,
    original_text: str,
    input_classification: dict[str, Any],
    metadata: dict[str, Any],
    layer1: dict[str, Any] | None = None,
    layer2: dict[str, Any] | None = None,
    flagged_segments: dict[str, Any] | None = None,
    purpose: str,
    force_full_review: bool = False,
) -> dict[str, Any]:
    layer1 = layer1 or {}
    layer2 = layer2 or {}
    flagged_segments = flagged_segments or {}
    risk_probability = float(layer2.get("risk_probability", 0.0) or 0.0)

    if purpose == "idea_generation":
        review_target = "idea"
        target_text = original_text
        token_savings = 0.0
        should_call = True
        reason = "Input is an idea, so use AI to generate the safer script."
    elif force_full_review:
        review_target = "full_script"
        target_text = original_text
        token_savings = 0.0
        should_call = True
        reason = "Full review was forced."
    else:
        review_target, target_text, token_savings = target_from_segments(original_text, flagged_segments)
        should_call = True
        reason = "Local filters found risky or ambiguous wording that needs AI review."

    return {
        "should_call": should_call,
        "reason": reason,
        "model_tier": "fast",
        "model": FAST_MODEL,
        "strong_model": STRONG_MODEL,
        "allow_strong_model": ALLOW_STRONG_MODEL,
        "review_target": review_target,
        "target_chars": len(target_text),
        "original_chars": len(original_text),
        "estimated_token_savings_ratio": round(token_savings, 3),
        "thresholds": {
            "low_pass_max": LOW_PASS_MAX,
            "high_refuse_min": HIGH_REFUSE_MIN,
        },
        "risk_probability": risk_probability,
        "ai_enabled": ai_enabled(),
        "purpose": purpose,
        "target_text": target_text,
        "routing_context": {
            "metadata": metadata,
            "input_classification": input_classification,
            "layer1": layer1,
            "layer2": layer2,
            "flagged_segments": flagged_segments,
        },
    }


def build_messages(ai_plan: dict[str, Any]) -> list[dict[str, Any]]:
    context = {
        "purpose": ai_plan["purpose"],
        "review_target": ai_plan["review_target"],
        "risk_probability": ai_plan["risk_probability"],
        "routing_context": ai_plan["routing_context"],
    }
    return [
        {
            "role": "developer",
            "content": [{"type": "input_text", "text": POLICY_PREFIX}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Routing context:\n"
                        f"{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}\n\n"
                        "Content to process:\n"
                        f"{ai_plan['target_text']}"
                    ),
                }
            ],
        },
    ]


def fallback_output(ai_plan: dict[str, Any], note: str) -> dict[str, Any]:
    if ai_plan["purpose"] == "idea_generation":
        final_script = ""
        finding = "AI generation is required to turn this idea into a finished script."
    elif ai_plan["review_target"] == "flagged_segments":
        final_script = ""
        finding = "Only flagged segments should be sent to AI to save tokens."
    else:
        final_script = ""
        finding = "Full script AI review is required."

    return {
        "overall_risk": "AI review pending",
        "category_checks": [
            {
                "category": "Local pre-screen",
                "status": "Routed to AI",
                "note": ai_plan["reason"],
            }
        ],
        "risk_findings": [
            {
                "severity": "review",
                "issue": finding,
                "evidence": ai_plan["review_target"],
                "fix": "Enable OPENAI_API_KEY in Vercel to complete generation or rewrite.",
            }
        ],
        "safer_rewrites": [],
        "final_safer_script": final_script,
        "live_brief": "",
        "notes": [note],
        "confidence": 0.0,
        "needs_strong_model": False,
    }


def extract_output_text(response: dict[str, Any]) -> str:
    if response.get("output_text"):
        return str(response["output_text"])
    chunks: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(str(content["text"]))
    return "\n".join(chunks)


def call_openai(ai_plan: dict[str, Any], model: str) -> dict[str, Any]:
    request_body = {
        "model": model,
        "input": build_messages(ai_plan),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "creator_script_safe_result",
                "strict": True,
                "schema": STRUCTURED_OUTPUT_SCHEMA,
            }
        },
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        raw = json.loads(response.read().decode("utf-8"))
    output_text = extract_output_text(raw)
    return json.loads(output_text)


def run_ai(ai_plan: dict[str, Any]) -> dict[str, Any]:
    public_plan = {key: value for key, value in ai_plan.items() if key not in {"target_text"}}
    if not ai_plan.get("should_call"):
        return {"status": "skipped", "plan": public_plan, "output": None}

    if not ai_enabled():
        return {
            "status": "planned_not_called",
            "plan": public_plan,
            "output": fallback_output(ai_plan, "AI is not enabled or OPENAI_API_KEY is missing."),
        }

    try:
        output = call_openai(ai_plan, ai_plan["model"])
        model_used = ai_plan["model"]
        if output.get("needs_strong_model") and ALLOW_STRONG_MODEL:
            output = call_openai(ai_plan, STRONG_MODEL)
            model_used = STRONG_MODEL
            public_plan["model_tier"] = "strong"
            public_plan["model"] = STRONG_MODEL
        return {
            "status": "completed",
            "model_used": model_used,
            "plan": public_plan,
            "output": output,
        }
    except (urllib.error.URLError, TimeoutError, RuntimeError, KeyError, json.JSONDecodeError, OSError) as exc:
        return {
            "status": "error",
            "plan": public_plan,
            "error": exc.__class__.__name__,
            "output": fallback_output(ai_plan, "AI call failed; local routing result is still available."),
        }
