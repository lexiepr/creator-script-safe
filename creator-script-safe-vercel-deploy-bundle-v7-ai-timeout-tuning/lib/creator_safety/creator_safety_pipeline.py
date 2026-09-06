#!/usr/bin/env python3
"""
Cost-throttled pipeline for Creator Script Safe.

Order:
1. Idea inputs go straight to AI generation with the fixed policy prefix
2. Script inputs pass Layer 1 machine/rule filtering
3. Script inputs pass Layer 2 local strategy/classification filtering
4. Only risky segments or ambiguous cases call AI
5. Duplicate inputs return from cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from .ai_router import HIGH_REFUSE_MIN, LOW_PASS_MAX, build_ai_plan, run_ai
    from .cache_manager import build_cache_key, cache_status, get_cached_result, set_cached_result
    from .input_classifier import IDEA, classify as classify_input
    from .layer1_machine_filter import check as layer1_check
    from .layer2_strategy_classifier import check as layer2_check
    from .segment_extractor import extract_flagged_segments
except ImportError:
    from ai_router import HIGH_REFUSE_MIN, LOW_PASS_MAX, build_ai_plan, run_ai
    from cache_manager import build_cache_key, cache_status, get_cached_result, set_cached_result
    from input_classifier import IDEA, classify as classify_input
    from layer1_machine_filter import check as layer1_check
    from layer2_strategy_classifier import check as layer2_check
    from segment_extractor import extract_flagged_segments


FAST_LOCAL_RESPONSE = "fast_local_response"
ASK_FOR_CONTEXT = "ask_for_context"
REWRITE_LOCALLY_FIRST = "rewrite_locally_first"
REFUSE_OR_REDIRECT = "refuse_or_redirect"
CALL_SKILL = "call_creator_script_safe"
CALL_AI = "call_ai"


def load_json_arg(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def apply_basic_rewrites(text: str, suggestions: list[dict[str, str]]) -> str:
    rewritten = text
    for suggestion in suggestions:
        original = suggestion.get("original")
        safer = suggestion.get("safer")
        if original and safer:
            rewritten = rewritten.replace(original, safer)
    return rewritten


def build_skill_handoff(
    original_text: str,
    layer1: dict[str, Any],
    layer2: dict[str, Any],
    metadata: dict[str, Any],
    rewritten_text: str | None = None,
    flagged_segments: dict[str, Any] | None = None,
) -> str:
    body = {
        "metadata": metadata,
        "layer1_pre_screen": layer1,
        "layer2_strategy_classification": layer2,
    }
    if flagged_segments:
        body["flagged_segment_extraction"] = flagged_segments

    rewritten_section = ""
    if rewritten_text and rewritten_text != original_text:
        rewritten_section = f"\nLocally rewritten draft:\n{rewritten_text}\n"

    if flagged_segments and flagged_segments.get("segments"):
        segment_lines = []
        for index, item in enumerate(flagged_segments["segments"], start=1):
            reasons = ", ".join(
                finding.get("rule_id", "unknown") for finding in item.get("findings", [])
            )
            segment_lines.append(
                f"{index}. Reasons: {reasons}\n"
                f"   Segment:\n"
                f"   {item['segment']}"
            )
        review_target = (
            "Flagged segments for token-efficient review:\n"
            + "\n\n".join(segment_lines)
            + "\n\nOnly review and rewrite the flagged segments unless full-script context is necessary.\n"
        )
    else:
        review_target = "Original script:\n" f"{original_text}\n"

    return (
        "Please run Creator Script Safe full review.\n\n"
        f"{review_target}"
        f"{rewritten_section}\n"
        "Local pre-screen context:\n"
        f"{json.dumps(body, ensure_ascii=False, indent=2)}\n\n"
        "Use Layer 1 and Layer 2 as routing context. Include them in the Pre-screen section. "
        "Do not treat any local result as platform approval, compliance, monetization safety, "
        "reach safety, or account safety."
    )


def build_idea_handoff(
    idea_text: str,
    input_classification: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    body = {
        "metadata": metadata,
        "input_classification": input_classification,
        "requested_task": "Generate a safer creator script from a rough idea.",
    }
    return (
        "Please use Creator Script Safe to generate a safer creator script from this idea.\n\n"
        "Idea:\n"
        f"{idea_text}\n\n"
        "Local routing context:\n"
        f"{json.dumps(body, ensure_ascii=False, indent=2)}\n\n"
        "Use the standard output sections exactly:\n"
        "Overall risk:\n"
        "Category checks:\n"
        "Risk findings:\n"
        "Safer rewrites:\n"
        "Final safer script:\n"
        "Live brief:\n"
        "Notes:\n\n"
        "Use cautious, evidence-based wording. Do not guarantee platform approval, reach, "
        "monetization, account safety, or warning removal."
    )


def fixed_local_output(
    *,
    overall_risk: str,
    category_status: str,
    category_note: str,
    risk_issue: str = "",
    risk_fix: str = "",
    final_safer_script: str = "",
    notes: list[str] | None = None,
) -> dict[str, Any]:
    findings = []
    if risk_issue:
        findings.append(
            {
                "severity": category_status.lower(),
                "issue": risk_issue,
                "evidence": "Local pre-screen",
                "fix": risk_fix,
            }
        )
    return {
        "overall_risk": overall_risk,
        "category_checks": [
            {
                "category": "Local pre-screen",
                "status": category_status,
                "note": category_note,
            }
        ],
        "risk_findings": findings,
        "safer_rewrites": [],
        "final_safer_script": final_safer_script,
        "live_brief": "",
        "notes": notes or ["This local result does not guarantee platform approval."],
    }


def attach_ai_result(result: dict[str, Any], ai_plan: dict[str, Any]) -> dict[str, Any]:
    ai_result = run_ai(ai_plan)
    result["ai"] = ai_result
    if ai_result.get("output"):
        result["structured_output"] = ai_result["output"]
    return result


def decide(
    text: str,
    metadata: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
    force_full_review: bool = False,
    auto_rewrite: bool = True,
) -> dict[str, Any]:
    metadata = metadata or {}
    cache_key = build_cache_key(
        text,
        metadata=metadata,
        force_full_review=force_full_review,
        auto_rewrite=auto_rewrite,
    )
    cached, cache_backend = get_cached_result(cache_key)
    if cached:
        cached["cache"] = cache_status(True, cache_key, cache_backend)
        return cached

    input_classification = classify_input(text)

    if input_classification["input_type"] == IDEA:
        ai_plan = build_ai_plan(
            original_text=text,
            input_classification=input_classification,
            metadata=metadata,
            purpose="idea_generation",
        )
        result = {
            "decision": CALL_AI,
            "input_classification": input_classification,
            "skill_handoff_prompt": build_idea_handoff(text, input_classification, metadata),
            "response": (
                "Input looks like an idea, not a finished script. "
                "Skip Layer 1/2 and generate a safer script with Creator Script Safe."
            ),
        }
        attach_ai_result(result, ai_plan)
        cache_backend = set_cached_result(cache_key, result)
        result["cache"] = cache_status(False, cache_key, cache_backend)
        return result

    layer1 = layer1_check(text)
    flagged_segments = extract_flagged_segments(text, layer1)

    if layer1["result"] == "Refuse/redirect":
        layer2 = layer2_check(text, layer1=layer1, metadata=metadata, weights=weights)
        result = {
            "decision": REFUSE_OR_REDIRECT,
            "input_classification": input_classification,
            "layer1": layer1,
            "flagged_segments": flagged_segments,
            "layer2": layer2,
            "structured_output": fixed_local_output(
                overall_risk="High",
                category_status="Blocked",
                category_note="Layer 1 found refusal-level risk.",
                risk_issue="The request is unsafe as written.",
                risk_fix="Redirect to a transparent, privacy-preserving, evidence-based, or educational alternative.",
            ),
            "response": (
                "Layer 1: Refuse/redirect. Layer 2: High. "
                "Do not generate the unsafe request as written; offer a transparent, "
                "privacy-preserving, evidence-based, or educational alternative."
            ),
        }
        cache_backend = set_cached_result(cache_key, result)
        result["cache"] = cache_status(False, cache_key, cache_backend)
        return result

    working_text = text
    local_rewrite_applied = False
    if layer1["result"] == "Immediate rewrite" and auto_rewrite:
        rewritten = apply_basic_rewrites(text, layer1.get("rewrite_suggestions", []))
        if rewritten != text:
            working_text = rewritten
            local_rewrite_applied = True
            layer1_after_rewrite = layer1_check(working_text)
        else:
            layer1_after_rewrite = layer1
    else:
        layer1_after_rewrite = layer1

    layer2 = layer2_check(working_text, layer1=layer1_after_rewrite, metadata=metadata, weights=weights)
    routing = layer2["routing"]
    risk_probability = float(layer2.get("risk_probability", 0.0) or 0.0)

    if routing == "Needs more context":
        result = {
            "decision": ASK_FOR_CONTEXT,
            "input_classification": input_classification,
            "layer1": layer1,
            "layer1_after_rewrite": layer1_after_rewrite if local_rewrite_applied else None,
            "flagged_segments": flagged_segments,
            "layer2": layer2,
            "missing_context": layer2["missing_context"],
            "response": "Ask only for the missing context fields before running the full Skill.",
        }
        cache_backend = set_cached_result(cache_key, result)
        result["cache"] = cache_status(False, cache_key, cache_backend)
        return result

    if routing == "Refuse/redirect":
        result = {
            "decision": REFUSE_OR_REDIRECT,
            "input_classification": input_classification,
            "layer1": layer1,
            "layer1_after_rewrite": layer1_after_rewrite if local_rewrite_applied else None,
            "flagged_segments": flagged_segments,
            "layer2": layer2,
            "structured_output": fixed_local_output(
                overall_risk="High",
                category_status="Blocked",
                category_note="Layer 2 routed this content to refusal/redirect.",
                risk_issue="The script is too risky to generate or improve as written.",
                risk_fix="Ask for a safer purpose or redirect to educational, neutral wording.",
            ),
            "response": "Do not generate the unsafe request as written; redirect to a safer alternative.",
        }
        cache_backend = set_cached_result(cache_key, result)
        result["cache"] = cache_status(False, cache_key, cache_backend)
        return result

    if risk_probability >= HIGH_REFUSE_MIN and not force_full_review:
        result = {
            "decision": REFUSE_OR_REDIRECT,
            "input_classification": input_classification,
            "layer1": layer1,
            "layer1_after_rewrite": layer1_after_rewrite if local_rewrite_applied else None,
            "flagged_segments": flagged_segments,
            "layer2": layer2,
            "structured_output": fixed_local_output(
                overall_risk="High",
                category_status="Blocked",
                category_note=f"Layer 2 score {risk_probability} is above the direct-block threshold {HIGH_REFUSE_MIN}.",
                risk_issue="The content is high risk based on local strategy signals.",
                risk_fix="Do not call the LLM for this version; redirect or ask for a safer revision.",
            ),
            "response": "Layer 2 score is high. Block locally and skip the LLM.",
        }
        cache_backend = set_cached_result(cache_key, result)
        result["cache"] = cache_status(False, cache_key, cache_backend)
        return result

    should_ai_review = (
        force_full_review
        or layer1_after_rewrite["result"] in {"Immediate rewrite", "Review needed"}
        or LOW_PASS_MAX < risk_probability < HIGH_REFUSE_MIN
    )

    if risk_probability <= LOW_PASS_MAX and not should_ai_review:
        result = {
            "decision": FAST_LOCAL_RESPONSE,
            "input_classification": input_classification,
            "layer1": layer1,
            "flagged_segments": flagged_segments,
            "layer2": layer2,
            "structured_output": fixed_local_output(
                overall_risk="Low",
                category_status="Pass",
                category_note=f"Layer 2 score {risk_probability} is at or below the cheap-pass threshold {LOW_PASS_MAX}.",
            ),
            "response": (
                "Layer 1: Pass. Layer 2: Low. No obvious local pre-screen trigger found. "
                "This does not guarantee platform approval."
            ),
        }
        cache_backend = set_cached_result(cache_key, result)
        result["cache"] = cache_status(False, cache_key, cache_backend)
        return result

    ai_plan = build_ai_plan(
        original_text=text,
        input_classification=input_classification,
        metadata=metadata,
        layer1=layer1_after_rewrite,
        layer2=layer2,
        flagged_segments=flagged_segments,
        purpose="script_review_or_rewrite",
        force_full_review=force_full_review,
    )
    result = {
        "decision": CALL_AI,
        "input_classification": input_classification,
        "layer1": layer1,
        "layer1_after_rewrite": layer1_after_rewrite if local_rewrite_applied else None,
        "flagged_segments": flagged_segments,
        "layer2": layer2,
        "rewritten_text": working_text if local_rewrite_applied else None,
        "skill_handoff_prompt": build_skill_handoff(
            text,
            layer1_after_rewrite,
            layer2,
            metadata,
            working_text,
            flagged_segments=flagged_segments,
        ),
        "response": "Route to AI with the fixed policy prefix and structured output schema.",
    }
    attach_ai_result(result, ai_plan)
    cache_backend = set_cached_result(cache_key, result)
    result["cache"] = cache_status(False, cache_key, cache_backend)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cost-throttled Creator Script Safe pipeline.")
    parser.add_argument("file", nargs="?", help="Text file to scan. Reads stdin if omitted.")
    parser.add_argument("--metadata", help="Metadata JSON string or path to JSON file.")
    parser.add_argument("--weights", help="Optional Layer 2 weights JSON string or path to JSON file.")
    parser.add_argument("--force-full-review", action="store_true", help="Always generate Skill handoff unless refused or context is missing.")
    parser.add_argument("--no-auto-rewrite", action="store_true", help="Do not apply simple local rewrite suggestions before Layer 2.")
    args = parser.parse_args()

    text = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    metadata = load_json_arg(args.metadata)
    weights = load_json_arg(args.weights)

    result = decide(
        text,
        metadata=metadata,
        weights=weights,
        force_full_review=args.force_full_review,
        auto_rewrite=not args.no_auto_rewrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
