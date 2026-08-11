#!/usr/bin/env python3
"""
Cost-throttled pipeline for Creator Script Safe.

Order:
1. Layer 1 machine/rule filter
2. Layer 2 local strategy/classification filter
3. Generate a handoff prompt for creator-script-safe only when needed

No LLM calls are made by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from .layer1_machine_filter import check as layer1_check
    from .layer2_strategy_classifier import check as layer2_check
except ImportError:
    from layer1_machine_filter import check as layer1_check
    from layer2_strategy_classifier import check as layer2_check


FAST_LOCAL_RESPONSE = "fast_local_response"
ASK_FOR_CONTEXT = "ask_for_context"
REWRITE_LOCALLY_FIRST = "rewrite_locally_first"
REFUSE_OR_REDIRECT = "refuse_or_redirect"
CALL_SKILL = "call_creator_script_safe"


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
) -> str:
    body = {
        "metadata": metadata,
        "layer1_pre_screen": layer1,
        "layer2_strategy_classification": layer2,
    }
    rewritten_section = ""
    if rewritten_text and rewritten_text != original_text:
        rewritten_section = f"\nLocally rewritten draft:\n{rewritten_text}\n"

    return (
        "Please run Creator Script Safe full review.\n\n"
        "Original script:\n"
        f"{original_text}\n"
        f"{rewritten_section}\n"
        "Local pre-screen context:\n"
        f"{json.dumps(body, ensure_ascii=False, indent=2)}\n\n"
        "Use Layer 1 and Layer 2 as routing context. Include them in the Pre-screen section. "
        "Do not treat any local result as platform approval, compliance, monetization safety, "
        "reach safety, or account safety."
    )


def decide(
    text: str,
    metadata: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
    force_full_review: bool = False,
    auto_rewrite: bool = True,
) -> dict[str, Any]:
    metadata = metadata or {}
    layer1 = layer1_check(text)

    if layer1["result"] == "Refuse/redirect":
        layer2 = layer2_check(text, layer1=layer1, metadata=metadata, weights=weights)
        return {
            "decision": REFUSE_OR_REDIRECT,
            "layer1": layer1,
            "layer2": layer2,
            "response": (
                "Layer 1: Refuse/redirect. Layer 2: High. "
                "Do not generate the unsafe request as written; offer a transparent, "
                "privacy-preserving, evidence-based, or educational alternative."
            ),
        }

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

    if routing == "Needs more context":
        return {
            "decision": ASK_FOR_CONTEXT,
            "layer1": layer1,
            "layer1_after_rewrite": layer1_after_rewrite if local_rewrite_applied else None,
            "layer2": layer2,
            "missing_context": layer2["missing_context"],
            "response": "Ask only for the missing context fields before running the full Skill.",
        }

    if routing == "Refuse/redirect":
        return {
            "decision": REFUSE_OR_REDIRECT,
            "layer1": layer1,
            "layer1_after_rewrite": layer1_after_rewrite if local_rewrite_applied else None,
            "layer2": layer2,
            "response": "Do not generate the unsafe request as written; redirect to a safer alternative.",
        }

    if routing == "Cheap pass" and not force_full_review:
        return {
            "decision": FAST_LOCAL_RESPONSE,
            "layer1": layer1,
            "layer2": layer2,
            "response": (
                "Layer 1: Pass. Layer 2: Low. No obvious local pre-screen trigger found. "
                "This does not guarantee platform approval."
            ),
        }

    if routing == "Rewrite first" and not force_full_review:
        return {
            "decision": REWRITE_LOCALLY_FIRST,
            "layer1": layer1,
            "layer1_after_rewrite": layer1_after_rewrite if local_rewrite_applied else None,
            "layer2": layer2,
            "rewritten_text": working_text,
            "response": "Rewrite flagged wording locally, re-run Layer 1 and Layer 2, then decide whether Skill review is needed.",
        }

    return {
        "decision": CALL_SKILL,
        "layer1": layer1,
        "layer1_after_rewrite": layer1_after_rewrite if local_rewrite_applied else None,
        "layer2": layer2,
        "rewritten_text": working_text if local_rewrite_applied else None,
        "skill_handoff_prompt": build_skill_handoff(text, layer1_after_rewrite, layer2, metadata, working_text),
        "response": "Send the generated handoff prompt to creator-script-safe.",
    }


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
