"""
guardrails.py

A lightweight, deterministic layer that runs before the main LLM call to
catch obvious bad inputs — prompt injections, off-topic questions, etc.

Keeping this rule-based rather than using a second LLM call because:
  (a) it's deterministic and easy to test
  (b) it's fast — no extra API round-trip
  (c) prompt injection attempts are usually recognisable patterns

When adding new patterns, prefer narrow, specific regexes over broad ones.
A false positive that blocks a legitimate hiring question is just as bad as
a missed injection.
"""

import re

# ---------------------------------------------------------------------------
# Prompt injection patterns
# These are phrases that almost always signal someone trying to override the
# system prompt or extract information about it.
# ---------------------------------------------------------------------------
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"ignore\s+your\s+(system\s+)?prompt",
    r"ignore\s+your\s+instructions?",
    r"disregard\s+(all\s+)?instructions?",
    r"forget\s+(everything|all|previous)",
    r"you\s+are\s+now\s+a?\s*different",
    r"act\s+as\s+(if\s+you\s+are\s+)?a?\s*(different|new|unrestricted)",
    r"pretend\s+(to\s+be|you(\s+are)?)",
    r"show\s+me\s+your\s+(system\s+)?prompt",
    r"reveal\s+your\s+(system\s+)?prompt",
    r"what\s+(are\s+)?your\s+instructions?",
    r"bypass\s+(your\s+)?(restrictions?|filters?|guardrails?)",
    r"\bjailbreak\b",
    r"\bDAN\s+mode\b",
    r"\bdeveloper\s+mode\b",
    r"override\s+(your\s+)?(instructions?|system)",
    r"<\/?(system|prompt|instruction)>",
]

# ---------------------------------------------------------------------------
# Off-topic patterns
# Topics that have nothing to do with SHL assessments or hiring.
# NOTE: Keep these specific — avoid patterns that could catch legitimate
# hiring-related queries (e.g., "translate" was removed because non-English
# speakers might describe roles in mixed language).
# ---------------------------------------------------------------------------
OFF_TOPIC_PATTERNS = [
    r"\b(recipe|cook(ing)?|restaurant)\b",
    r"\b(movie|film|netflix|tv\s+show)\b",
    r"\b(weather|forecast)\b",
    r"\b(stock\s+market|cryptocurrency|bitcoin)\b",
    r"\b(dating|romance)\b",
    r"\b(sports?\s+score|football\s+result|soccer\s+result)\b",
    r"\b(political\s+party|election\s+result|vote\s+for)\b",
    r"\b(medical\s+diagnosis|write\s+me\s+a\s+prescription)\b",
    r"\b(legal\s+advice|file\s+a\s+lawsuit)\b",
    r"\b(write\s+(me\s+)?an?\s+essay|do\s+my\s+homework)\b",
    r"\b(programming\s+tutorial|learn\s+python\s+from\s+scratch)\b",
]

_injection_re = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)
_offtopic_re = re.compile("|".join(OFF_TOPIC_PATTERNS), re.IGNORECASE)


def check(messages: list[dict]) -> tuple[bool, str]:
    """
    Checks the most recent user message for injection or off-topic content.

    Returns (flagged, reason) where:
      - flagged is True if the message should be refused
      - reason is "injection" or "off_topic" (empty string if not flagged)

    Only looks at the latest user message — we don't re-check history on
    every turn, that would be too slow and would double-flag things.
    """
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return False, ""

    text = user_msgs[-1].get("content", "")

    if _injection_re.search(text):
        return True, "injection"

    if _offtopic_re.search(text):
        return True, "off_topic"

    return False, ""


def make_refusal_response(reason: str) -> dict:
    """
    Returns a schema-valid refusal response.

    The response schema has to be valid even when we're refusing a request,
    so this always returns the full reply / recommendations / end_of_conversation
    structure.
    """
    if reason == "injection":
        reply = (
            "I can't respond to that — it looks like an attempt to change my instructions. "
            "I'm here to help you find the right SHL assessments. What role are you hiring for?"
        )
    else:
        reply = (
            "I'm focused on SHL assessment selection and can't help with that topic. "
            "If you're looking for the right assessment for a role or team, I'm happy to help."
        )

    return {
        "reply": reply,
        "recommendations": [],
        "end_of_conversation": False,
    }
