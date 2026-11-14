"""Input and output guardrails around the intent classifier (Phase 8).

Every request ends in exactly one of the 77 labels or "unknown" (hand off to a human):

    message -> INPUT:  empty / too long / small talk / prompt injection -> blocked, no model call
                       card numbers, IBANs, emails -> masked before the model sees them
            -> model (Ollama, with token probabilities)
            -> OUTPUT: not one of the 77 labels -> "unknown"
                       model unsure (low confidence) -> "unknown"

Try it:
    python backend/guardrails.py "my card hasn't arrived yet"
    python backend/guardrails.py "I want to open a new bank account"
"""

import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, field

import requests
from pydantic import BaseModel, ValidationError, field_validator

from eval_baseline import OLLAMA_URL, parse_response
from labels import LABEL_SET
from prompts import build_messages

DEFAULT_MODEL = "banking77-qwen"
UNKNOWN = "unknown"

# Real Banking77 messages are at most 433 characters; this leaves generous headroom.
MAX_CHARS = 1000

# Below this, the model is too unsure to trust and we answer "unknown".
# Chosen on validation data only (see eval_guardrails.py), never on the test set:
# from 0.5 to 0.6 out-of-scope catches rise 79% -> 85% and wrong answers fall 6.2% -> 4.9%;
# beyond 0.6 each step sends ~4% more real customers to a human for ~1% more catches.
CONFIDENCE_THRESHOLD = 0.6

# --- Input rules -------------------------------------------------------------------

# Only messages that are *nothing but* small talk: 190 real Banking77 messages start
# with "Hi"/"Hello" and then ask a real question, and those must get through.
_SMALLTALK_PHRASE = (
    r"(hi|hello|hey|hiya|yo|good\s+(morning|afternoon|evening|night)|"
    r"thanks?(\s+(you|so\s+much|a\s+lot))?|thank\s+you(\s+(so\s+much|very\s+much))?|thx|"
    r"bye|goodbye|see\s+you|ok(ay)?|cool|great|nice|test(ing)?|"
    r"how\s+are\s+you(\s+doing)?|who\s+are\s+you|what('?s|\s+is)\s+your\s+name|"
    r"there|again|bot)"
)
SMALLTALK = re.compile(rf"^\W*(?:{_SMALLTALK_PHRASE}[\s\W]*){{1,4}}$", re.IGNORECASE)

PROMPT_INJECTION = re.compile(
    r"\b(ignore|disregard|forget|override)\b.{0,30}\b(instructions?|rules|prompt|list|above)\b"
    r"|\bsystem\s+prompt\b|\byou\s+are\s+now\b|\bpretend\s+(to\s+be|you\s+are)\b"
    r"|\bjailbreak\b|\bdeveloper\s+mode\b|\bnew\s+instructions\b"
    r"|\b(reveal|show|print|repeat)\b.{0,20}\b(instructions|prompt)\b",
    re.IGNORECASE,
)

# Personal data is masked, not blocked: "my card 4111 1111 1111 1111 was declined" is a
# real question, but the number itself must never reach a model or a log file.
# Order matters: an IBAN's digit groups would otherwise be caught as a card number.
PII_PATTERNS = {
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b"),
    "CARD_NUMBER": re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),  # 13-19 digits; years like 2018 don't match
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
}


@dataclass
class InputCheck:
    allowed: bool
    reason: str | None  # why it was blocked
    text: str           # the message with PII masked
    pii_found: list[str] = field(default_factory=list)


def check_input(text: str) -> InputCheck:
    """Validate and clean a customer message before any model sees it."""
    if not text or not text.strip():
        return InputCheck(False, "empty", "")
    if len(text) > MAX_CHARS:
        return InputCheck(False, "too_long", "")

    masked, found = text, []
    for name, pattern in PII_PATTERNS.items():
        masked, n = pattern.subn(f"[{name}]", masked)
        found += [name] * n

    if SMALLTALK.match(masked):
        return InputCheck(False, "small_talk", masked, found)
    if PROMPT_INJECTION.search(masked):
        return InputCheck(False, "prompt_injection", masked, found)
    return InputCheck(True, None, masked, found)


# --- Output rules ------------------------------------------------------------------

class Prediction(BaseModel):
    """The only output shape we accept: an exact Banking77 label."""
    intent: str

    @field_validator("intent")
    @classmethod
    def must_be_a_real_label(cls, v: str) -> str:
        if v not in LABEL_SET:
            raise ValueError(f"not one of the 77 labels: {v!r}")
        return v


# The fine-tuned model sometimes garbles the closing quote and brace:
#   {"intent": "automatic_top_up}      {"intent": "automatic_top_up}"}      {"intent": "automatic_top_up}"
_GARBLED_ENDING = re.compile(r'^\s*\{\s*"intent"\s*:\s*"([A-Za-z_?]+)[\s"}]*$')


def parse_output(raw: str) -> tuple[str | None, str]:
    """Return (label or None, status). Status: ok, repaired, invalid_json, invented_label."""
    intent, json_valid, _ = parse_response(raw)  # the same strict parser as the eval
    status = "ok"
    if not json_valid or intent not in LABEL_SET:
        # Repair punctuation only: the label itself must still be an exact match.
        m = _GARBLED_ENDING.match(raw)
        if m and m.group(1) in LABEL_SET:
            intent, status = m.group(1), "repaired"
        elif not json_valid:
            return None, "invalid_json"
    try:
        return Prediction(intent=intent).intent, status
    except ValidationError:
        return None, "invented_label"


# --- The guarded classifier --------------------------------------------------------

@dataclass
class GuardedResult:
    intent: str                     # one of the 77 labels, or "unknown"
    status: str                     # ok, repaired, low_confidence, invented_label, invalid_json, blocked:<reason>
    confidence: float | None = None
    model_intent: str | None = None  # what the model said, before the confidence check
    pii_found: list[str] = field(default_factory=list)
    raw: str | None = None
    latency_s: float | None = None


def call_model(text: str, model: str = DEFAULT_MODEL) -> tuple[str, float, float]:
    """Classify one message. Returns (raw reply, confidence, latency in seconds).

    Confidence is the probability of the whole reply: the product of every generated
    token's probability. The JSON scaffolding is ~1.0, so it is effectively the
    probability of the label (and of closing it properly).
    """
    payload = {
        "model": model,
        "messages": build_messages(text),
        "stream": False,
        "options": {"temperature": 0},
        "logprobs": True,
    }
    start = time.perf_counter()
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    latency = time.perf_counter() - start
    resp.raise_for_status()
    body = resp.json()
    confidence = math.exp(sum(t["logprob"] for t in body.get("logprobs") or []))
    return body["message"]["content"], confidence, latency


def decide(raw: str, confidence: float, threshold: float = CONFIDENCE_THRESHOLD) -> tuple[str, str, str | None]:
    """Apply the output guardrails. Returns (final intent, status, model's own label)."""
    intent, status = parse_output(raw)
    if intent is None:
        return UNKNOWN, status, None
    if confidence < threshold:
        return UNKNOWN, "low_confidence", intent
    return intent, status, intent


def classify_safe(text: str, model: str = DEFAULT_MODEL) -> GuardedResult:
    """The production entry point: always returns a real label or "unknown"."""
    check = check_input(text)
    if not check.allowed:
        return GuardedResult(UNKNOWN, f"blocked:{check.reason}", pii_found=check.pii_found)
    raw, confidence, latency = call_model(check.text, model)
    intent, status, model_intent = decide(raw, confidence)
    return GuardedResult(intent, status, round(confidence, 4), model_intent,
                         check.pii_found, raw, round(latency, 4))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python backend/guardrails.py "<customer message>"')
    print(json.dumps(asdict(classify_safe(" ".join(sys.argv[1:]))), indent=2))
