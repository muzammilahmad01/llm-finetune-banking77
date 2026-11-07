"""The single source of truth for the classification prompt.

Both evaluation (eval_baseline.py) and training data (prepare_sft.py) import
from here, so the prompt the model is trained on is *exactly* the prompt it is
evaluated with. A train/eval prompt mismatch is a classic silent bug.
"""

import json

from labels import LABELS

# The label list goes in the *system* message, identical for every request.
# Because each prompt starts with the same long prefix, Ollama can reuse its
# KV cache for that prefix, so only the short user query is processed fresh each time.
SYSTEM_PROMPT = (
    "You are an intent classifier for a banking customer-support assistant.\n"
    "Classify the user's message into exactly one of these intents:\n\n"
    + "\n".join(LABELS)
    + "\n\nRespond with JSON only, no other text, in this exact format:\n"
    '{"intent": "<one intent from the list above>"}'
)


def build_messages(text: str) -> list[dict]:
    """The prompt part of a conversation: system instructions + the customer's message."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def format_answer(intent: str) -> str:
    """The ideal assistant reply, e.g. '{"intent": "card_arrival"}'."""
    return json.dumps({"intent": intent})
