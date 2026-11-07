"""The 77 Banking77 intent names, loaded once at import time.

Usage:
    from labels import LABELS, LABEL_SET
    LABELS[example["label"]]  # -> e.g. "card_arrival"
"""

from data import load_label_names

LABELS: list[str] = load_label_names()  # index == integer label id
LABEL_SET: frozenset[str] = frozenset(LABELS)  # fast "is this a valid label?" checks

assert len(LABELS) == 77, f"expected 77 labels, got {len(LABELS)}"
