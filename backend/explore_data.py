"""Phase 1: explore Banking77 before touching any model.

Run:  python backend/explore_data.py

Each section answers a question that drives a later decision:
  1. Split sizes      -> how much data do we have to train on and test with?
  2. Class balance    -> is "accuracy" a fair metric? Do we need to rebalance?
  3. Ordering         -> must we shuffle before splitting/training?
  4. Text length      -> how long are prompts? (sets max_seq_length for fine-tuning)
  5. Data hygiene     -> duplicates, and test examples that leak into train
  6. Samples          -> what do the intents actually look like? which are confusable?
"""

import random
import statistics
from collections import Counter

from data import load_split
from labels import LABELS


def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def length_stats(texts: list[str]) -> dict:
    """Word-count statistics. Words are a rough proxy for tokens (~1.3 tokens/word in English)."""
    words = sorted(len(t.split()) for t in texts)
    return {
        "min": words[0],
        "mean": round(statistics.mean(words), 1),
        "median": statistics.median(words),
        "p95": words[int(0.95 * (len(words) - 1))],  # 95% of messages are this short or shorter
        "max": words[-1],
    }


def main() -> None:
    train, test = load_split("train"), load_split("test")

    # 1. Split sizes ---------------------------------------------------------
    header("1. SPLIT SIZES")
    total = len(train) + len(test)
    for name, rows in (("train", train), ("test", test)):
        print(f"  {name:5s}: {len(rows):6,} examples ({len(rows) / total:.0%})")
    print(f"  per intent, on average: {len(train) / len(LABELS):.0f} train, "
          f"{len(test) / len(LABELS):.0f} test")

    # 2. Class balance -------------------------------------------------------
    header("2. CLASS BALANCE (examples per intent)")
    for name, rows in (("train", train), ("test", test)):
        counts = Counter(r["label"] for r in rows)
        lo, hi = min(counts.values()), max(counts.values())
        print(f"  {name:5s}: {len(counts)} intents present | min {lo}, max {hi}, "
              f"imbalance ratio {hi / lo:.1f}x")
    train_counts = Counter(r["label"] for r in train)
    print("\n  Fewest training examples:")
    for label, n in sorted(train_counts.items(), key=lambda kv: kv[1])[:5]:
        print(f"    {n:4d}  {LABELS[label]}")
    print("  Most training examples:")
    for label, n in train_counts.most_common(5):
        print(f"    {n:4d}  {LABELS[label]}")
    # What would a "dumb" model score? Always predicting the most common test intent:
    test_counts = Counter(r["label"] for r in test)
    majority = test_counts.most_common(1)[0][1] / len(test)
    print(f"\n  Majority-class baseline on test: {majority:.1%}  (any real model must beat this)")
    print(f"  Random-guess baseline:           {1 / len(LABELS):.1%}")

    # 3. Ordering ------------------------------------------------------------
    header("3. ORDERING (is the train file grouped by intent?)")
    # Count how often the label changes between consecutive rows. Shuffled data
    # changes label almost every row; grouped data changes only ~77 times.
    changes = sum(a["label"] != b["label"] for a, b in zip(train, train[1:]))
    print(f"  label changes between consecutive train rows: {changes:,} of {len(train) - 1:,}")
    print("  -> GROUPED: must shuffle before splitting/training" if changes < len(train) * 0.5
          else "  -> looks shuffled")

    # 4. Text length ---------------------------------------------------------
    header("4. TEXT LENGTH (in words)")
    for name, rows in (("train", train), ("test", test)):
        print(f"  {name:5s}: {length_stats([r['text'] for r in rows])}")
    by_len = sorted(train, key=lambda r: len(r["text"].split()))
    print("\n  Shortest train examples:")
    for r in by_len[:3]:
        print(f"    {r['text']!r:45s} -> {LABELS[r['label']]}")
    print("  Longest train example:")
    r = by_len[-1]
    print(f"    {r['text'][:120]!r}... -> {LABELS[r['label']]}")

    # 5. Data hygiene --------------------------------------------------------
    header("5. DATA HYGIENE (duplicates and leakage)")
    norm = lambda t: " ".join(t.lower().split())  # ignore case and extra spaces
    train_texts = [norm(r["text"]) for r in train]
    test_texts = [norm(r["text"]) for r in test]
    print(f"  duplicate texts within train: {len(train_texts) - len(set(train_texts))}")
    print(f"  duplicate texts within test:  {len(test_texts) - len(set(test_texts))}")
    # Leakage: a test example also seen in training inflates test accuracy
    # (the model can memorize instead of generalize).
    leaked = set(train_texts) & set(test_texts)
    print(f"  test texts also in train (leakage): {len(leaked)}")
    # Same text, different labels = ambiguous / noisy labels.
    labels_by_text: dict[str, set[int]] = {}
    for t, r in zip(train_texts, train):
        labels_by_text.setdefault(t, set()).add(r["label"])
    conflicts = {t: ls for t, ls in labels_by_text.items() if len(ls) > 1}
    print(f"  train texts with conflicting labels: {len(conflicts)}")
    for t, ls in list(conflicts.items())[:3]:
        print(f"    {t!r} -> {[LABELS[l] for l in ls]}")

    # 6. Samples -------------------------------------------------------------
    header("6. SAMPLES (confusable intent groups)")
    random.seed(0)  # fixed seed so you see the same samples every run
    groups = [
        ["card_arrival", "card_delivery_estimate", "lost_or_stolen_card"],
        ["top_up_failed", "top_up_reverted", "pending_top_up"],
        ["transfer_not_received_by_recipient", "pending_transfer", "failed_transfer"],
    ]
    for group in groups:
        print()
        for name in group:
            label_id = LABELS.index(name)
            texts = [r["text"] for r in train if r["label"] == label_id]
            print(f"  [{name}]")
            for t in random.sample(texts, 2):
                print(f"     - {t}")


if __name__ == "__main__":
    main()
