"""Phase 1: carve a validation split out of the Banking77 train split.

Run:  python backend/make_splits.py

Produces (from data/train.jsonl):
    data/train_split.jsonl  ~90%: what the model trains on
    data/val_split.jsonl    ~10%: checked during training to spot overfitting

The test split is never touched here; it stays sealed for final evaluation.
"""

import json
import random
from collections import Counter, defaultdict

from data import DATA_DIR, load_split
from labels import LABELS

SEED = 42          # fixed seed -> the same split on every machine, every run
VAL_FRACTION = 0.10


def main() -> None:
    train = load_split("train")

    # Drop exact duplicate texts (case/whitespace-insensitive), so the same
    # message can't end up in both train_split and val_split.
    seen, unique = set(), []
    for row in train:
        key = " ".join(row["text"].lower().split())
        if key not in seen:
            seen.add(key)
            unique.append(row)
    print(f"train: {len(train):,} rows, {len(train) - len(unique)} duplicates dropped "
          f"-> {len(unique):,}")

    # Stratified split: shuffle and split *each intent separately*, so every
    # intent keeps ~10% in validation, even the rare ones.
    rng = random.Random(SEED)
    by_label = defaultdict(list)
    for row in unique:
        by_label[row["label"]].append(row)

    train_split, val_split = [], []
    for label in sorted(by_label):
        rows = by_label[label]
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * VAL_FRACTION))
        val_split += rows[:n_val]
        train_split += rows[n_val:]

    # The source file is grouped by intent (see explore_data.py), so shuffle the
    # final lists too. Otherwise training would see one intent at a time.
    rng.shuffle(train_split)
    rng.shuffle(val_split)

    # Sanity checks: nothing lost, no overlap, every intent on both sides.
    assert len(train_split) + len(val_split) == len(unique)
    train_texts = {r["text"].lower() for r in train_split}
    assert not any(r["text"].lower() in train_texts for r in val_split), "train/val overlap!"
    assert len({r["label"] for r in train_split}) == len({r["label"] for r in val_split}) == 77

    for name, rows in (("train_split", train_split), ("val_split", val_split)):
        path = DATA_DIR / f"{name}.jsonl"
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        counts = Counter(r["label"] for r in rows)
        print(f"{name:11s}: {len(rows):5,} rows -> {path.relative_to(DATA_DIR.parent)} "
              f"(per intent: min {min(counts.values())}, max {max(counts.values())})")

    label_changes = sum(a["label"] != b["label"] for a, b in zip(train_split, train_split[1:]))
    print(f"\nshuffled check: label changes in {label_changes:,} of {len(train_split) - 1:,} "
          f"consecutive train_split rows (was 76 before shuffling)")
    rarest = min(by_label, key=lambda l: len(by_label[l]))
    n_val_rare = sum(r["label"] == rarest for r in val_split)
    print(f"rarest intent '{LABELS[rarest]}': {len(by_label[rarest]) - n_val_rare} train / "
          f"{n_val_rare} val")


if __name__ == "__main__":
    main()
