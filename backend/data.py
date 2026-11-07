"""Load Banking77 splits, caching them locally as JSONL in data/.

After the first download, everything reads from data/*.jsonl, so later runs
don't need the network or even the `datasets` library.
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_ID = "PolyAI/banking77"  # exact ID; plain "banking77" is deprecated

# Keep the Hugging Face download cache inside the project folder instead of
# ~/.cache/huggingface. Must be set before `datasets` is imported.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))


def _download(split: str):
    """Download one split from the Hub, returning the HF Dataset object."""
    from datasets import load_dataset  # imported lazily: only needed once

    return load_dataset(DATASET_ID, split=split)


def load_label_names() -> list[str]:
    """Return the 77 intent names, in label-id order (index i == label i)."""
    path = DATA_DIR / "labels.json"
    if not path.exists():
        ds = _download("test")
        # `label` is a ClassLabel feature: ints on disk, names stored in the schema.
        names = ds.features["label"].names
        DATA_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(names, indent=2))
    return json.loads(path.read_text())


def load_split(split: str) -> list[dict]:
    """Return a split as a list of {"text": str, "label": int} dicts.

    "train" / "test":           the original Hugging Face splits (downloaded on first use)
    "train_split" / "val_split": our 90/10 split of train (created by make_splits.py)
    """
    path = DATA_DIR / f"{split}.jsonl"
    if split in ("train_split", "val_split"):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run: python backend/make_splits.py")
    elif split not in ("train", "test"):
        raise ValueError(f"unknown split: {split}")
    elif not path.exists():
        ds = _download(split)
        DATA_DIR.mkdir(exist_ok=True)
        with path.open("w") as f:
            for row in ds:
                f.write(json.dumps({"text": row["text"], "label": row["label"]}) + "\n")
    with path.open() as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    # `python backend/data.py` downloads (first time only) and caches everything.
    labels = load_label_names()
    for split in ("train", "test"):
        rows = load_split(split)
        print(f"{split:5s}: {len(rows):,} examples -> data/{split}.jsonl")
    print(f"labels: {len(labels)} intents -> data/labels.json")
    print("\nExample row:", rows[0], "->", labels[rows[0]["label"]])
