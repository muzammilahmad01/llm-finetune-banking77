"""Zero-shot baseline: ask an Ollama model to classify Banking77 test queries.

Examples:
    python backend/eval_baseline.py --limit 50
    python backend/eval_baseline.py --limit 20 --show   # watch each Q&A live
    python backend/eval_baseline.py                      # full 3,080 test examples
    python backend/eval_baseline.py --model qwen2.5:1.5b

Outputs (one pair per model):
    results/baseline_<model>.csv   per-example predictions (append-only, resumable)
    results/baseline_<model>.json  summary metrics
"""

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

from data import PROJECT_ROOT, load_split
from labels import LABEL_SET, LABELS
from prompts import SYSTEM_PROMPT, build_messages  # shared with training data

OLLAMA_URL = "http://localhost:11434"
RESULTS_DIR = PROJECT_ROOT / "results"
CSV_FIELDS = ["idx", "text", "gold", "pred", "json_valid", "in_vocab", "correct",
              "latency_s", "raw_response"]


def check_ollama(model: str) -> None:
    """Fail fast with a readable message if Ollama is down or the model is missing."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"ERROR: can't reach Ollama at {OLLAMA_URL} ({e}).\n"
                 "Start it in another terminal with:  ollama serve")
    available = [m["name"] for m in resp.json().get("models", [])]
    if model not in available:
        sys.exit(f"ERROR: model '{model}' isn't pulled in Ollama. Available: {available}\n"
                 f"Pull it with:  ollama pull {model}")


def classify(model: str, text: str) -> tuple[str, float]:
    """Send one query to Ollama. Returns (raw response text, latency in seconds)."""
    payload = {
        "model": model,
        "messages": build_messages(text),
        "stream": False,                    # one complete JSON response, not chunks
        "options": {"temperature": 0},      # greedy decoding: deterministic, repeatable
    }
    start = time.perf_counter()
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    latency = time.perf_counter() - start
    resp.raise_for_status()
    return resp.json()["message"]["content"], latency


def parse_response(raw: str) -> tuple[str | None, bool, bool]:
    """Parse the model's reply. Returns (predicted intent or None, json_valid, in_vocab).

    Small models often wrap JSON in markdown fences (```json ... ```). We strip
    those, but otherwise stay strict: extra prose around the JSON counts as invalid.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, False, False
    if not isinstance(obj, dict) or not isinstance(obj.get("intent"), str):
        # Parsed, but not the shape we asked for -> treat as invalid output.
        return None, False, False
    intent = obj["intent"].strip()
    return intent, True, intent in LABEL_SET


def load_done(csv_path: Path) -> dict[int, dict]:
    """Rows already in the CSV, keyed by example index (this is what makes runs resumable)."""
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="") as f:
        return {int(row["idx"]): row for row in csv.DictReader(f)}


def summarize(rows: list[dict], model: str) -> dict:
    """Aggregate per-example rows into summary metrics. All rates use n as denominator."""
    n = len(rows)
    flag = lambda key: sum(row[key] in ("True", True) for row in rows)
    json_valid = flag("json_valid")
    in_vocab = flag("in_vocab")
    return {
        "model": model,
        "n": n,
        "accuracy": flag("correct") / n,
        "json_valid_rate": json_valid / n,
        # Valid JSON, but the intent isn't one of the 77 labels (a made-up label).
        "oov_rate": (json_valid - in_vocab) / n,
        "mean_latency_s": sum(float(row["latency_s"]) for row in rows) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--limit", type=int, default=None,
                        help="only evaluate N test examples (a fixed random sample)")
    parser.add_argument("--show", action="store_true",
                        help="print every question, the model's answer, and the correct intent")
    args = parser.parse_args()

    check_ollama(args.model)

    test = load_split("test")  # never used for training, only for evaluation

    # The test file is grouped by intent, so "the first N rows" would cover only
    # one or two intents. Instead, visit examples in a fixed shuffled order
    # (seed 42): --limit N then gives a representative sample across intents,
    # and the same N examples every time. `idx` always refers to the original row.
    order = list(range(len(test)))
    random.Random(42).shuffle(order)
    if args.limit:
        order = order[: args.limit]

    # "qwen2.5:0.5b" -> "qwen2.5_0.5b" so it's a safe filename.
    safe_name = re.sub(r"[^\w.-]", "_", args.model)
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / f"baseline_{safe_name}.csv"
    json_path = RESULTS_DIR / f"baseline_{safe_name}.json"

    done = load_done(csv_path)
    todo = [i for i in order if i not in done]
    print(f"{len(order)} examples: {len(order) - len(todo)} already done, {len(todo)} to run")

    is_new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new_file:
            writer.writeheader()
        for i in tqdm(todo, desc=args.model, unit="ex"):
            ex = test[i]
            try:
                raw, latency = classify(args.model, ex["text"])
            except requests.RequestException as e:
                # Stop instead of recording a fake failure; rerun later to resume here.
                sys.exit(f"\nERROR: request failed on example {i}: {e}\n"
                         "Progress so far is saved. Rerun the same command to resume.")
            pred, json_valid, in_vocab = parse_response(raw)
            gold = LABELS[ex["label"]]
            row = {"idx": i, "text": ex["text"], "gold": gold, "pred": pred or "",
                   "json_valid": json_valid, "in_vocab": in_vocab,
                   "correct": in_vocab and pred == gold,
                   "latency_s": round(latency, 4), "raw_response": raw}
            writer.writerow(row)
            f.flush()  # write each row to disk immediately so a crash loses nothing
            if args.show:
                # tqdm.write prints above the progress bar without breaking it
                mark = "✅" if row["correct"] else ("❌" if json_valid else "⚠️ ")
                tqdm.write(f"{mark} Q: {ex['text']}\n   model: {raw.strip()}\n   correct: {gold}\n")
            done[i] = row

    # Metrics cover exactly the examples requested this run (the N-sample if --limit).
    metrics = summarize([done[i] for i in order], args.model)
    json_path.write_text(json.dumps(metrics, indent=2))

    print(f"\nResults for {args.model} on {metrics['n']} examples:")
    print(f"  accuracy        {metrics['accuracy']:.1%}")
    print(f"  JSON-valid rate {metrics['json_valid_rate']:.1%}")
    print(f"  OOV-label rate  {metrics['oov_rate']:.1%}")
    print(f"  mean latency    {metrics['mean_latency_s']:.3f}s")
    print(f"Saved: {csv_path.relative_to(PROJECT_ROOT)}, {json_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
