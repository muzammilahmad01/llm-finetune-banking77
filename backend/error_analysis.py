"""Phase 3: analyze *how* a model fails, using its saved eval predictions.

Run:  python backend/error_analysis.py                    # base model
      python backend/error_analysis.py --model banking77-qwen   # later, fine-tuned

Reads   results/baseline_<model>.csv   (written by eval_baseline.py; no model calls here)
Writes  results/error_analysis_<model>.json

Sections:
  1. Outcome breakdown  -> what kind of errors dominate?
  2. Near-miss labels   -> are invented labels close to real ones?
  3. Per-intent accuracy-> which intents are easy / hard?
  4. Confused pairs     -> which real intents get mixed up?
  5. Label bias         -> which answers does the model overuse?
  6. Training data size -> do intents with fewer training examples do worse?
"""

import argparse
import csv
import difflib
import json
import re
from collections import Counter, defaultdict

from data import PROJECT_ROOT, load_split
from labels import LABEL_SET, LABELS

RESULTS_DIR = PROJECT_ROOT / "results"


def header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def is_true(value: str) -> bool:
    return value == "True"  # CSV stores booleans as the strings "True"/"False"


def nearest_label(pred: str) -> tuple[str | None, str]:
    """Classify an invented label: how close is it to a real one?"""
    lowered = {l.lower(): l for l in LABELS}
    if pred.lower() in lowered:
        return lowered[pred.lower()], "case only"  # e.g. refund_not_showing_up
    # difflib ratio: 1.0 = identical strings. 0.8+ = a small spelling difference.
    match = difflib.get_close_matches(pred, LABELS, n=1, cutoff=0.8)
    if match:
        return match[0], "close spelling"
    return None, "no close label"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:0.5b")
    args = parser.parse_args()

    safe_name = re.sub(r"[^\w.-]", "_", args.model)
    csv_path = RESULTS_DIR / f"baseline_{safe_name}.csv"
    rows = list(csv.DictReader(csv_path.open(newline="")))
    n = len(rows)
    print(f"Analyzing {n:,} predictions from {csv_path.relative_to(PROJECT_ROOT)}")

    # 1. Outcome breakdown -----------------------------------------------------
    # Every prediction falls into exactly one bucket.
    header("1. OUTCOME BREAKDOWN")
    buckets = Counter()
    for r in rows:
        if is_true(r["correct"]):
            buckets["correct"] += 1
        elif not is_true(r["json_valid"]):
            buckets["invalid JSON"] += 1
        elif not is_true(r["in_vocab"]):
            buckets["invented label (OOV)"] += 1
        else:
            buckets["real label, wrong intent"] += 1
    for name in ("correct", "real label, wrong intent", "invented label (OOV)", "invalid JSON"):
        print(f"  {name:26s} {buckets[name]:5,}  ({buckets[name] / n:5.1%})")

    # 2. Near-miss invented labels ---------------------------------------------
    header("2. INVENTED LABELS: how close to a real one?")
    oov_rows = [r for r in rows if is_true(r["json_valid"]) and not is_true(r["in_vocab"])]
    kinds = Counter()
    would_be_correct = 0  # if we mapped the near-miss to its real label
    examples = defaultdict(list)
    for r in oov_rows:
        nearest, kind = nearest_label(r["pred"])
        kinds[kind] += 1
        if nearest == r["gold"]:
            would_be_correct += 1
        if len(examples[kind]) < 4:
            examples[kind].append(f"{r['pred']!r} -> {nearest or '?'}  (gold: {r['gold']})")
    print(f"  {len(oov_rows):,} invented answers, {len({r['pred'] for r in oov_rows})} distinct strings")
    for kind in ("case only", "close spelling", "no close label"):
        print(f"\n  {kind}: {kinds[kind]:,}")
        for e in examples[kind]:
            print(f"    {e}")
    print(f"\n  If near-misses were mapped to the nearest real label, accuracy would rise by "
          f"{would_be_correct / n:.1%} (+{would_be_correct} examples).")
    print("  (We keep strict scoring: a production system must emit exact labels.)")

    # 3. Per-intent accuracy ---------------------------------------------------
    header("3. PER-INTENT ACCURACY (40 test examples each)")
    per_intent = defaultdict(lambda: [0, 0])  # gold -> [correct, total]
    for r in rows:
        per_intent[r["gold"]][0] += is_true(r["correct"])
        per_intent[r["gold"]][1] += 1
    acc = {label: c / t for label, (c, t) in per_intent.items()}
    ranked = sorted(acc.items(), key=lambda kv: kv[1])
    print("  Hardest:")
    for label, a in ranked[:10]:
        print(f"    {a:5.0%}  {label}")
    print("  Easiest:")
    for label, a in ranked[::-1][:10]:
        print(f"    {a:5.0%}  {label}")
    zero = [label for label, a in acc.items() if a == 0]
    print(f"\n  Intents with 0% accuracy: {len(zero)} of 77")

    # 4. Confused pairs --------------------------------------------------------
    header("4. MOST CONFUSED PAIRS (real label, wrong intent)")
    confusions = Counter((r["gold"], r["pred"]) for r in rows
                         if is_true(r["in_vocab"]) and not is_true(r["correct"]))
    for (gold, pred), count in confusions.most_common(12):
        print(f"  {count:3d}x  {gold:42s} -> {pred}")

    # 5. Label bias ------------------------------------------------------------
    header("5. LABEL BIAS (each intent truly appears 40 times)")
    pred_counts = Counter(r["pred"] for r in rows if is_true(r["in_vocab"]))
    for label, count in pred_counts.most_common(8):
        right = sum(1 for r in rows if r["pred"] == label and is_true(r["correct"]))
        print(f"  predicted {count:4d}x ({count / 40:.1f}x expected), right only {right:2d}x  {label}")
    never = [l for l in LABELS if l not in pred_counts]
    print(f"\n  Real intents the model NEVER predicts: {len(never)} of 77")
    print(f"    e.g. {never[:6]}")

    # 6. Training data size vs accuracy ----------------------------------------
    header("6. TRAINING-DATA SIZE vs ACCURACY")
    train_counts = Counter(LABELS[r["label"]] for r in load_split("train_split"))
    by_size = sorted(LABELS, key=lambda l: train_counts[l])
    rare, common = by_size[:15], by_size[-15:]
    mean = lambda ls: sum(acc[l] for l in ls) / len(ls)
    print(f"  15 rarest intents in training   ({train_counts[rare[0]]}-{train_counts[rare[-1]]} examples): "
          f"mean accuracy {mean(rare):.1%}")
    print(f"  15 most common intents in training ({train_counts[common[0]]}-{train_counts[common[-1]]} examples): "
          f"mean accuracy {mean(common):.1%}")
    print("  (The base model never saw our training data, so any gap here is chance.\n"
          "   After fine-tuning, a gap would mean rare intents need more data.)")

    # Save everything for the Phase 7 before/after comparison --------------------
    out = {
        "model": args.model,
        "n": n,
        "outcomes": {k: v / n for k, v in buckets.items()},
        "oov_near_miss": dict(kinds),
        "accuracy_gain_if_near_misses_mapped": would_be_correct / n,
        "per_intent_accuracy": dict(sorted(acc.items())),
        "zero_accuracy_intents": sorted(zero),
        "top_confusions": [{"gold": g, "pred": p, "count": c}
                           for (g, p), c in confusions.most_common(20)],
        "never_predicted": never,
        "rare_vs_common_train_accuracy": {"rare15": mean(rare), "common15": mean(common)},
    }
    out_path = RESULTS_DIR / f"error_analysis_{safe_name}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
