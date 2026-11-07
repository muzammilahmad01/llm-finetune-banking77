# LLM Fine-Tuning on Banking77

An end-to-end LLM engineering project: measure a small LLM on a 77-class banking intent
task, fine-tune it with LoRA, re-measure, and wrap it in guardrails.

**Status:** 🚧 In progress. Data, baseline and error analysis done; fine-tuning next.
See [plan.md](plan.md) for the full roadmap.

## Results so far

| Model | Accuracy | JSON-valid | Invented labels | Latency |
|---|---|---|---|---|
| `qwen2.5:0.5b` zero-shot (baseline) | 25.2% | 97.1% | 25.2% | 0.15 s |
| `qwen2.5:0.5b` + LoRA fine-tune | *coming* | | | |

Evaluated on the full Banking77 test set (3,080 examples, 40 per intent). Random guessing scores 1.3%.

**Why the baseline fails** ([error analysis](results/error_analysis_qwen2.5_0.5b.json)): it matches
keywords to label names instead of separating similar intents. For example, anything mentioning "verify"
becomes `verify_my_identity`, so its siblings `why_verify_identity` and `unable_to_verify_identity`
score 0%. A quarter of its answers are labels that don't exist.

## Project structure

```
backend/
  data.py            Download Banking77, cache as JSONL
  labels.py          The 77 intent names
  explore_data.py    Dataset statistics (balance, lengths, leakage)
  make_splits.py     Stratified train/validation split (seed 42)
  eval_baseline.py   Evaluate any Ollama model on the test set
  error_analysis.py  Break down how and why a model fails
results/             Metrics (JSON) and per-example predictions (CSV)
plan.md              Phase-by-phase roadmap and findings
```

## Reproduce

Requires Python 3.12 (via [uv](https://github.com/astral-sh/uv)) and [Ollama](https://ollama.com).

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

python backend/data.py            # download dataset
python backend/explore_data.py    # dataset statistics
python backend/make_splits.py     # train/validation split

ollama pull qwen2.5:0.5b
python backend/eval_baseline.py --limit 50 --show   # quick look
python backend/eval_baseline.py                     # full test set
python backend/error_analysis.py
```

## Dataset

[Banking77](https://huggingface.co/datasets/PolyAI/banking77) by PolyAI (Casanueva et al., 2020),
licensed CC BY 4.0. Prediction files in `results/` include test-set texts from this dataset.
