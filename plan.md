# Project Plan: Banking77 Intent Classifier

**Fine-tune a small LLM, measure it honestly, and wrap it in guardrails.**

This is our roadmap. Each phase is one step of the standard **LLM fine-tuning lifecycle**.
We finish and understand each phase before we start the next.

---

## The big picture

A RAG app follows a fixed recipe: *gather data → chunk → embed → store → retrieve → generate*.
Fine-tuning has a recipe too, and this project follows it:

```
 1. Define the task        What should the model do? How will we know it's good?
 2. Get the data           Load it, look at it, split it (train / validation / test)
 3. Measure a baseline     How good is the model BEFORE we change anything?
 4. Analyze errors         Where and why does it fail?
 5. Prepare training data  Turn examples into the chat format the model learns from
 6. Fine-tune (LoRA)       Teach the model on a GPU (Colab/Kaggle)
 7. Export & serve         Bring the trained model back into Ollama
 8. Re-evaluate            Same test, same script → compare before vs. after
 9. Add guardrails         Validate inputs and outputs for production safety
10. Ship & write up        Results table + portfolio write-up on GitHub
```

The **golden rule** of this project: *the test set is only for measuring.* We never train on it,
never tune prompts by looking at it, and always use the same eval script so the
before/after comparison is fair.

### What runs where

| Where | What | Why |
|---|---|---|
| MacBook (local) | Data loading, evaluation, guardrails | Light work; Ollama runs the small model |
| Ollama (local) | `qwen2.5:0.5b`, then our fine-tuned version | Free, private inference |
| Colab / Kaggle (free GPU) | LoRA fine-tuning | Training needs a GPU; your Mac's RAM is too small |
| Groq API | Big-model comparison, LLM-as-judge (optional) | Fast inference only; no fine-tuning |

---

## Repo layout (target)

```
LLM/
├── README.md               Project overview + final results table (Phase 9)
├── plan.md                 This file
├── requirements.txt        Python dependencies
├── .gitignore              Keeps .venv/, caches, secrets out of git
├── backend/                All Python code
│   ├── data.py             Load Banking77 + cache as JSONL
│   ├── labels.py           The 77 intent names
│   ├── explore_data.py     Dataset statistics (Phase 1)
│   ├── make_splits.py      Train/validation split (Phase 1)
│   ├── error_analysis.py   How/why a model fails (Phases 3 & 7)
│   ├── eval_baseline.py    Evaluate any Ollama model (Phases 2 & 7)
│   ├── prepare_sft.py      Build training files (Phase 4)
│   └── guardrails.py       Input/output validation (Phase 8)
├── notebooks/
│   └── finetune_lora.ipynb Colab training notebook (Phase 5)
├── data/                   Cached dataset files (not committed to git)
└── results/                Metrics JSON + prediction CSVs (committed)
```

---

## Phase 0: Setup and project scaffolding

**Goal:** a clean, reproducible project that anyone can clone and run.

**You'll learn:** virtual environments, pinned dependencies, what goes in git and what doesn't.

**Steps**
1. ✅ Create the virtual environment with **uv** (Python 3.12) and install `requirements.txt`:
   ```bash
   uv venv .venv --python 3.12 --python-preference only-managed
   source .venv/bin/activate
   uv pip install -r requirements.txt     # use `uv pip`, not `pip`/`pip3`
   ```
2. Add `.gitignore` (ignore `.venv/`, `.cache/`, `data/`, `.env`).
3. `git init`, first commit, push to GitHub.
4. Check Ollama: `curl http://localhost:11434/api/tags` should list `qwen2.5:0.5b`.

**Done when:** `python -c "import datasets"` works inside `.venv`, and the repo is on GitHub.

> **Why uv?** Homebrew's Python builds (3.12 and 3.15) are broken on macOS 26.0.1: their XML
> module needs a newer system `libexpat`, which breaks pip. uv's Python bundles its own libraries.
> A uv venv has no `pip` inside it, so a plain `pip3` falls through to Homebrew's system pip,
> which refuses to install (PEP 668 "externally-managed-environment"). Always use `uv pip install`.

---

## Phase 1: Load and explore the data

**Goal:** know your data before touching any model. Most real ML problems are data problems.

**You'll learn:** Hugging Face `datasets`, the `ClassLabel` feature (labels are stored as ints,
with names in the schema), train/test splits, why we also need a **validation** split, and class balance.

**Steps**
1. ✅ Download `PolyAI/banking77` and cache each split as JSONL in `data/` (`python backend/data.py`).
2. ✅ `explore_data.py`: split sizes, examples per intent, text length, duplicates/leakage, samples.
3. ✅ Find confusable intents by reading samples, e.g. `card_arrival` vs. `card_delivery_estimate`.
4. ✅ Carve a stratified **validation split** (10%, seed 42) out of *train* (`python backend/make_splits.py`).

**Done when:** you can explain, in your own words, what the 77 intents look like and which
ones will be hard.

**What we found**
- Train 10,003 / test 3,080. Test is perfectly balanced (40 per intent), so accuracy is a fair metric.
  Train is imbalanced 5.3× (35 to 187 per intent).
- The train file is **grouped by intent** (only 76 label changes in 10,002 rows). Always shuffle.
- Messages are short: median 10 words, p95 29 words. The 77-label list will dominate the prompt length.
- 4 duplicates in train (dropped), 7 test texts also appear in train (0.2% leakage; mention in write-up).
- Random-guess and majority-class baselines are both 1.3%.
- Final files: `train_split.jsonl` 8,997 rows, `val_split.jsonl` 1,002 rows (every intent present in both),
  `test.jsonl` 3,080 rows (untouched).

---

## Phase 2: Zero-shot baseline evaluation

**Goal:** measure how well the *unmodified* model does. Without a baseline, "fine-tuning helped" is just a claim.

**You'll learn:** prompting for classification, structured (JSON) output, temperature 0 for
repeatable results, the Ollama REST API, and choosing metrics.

**Steps** *(code written: `backend/eval_baseline.py`)*
1. For each test query, send the 77 labels plus the query to Ollama and ask for `{"intent": "<label>"}`.
2. Score every response:
   - **Accuracy**: predicted intent == true intent
   - **JSON-validity rate**: did it return parseable JSON in the right shape?
   - **OOV rate**: valid JSON, but an intent that doesn't exist (a made-up label)
   - **Mean latency**: seconds per request
3. Run on `--limit 50` first, read the outputs by hand, then run the full 3,080.
4. *(Optional)* Run the same eval on a large model via Groq as a reference "ceiling".

**Done when:** `results/baseline_qwen2.5_0.5b.json` exists for the full test set.

**Expectation:** a 0.5B model will likely do poorly here (77 classes is hard zero-shot).
That's fine. A low baseline gives fine-tuning room to show its value.

---

## Phase 3: Error analysis

**Goal:** understand *how* the baseline fails, so we know what fine-tuning should fix.

**You'll learn:** confusion analysis, the difference between *format* errors and *knowledge* errors.

**Steps**
1. From the predictions CSV: the top 10 most-confused (true → predicted) pairs.
2. Per-intent accuracy: which intents are easiest and hardest?
3. Categorize the failures: broken JSON? Invented labels? Plausible-but-wrong labels?

**Done when:** you can say "the base model mostly fails because of X."

**What we found** (`python backend/error_analysis.py`)
- Outcomes: 25.2% correct, **46.7% real-but-wrong label**, 25.2% invented label, 2.9% invalid JSON.
- Diagnosis: *the base model mostly fails because it matches keywords in the message to words in
  label names, instead of learning the fine distinctions between similar intents.* E.g. anything
  with "verify" → `verify_my_identity` (predicted 131×, right only 37×).
- Invented labels: 47 differ only in case, 215 are close spellings, 515 are made up. Mapping
  near-misses would add only +3.6%, so making scoring more lenient won't fix it; the model must learn.
- 20 of 77 intents score 0%; 12 intents are never predicted at all.
- Top confusions are sibling intents: `why_verify_identity` / `unable_to_verify_identity` →
  `verify_my_identity`; `failed_transfer` → `transfer_not_received_by_recipient`.

---

## Phase 4: Prepare the fine-tuning dataset

**Goal:** turn Banking77's `train` split into examples the model can learn from.

**You'll learn:** supervised fine-tuning (SFT) data format, chat templates, tokens and sequence
length, and why training and inference prompts must match.

**Steps**
1. Convert each train example into a chat conversation:
   `system` (instructions) → `user` (query) → `assistant` (`{"intent": "card_arrival"}`).
2. Decide whether the 77-label list stays in the prompt (a trade-off: longer prompts vs. a model
   that has memorized the labels). We'll discuss this when we get there.
3. Write `data/sft_train.jsonl` and `data/sft_val.jsonl`.
4. Count tokens per example to choose a max sequence length.

**Done when:** the JSONL files exist and a sample row looks right.

---

## Phase 5: LoRA fine-tuning (on Colab / Kaggle)

**Goal:** fine-tune `Qwen2.5-0.5B-Instruct` on our training data using a free GPU.

**You'll learn:**
- **LoRA**: instead of updating all 500M weights, train small "adapter" matrices (~1% of weights)
- **QLoRA**: load the base model in 4-bit to save GPU memory
- Hyperparameters: rank `r`, `alpha`, learning rate, epochs, batch size
- Reading **loss curves**: training loss vs. validation loss, and spotting overfitting
- Tools: Unsloth (fast, beginner-friendly) or Hugging Face `peft` + `trl`'s `SFTTrainer`

**Steps**
1. Build `notebooks/finetune_lora.ipynb`: install libraries, load the model, attach LoRA, train.
2. Train 1–3 epochs and watch validation loss.
3. Quick sanity check in the notebook on a few validation examples.
4. Save the LoRA adapter and push it to the Hugging Face Hub (optional, but good for your portfolio).

**Done when:** validation loss has dropped and the sample predictions look sensible.

---

## Phase 6: Export and serve locally

**Goal:** run the fine-tuned model in Ollama on your Mac, exactly like the base model.

**You'll learn:** merging adapters into base weights, the **GGUF** format, **quantization**
(e.g. Q4_K_M: smaller and faster, slightly less accurate), and Ollama `Modelfile`s.

**Steps**
1. In Colab: merge LoRA into the base model and export to GGUF (Unsloth does this in one call).
2. Download the `.gguf` file, write a `Modelfile`, and run `ollama create banking77-qwen -f Modelfile`.
3. Smoke test: `ollama run banking77-qwen "my card hasn't arrived yet"`.

**Done when:** `ollama list` shows the fine-tuned model.

---

## Phase 7: Re-evaluate and compare

**Goal:** a fair before/after comparison.

**You'll learn:** controlled experiments: change one thing (the weights) and keep everything else fixed.

**Steps**
1. `python backend/eval_baseline.py --model banking77-qwen` on the full test set (same script, same metrics).
2. Build the comparison table:

   | Model | Accuracy | JSON-valid | OOV | Latency |
   |---|---|---|---|---|
   | qwen2.5:0.5b (zero-shot) | | | | |
   | banking77-qwen (LoRA) | | | | |
   | large model via Groq (reference) | | | | |

3. Repeat the Phase 3 error analysis on the new model: which confusions remain?

**Done when:** the table is filled in with real numbers.

---

## Phase 8: Guardrails

**Goal:** make the classifier safe to put in front of real users.

**You'll learn:** defense in depth: never trust inputs, never trust model outputs.

**Input guardrails** (before the model)
- Empty or overly long input → reject
- Mask PII (card numbers, IBANs, emails) before sending text to any model
- Off-topic / prompt-injection detection ("ignore previous instructions…")

**Output guardrails** (after the model)
- Validate with a `pydantic` schema: the intent must be one of the 77 labels
- On invalid output: retry once, then fall back to `"unknown"` and hand off to a human

**Steps:** write `backend/guardrails.py`, plus a small set of adversarial test inputs, and measure
how often each guardrail fires.

**Done when:** bad inputs are blocked and every output is either a valid label or `"unknown"`.

---

## Phase 9: Write-up and ship

**Goal:** a portfolio piece someone can understand in 2 minutes.

**Steps**
1. `README.md`: problem, approach, results table, how to reproduce, lessons learned.
2. Clean up code and commit the results.
3. A short write-up: what fine-tuning changed, what it cost, what you'd do next.

**Done when:** the GitHub repo is public and the README tells the whole story.

---

## Progress tracker

- [x] Phase 0: Setup (repo: https://github.com/muzammilahmad01/llm-finetune-banking77)
- [x] Phase 1: Load and explore data
- [x] Phase 2: Baseline eval (qwen2.5:0.5b zero-shot: 25.2% accuracy, 97.1% JSON-valid, 25.2% OOV)
- [x] Phase 3: Error analysis (main failure: keyword-matching between sibling intents)
- [ ] Phase 4: Prepare SFT data
- [ ] Phase 5: LoRA fine-tune
- [ ] Phase 6: Export to Ollama
- [ ] Phase 7: Re-evaluate and compare
- [ ] Phase 8: Guardrails
- [ ] Phase 9: Write-up and ship
