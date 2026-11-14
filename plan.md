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
├── WRITEUP.md              Short write-up: what changed, cost, next steps (Phase 9)
├── Modelfile               Ollama recipe for banking77-qwen (Phase 6)
├── plan.md                 This file
├── requirements.txt        Python dependencies
├── .gitignore              Keeps .venv/, caches, secrets out of git
├── backend/                All Python code
│   ├── data.py             Load Banking77 + cache as JSONL
│   ├── labels.py           The 77 intent names
│   ├── prompts.py          Shared prompt for eval + training (Phase 4)
│   ├── explore_data.py     Dataset statistics (Phase 1)
│   ├── make_splits.py      Train/validation split (Phase 1)
│   ├── error_analysis.py   How/why a model fails (Phases 3 & 7)
│   ├── eval_baseline.py    Evaluate any Ollama model (Phases 2 & 7)
│   ├── prepare_sft.py      Build training files (Phase 4)
│   ├── merge_adapter.py    Merge LoRA adapter, write Modelfile (Phase 6)
│   ├── guardrails.py       Input/output validation (Phase 8)
│   └── eval_guardrails.py  Measure guardrails, pick threshold (Phase 8)
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
1. ✅ Convert each train example into a chat conversation:
   `system` (instructions) → `user` (query) → `assistant` (`{"intent": "card_arrival"}`).
2. ✅ Decision: **keep the 77-label list in the prompt** (option A), so Phase 7 changes only the
   weights. The prompt now lives in `backend/prompts.py`, shared by eval and training (verified
   byte-identical to the baseline prompt). A short-prompt variant is a possible follow-up experiment.
3. ✅ Write `data/sft_train.jsonl` and `data/sft_val.jsonl` (`python backend/prepare_sft.py`).
4. ✅ Count tokens per example with Qwen's real tokenizer.

**Done when:** the JSONL files exist and a sample row looks right.

**What we found**
- 8,997 train / 1,002 val conversations. Every target answer passes our own eval parser.
- ~474 tokens per conversation: **441 are the shared system prompt**, ~20 the customer message,
  ~9 the answer. Only ~2% of tokens are the answer, so in Phase 5 we compute the loss
  **only on the assistant answer** (otherwise most of the training signal is spent re-reading the label list).
- Longest example: 557 tokens, so **`max_seq_length = 576`** (no truncation).
- ~4.3M tokens per epoch.

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
1. ✅ Build `notebooks/finetune_lora.ipynb`: install libraries, load the model, attach LoRA, train.
2. ✅ Train 1–3 epochs and watch validation loss.
3. ✅ Quick sanity check in the notebook on a few validation examples.
4. ✅ Save the LoRA adapter (downloaded to `models/banking77-qwen-adapter/`, not in git). HF Hub push skipped for now.

**Done when:** validation loss has dropped and the sample predictions look sensible.

**What we found** (Colab T4, 35 min)
- QLoRA via Unsloth: r=16, alpha=32, all 7 attention/MLP projections, 8.8M trainable params (1.75%).
  2 epochs, effective batch 16, lr 2e-4 cosine, 1,126 steps. Loss computed on the assistant answer only.
- Validation loss: 0.060 (step 250) → 0.040 → 0.032 → 0.026 → **0.026** (step 1126). It kept falling and
  then flattened; it never rose, so **no overfitting**. Train loss ended ~0.016, a small gap below validation.
  Plot: `results/loss_curves_lora.png`.
- Sanity check: **10/10** validation examples correct, all valid JSON.
- Adapter: 35 MB (`adapter_model.safetensors`). Trained against Unsloth's 4-bit base; we merge into the
  full-precision base in Phase 6.
- Lessons: newer TRL removed `DataCollatorForCompletionOnlyLM` (we wrote our own collator), and merging +
  GGUF export inside Colab ran out of memory and killed the runtime, losing the first run. Save only the
  small adapter in Colab and download it right away; do the merge locally.

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

**What we did** (all on the Mac, no Colab)
- `python backend/merge_adapter.py`: merged the LoRA adapter into the full-precision
  `Qwen/Qwen2.5-0.5B-Instruct` on CPU. 168 of 291 tensors changed (24 layers × 7 projections),
  as expected. Also writes the `Modelfile`.
- Ollama 0.40 can't make Q4_K_M from safetensors (only int4/int8/fp4/fp8). The baseline is Q4_K_M,
  so for a fair comparison we used llama.cpp: `convert_hf_to_gguf.py` → f16 GGUF (948 MB) →
  `llama-quantize … Q4_K_M` (379 MB) → `ollama create banking77-qwen -f Modelfile`.
- The `Modelfile` copies the baseline's chat template and sets our classifier prompt as the default
  system prompt, so `ollama run banking77-qwen "my card hasn't arrived"` works out of the box.
- Result: `banking77-qwen` is Q4_K_M, 397 MB, same as `qwen2.5:0.5b`. Only the weights differ.

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

**What we found** (full test set, 3,080 examples, same script and prompt, both Q4_K_M)

| Model | Accuracy | JSON-valid | OOV | Latency |
|---|---|---|---|---|
| qwen2.5:0.5b (zero-shot) | 25.2% | 97.1% | 25.2% | 0.146 s |
| banking77-qwen (LoRA) | **90.7%** | **99.2%** | **0.2%** | 0.143 s |

- **3.6× accuracy** at the same size and speed. Fine-tuning changed only the weights.
- Outcomes: 90.7% correct, 8.3% real-but-wrong label, 0.2% invented, 0.8% invalid JSON.
  The baseline was 25.2% / 46.7% / 25.2% / 2.9%.
- **No intent scores 0%** (the baseline had 20 at 0%). The remaining confusions are genuine siblings:
  `topping_up_by_card` → `top_up_reverted` (7×), `beneficiary_not_allowed` → `failed_transfer`,
  `pending_transfer` → `transfer_not_received_by_recipient`, `card_delivery_estimate` → `card_arrival`.
  Some of these are arguably label noise in Banking77.
- **Formatting glitch:** all 24 invalid-JSON answers (and the 7 "invented" labels) have the *right*
  intent but drop the closing quote: `{"intent": "automatic_top_up}`. 16 of them are
  `automatic_top_up`, which explains its 50% score. Fixing this would add up to ~1% accuracy.
  Possible causes: Q4 quantization, or the merge into the full-precision base instead of the 4-bit one
  it was trained on. We keep strict scoring here; Phase 8's output guardrail (retry → `unknown`) handles it.
- Mid-run, the Homebrew `ollama serve` process stopped; the eval resumed from its saved CSV with no lost work.
  Tip: `brew services start ollama` keeps it running.

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

**What we built** (`backend/guardrails.py`, measured by `backend/eval_guardrails.py`)
- Input: empty / over 1,000 chars / pure small talk / prompt injection → blocked before the model.
  Card numbers, IBANs and emails → masked (`[CARD_NUMBER]`, `[IBAN]`, `[EMAIL]`), also in our logs.
  Small talk is only blocked when the *whole* message is small talk: 190 real messages start with "Hi"/"Hello".
- Output: strict parse → **repair** the garbled-ending glitch (`"automatic_top_up}`) only if the label is an
  exact match → pydantic check that it is one of the 77 → **confidence** check → else `"unknown"`.
- Changed from the plan: no "retry once". At temperature 0 a retry returns the identical answer, so we
  repair deterministically instead.
- **Confidence** = probability of the whole reply (product of token probabilities, via Ollama `logprobs`).
  In scope it is ~0.98; "open a new bank account" 0.41; "hey" 0.03.
- **Threshold 0.6**, chosen on validation data only: our `val_split` (1,002) + CLINC150 out-of-scope
  validation (100). Out-of-scope test data: CLINC150 OOS test (1,000), never used for choosing.

**What we found** (test sets)

| Banking77 test (3,080) | No guardrails | Repair only | + confidence ≥ 0.6 |
|---|---|---|---|
| Correct | 90.7% | 91.7% | 88.4% |
| **Wrong** (confidently) | 9.3% | 8.3% | **4.4%** |
| Sent to a human (`unknown`) | 0% | 0% | 7.1% |
| Precision (right when it answers) | 90.7% | 91.7% | **95.2%** |

| Out-of-scope test (CLINC150, 1,000) | Repair only | + confidence ≥ 0.6 |
|---|---|---|
| Caught (`unknown`) | 34.4% (invented labels) | **82.7%** |
| Wrongly given a banking label | 65.6% | 17.3% |

- Input rules: **0 of 13,083** real Banking77 messages blocked (no false positives). Hand-written
  adversarial set: **27/27** as expected (empty, too long, small talk, 5 injections, 8 off-topic, PII, "Hi, …").
- Trade-off: the confidence check halves confident mistakes (8.3% → 4.4%) and catches 83% of off-topic
  messages, at the cost of 7.1% of real customers going to a human. Latency unchanged (0.143 s).
- Limits: rules catch known patterns only; 17% of off-topic messages still get a label. The real fix is to
  retrain with an `out_of_scope` class (e.g. CLINC150 OOS train data) as a follow-up.

---

## Phase 9: Write-up and ship

**Goal:** a portfolio piece someone can understand in 2 minutes.

**Steps**
1. `README.md`: problem, approach, results table, how to reproduce, lessons learned.
2. Clean up code and commit the results.
3. A short write-up: what fine-tuning changed, what it cost, what you'd do next.

**Done when:** the GitHub repo is public and the README tells the whole story.

**What we did**
- `README.md` rewritten: headline result, before/after and guardrail tables, pipeline diagram, findings,
  fine-tuning setup, full reproduce steps, limitations.
- `WRITEUP.md`: what fine-tuning changed, what it cost, how the guardrails work, what's next.
- Cleanup: the notebook's Colab GGUF export cells (which crashed the runtime) are replaced with
  "zip and download the adapter", plus pointers to the local merge.
- Repo is public: https://github.com/muzammilahmad01/llm-finetune-banking77

---

## Progress tracker

- [x] Phase 0: Setup (repo: https://github.com/muzammilahmad01/llm-finetune-banking77)
- [x] Phase 1: Load and explore data
- [x] Phase 2: Baseline eval (qwen2.5:0.5b zero-shot: 25.2% accuracy, 97.1% JSON-valid, 25.2% OOV)
- [x] Phase 3: Error analysis (main failure: keyword-matching between sibling intents)
- [x] Phase 4: Prepare SFT data (label list kept in prompt; max_seq_length 576)
- [x] Phase 5: LoRA fine-tune (val loss 0.026, no overfitting, 10/10 sanity check)
- [x] Phase 6: Export to Ollama (`banking77-qwen`, Q4_K_M via llama.cpp, matches baseline format)
- [x] Phase 7: Re-evaluate and compare (25.2% → 90.7% accuracy; OOV 25.2% → 0.2%)
- [x] Phase 8: Guardrails (wrong answers 9.3% → 4.4%, 83% of off-topic caught, 0 real messages blocked)
- [x] Phase 9: Write-up and ship (README, WRITEUP.md, notebook cleanup)
