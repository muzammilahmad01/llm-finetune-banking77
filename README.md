# Fine-tuning a small LLM for banking intent classification

A 0.5B-parameter model went from **25% to 91% accuracy** on a 77-class banking task after LoRA fine-tuning.
It is the same size and the same speed, and it runs locally in Ollama on a laptop. Guardrails then
**halved its confident mistakes** and taught it to say "I don't know" to off-topic questions.

The project runs end to end: measure the baseline, find out why it fails, fine-tune on a free Colab GPU,
export to Ollama, re-measure with the same script, then make it safe to put in front of users.

**Adapter on Hugging Face:** [MuzammilAhmad/banking77-qwen2.5-0.5b-lora](https://huggingface.co/MuzammilAhmad/banking77-qwen2.5-0.5b-lora)

## Results

Full [Banking77](https://huggingface.co/datasets/PolyAI/banking77) test set: 3,080 customer messages,
40 per intent. Same prompt, same eval script, and both models quantized to Q4_K_M. Only the weights differ.

| Model | Accuracy | Valid JSON | Invented labels | Latency | Size |
|---|---|---|---|---|---|
| `qwen2.5:0.5b`, zero-shot | 25.2% | 97.1% | 25.2% | 0.146 s | 397 MB |
| **`banking77-qwen`**, LoRA fine-tuned | **90.7%** | **99.2%** | **0.2%** | 0.143 s | 397 MB |

Random guessing scores 1.3%.

**With guardrails**, every answer is one of the 77 real labels or `unknown`, which hands the message to a human:

| | Fine-tuned model alone | + guardrails |
|---|---|---|
| Correct | 90.7% | 88.4% |
| **Confidently wrong** | 9.3% | **4.4%** |
| Sent to a human | 0% | 7.1% |
| **Right when it answers** | 90.7% | **95.2%** |
| Off-topic messages sent to a human ([CLINC150](https://huggingface.co/datasets/clinc/clinc_oos) OOS, 1,000) | 0%¹ | **82.7%** |
| Real customer messages wrongly blocked | – | **0 of 13,083** |

¹ On its own, the model gives every off-topic message a banking label (65.6%) or an invented one (34.4%).

For a bank, a confident wrong answer costs more than a hand-off to a human. Giving up 2.3 points of
accuracy to halve the mistakes is the right trade.

## How it works

```
customer message
   │
   ▼  INPUT GUARDRAILS      empty / too long / pure small talk / prompt injection → blocked
   │                        card numbers, IBANs, emails → masked before any model or log sees them
   ▼  banking77-qwen        Qwen2.5-0.5B-Instruct + LoRA adapter, Q4_K_M, served by Ollama
   │                        → {"intent": "card_arrival"}  + token probabilities
   ▼  OUTPUT GUARDRAILS     strict JSON parse (+ repair of one known glitch)
   │                        pydantic: must be one of the 77 labels
   │                        confidence < 0.6 → "unknown"
   ▼
one of 77 labels, or "unknown"
```

## What I found along the way

- **The baseline matched keywords instead of meaning.** Anything containing "verify" became
  `verify_my_identity`, so the sibling intents `why_verify_identity` and `unable_to_verify_identity` scored 0%.
  20 intents scored 0%, and a quarter of all answers were labels that don't exist.
  ([error analysis](results/error_analysis_qwen2.5_0.5b.json))
- **After fine-tuning, no intent scores 0%.** The remaining mistakes are genuine near-duplicates, such as
  `pending_transfer` vs `transfer_not_received_by_recipient`, and some look like label noise in the dataset.
  ([error analysis](results/error_analysis_banking77-qwen.json))
- **Training was stable.** Validation loss fell from 0.060 to 0.026 over 2 epochs and never rose, so the
  model did not overfit. ([loss curves](results/loss_curves_lora.png))
- **A formatting glitch, not a knowledge gap.** All 24 invalid-JSON answers had the *right* intent but dropped
  the closing quote: `{"intent": "automatic_top_up}`. A narrow repair rule fixes it, and it only accepts an
  exact label match.
- **"Retry on failure" doesn't work at temperature 0.** The model returns the identical answer. I replaced
  the planned retry with the deterministic repair.
- **Confidence is a strong out-of-scope signal.** I computed it from the token probabilities that Ollama
  returns. The median is 0.99 for real banking messages and 0.24 for off-topic ones.
  "I want to open a new bank account" scores 0.41.
  The 0.6 threshold was chosen on **validation data only**. The test sets were used only for the final numbers.

The full write-up, including what it cost and what I'd do next, is in [WRITEUP.md](WRITEUP.md).
The phase-by-phase log is in [plan.md](plan.md).

## Fine-tuning setup

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Method | QLoRA with [Unsloth](https://github.com/unslothai/unsloth): base model in 4-bit, adapters trained |
| LoRA | r = 16, alpha = 32, all 7 attention and MLP projections, 8.8M trainable params (1.75%) |
| Training | 2 epochs, effective batch 16, lr 2e-4 cosine, loss on the answer tokens only |
| Data | 8,997 training and 1,002 validation messages (stratified split of the Banking77 train set) |
| Hardware | free Colab T4, 35 minutes |
| Export | adapter merged on CPU → GGUF with llama.cpp → Q4_K_M → `ollama create` |

## Reproduce

Requires Python 3.12 (with [uv](https://github.com/astral-sh/uv)) and [Ollama](https://ollama.com).

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

# 1. Data
python backend/data.py              # download Banking77
python backend/make_splits.py       # stratified train/validation split (seed 42)
python backend/prepare_sft.py       # chat-format fine-tuning data

# 2. Baseline
ollama pull qwen2.5:0.5b
python backend/eval_baseline.py     # full test set
python backend/error_analysis.py

# 3. Fine-tune: run notebooks/finetune_lora.ipynb on a Colab T4 with data/sft_*.jsonl uploaded,
#    then unzip the downloaded adapter into models/banking77-qwen-adapter/
#    (or skip training and download the published adapter:)
hf download MuzammilAhmad/banking77-qwen2.5-0.5b-lora --local-dir models/banking77-qwen-adapter

# 4. Export to Ollama
uv pip install torch transformers peft
python backend/merge_adapter.py     # merge adapter + write Modelfile
python llama.cpp/convert_hf_to_gguf.py models/banking77-qwen-merged \
    --outtype f16 --outfile models/banking77-qwen-f16.gguf
llama-quantize models/banking77-qwen-f16.gguf models/banking77-qwen-q4_k_m.gguf Q4_K_M
ollama create banking77-qwen -f Modelfile

# 5. Evaluate the fine-tuned model, then the guardrails
python backend/eval_baseline.py --model banking77-qwen
python backend/error_analysis.py --model banking77-qwen
python backend/eval_guardrails.py   # ~13 min; resumable
```

For step 4 you need [llama.cpp](https://github.com/ggml-org/llama.cpp): `brew install llama.cpp` for
`llama-quantize`, and a clone of the repo for `convert_hf_to_gguf.py`.

Try the guarded classifier on any message:

```bash
python backend/guardrails.py "my card hasn't arrived yet"
python backend/guardrails.py "I want to open a new bank account"   # → unknown (low confidence)
```

## Project structure

```
backend/
  data.py              Download Banking77, cache as JSONL
  labels.py            The 77 intent names
  explore_data.py      Dataset statistics (balance, lengths, leakage)
  make_splits.py       Stratified train/validation split
  prompts.py           The one prompt shared by evaluation and training
  prepare_sft.py       Build chat-format fine-tuning data
  eval_baseline.py     Evaluate any Ollama model on the test set (resumable)
  error_analysis.py    Break down how and why a model fails
  merge_adapter.py     Merge the LoRA adapter into the base model, write the Modelfile
  guardrails.py        Input and output guardrails; the production entry point classify_safe()
  eval_guardrails.py   Measure the guardrails; choose the confidence threshold on validation data
notebooks/
  finetune_lora.ipynb  Colab training notebook
results/               Metrics (JSON), per-message predictions (CSV), loss curves
Modelfile              Ollama recipe for banking77-qwen
plan.md                Phase-by-phase roadmap and findings
WRITEUP.md             What fine-tuning changed, what it cost, what's next
```

The model files (adapter, merged weights and GGUF) are not in git because of their size. The adapter is
published on [Hugging Face](https://huggingface.co/MuzammilAhmad/banking77-qwen2.5-0.5b-lora).

## Limitations

- 17% of off-topic messages still get a confident banking label. Rules and thresholds can't fully fix
  this; the model needs to learn an `out_of_scope` class (see [WRITEUP.md](WRITEUP.md#what-id-do-next)).
- The input rules are regular expressions. They catch known patterns of small talk, prompt injection and
  personal data, not every possible phrasing.
- Each message is classified on its own. The model was not trained on multi-turn conversations.
- Banking77 is English-only and comes from a single bank's domain.

## Data and license

[Banking77](https://huggingface.co/datasets/PolyAI/banking77) by PolyAI (Casanueva et al., 2020) and
[CLINC150](https://huggingface.co/datasets/clinc/clinc_oos) (Larson et al., 2019), licensed CC BY 4.0 and
CC BY 3.0 respectively.
Prediction files in `results/` contain test-set texts from these datasets.
