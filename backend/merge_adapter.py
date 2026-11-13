"""Merge the LoRA adapter into the full-precision base model (Phase 6).

Ollama can't load a separate LoRA adapter for Qwen2, so we bake the adapter's
weight updates into the base weights and save a normal Hugging Face model.
A 0.5B model fits easily in a Mac's RAM, so this runs on the CPU in about a minute.

    python backend/merge_adapter.py

Input:  models/banking77-qwen-adapter/   (downloaded from Colab, Phase 5)
Output: models/banking77-qwen-merged/    (safetensors)
        Modelfile                        (points Ollama at the Q4_K_M GGUF below)

Then convert and import. Ollama 0.40 can't produce Q4_K_M from safetensors, and the
baseline is Q4_K_M, so we use llama.cpp to match it exactly:

    python llama.cpp/convert_hf_to_gguf.py models/banking77-qwen-merged \\
        --outtype f16 --outfile models/banking77-qwen-f16.gguf
    llama-quantize models/banking77-qwen-f16.gguf models/banking77-qwen-q4_k_m.gguf Q4_K_M
    ollama create banking77-qwen -f Modelfile
"""

import subprocess

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import PROJECT_ROOT

# The adapter was trained against Unsloth's 4-bit copy of this model. We merge into
# the original full-precision weights instead: same architecture, no 4-bit rounding.
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = PROJECT_ROOT / "models" / "banking77-qwen-adapter"
MERGED_DIR = PROJECT_ROOT / "models" / "banking77-qwen-merged"
GGUF_PATH = PROJECT_ROOT / "models" / "banking77-qwen-q4_k_m.gguf"
MODELFILE = PROJECT_ROOT / "Modelfile"
BASELINE_OLLAMA_MODEL = "qwen2.5:0.5b"


def write_modelfile() -> None:
    """Point Ollama at the quantized GGUF, with the baseline's exact chat template."""
    from prompts import SYSTEM_PROMPT

    # Copied from the installed baseline so prompt formatting is identical in the
    # before/after comparison: only the weights differ.
    template = subprocess.run(
        ["ollama", "show", BASELINE_OLLAMA_MODEL, "--template"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    MODELFILE.write_text(
        f"FROM ./{GGUF_PATH.relative_to(PROJECT_ROOT)}\n\n"
        f'TEMPLATE """{template}"""\n\n'
        # The default system prompt, used when a request sends none (e.g. `ollama run`).
        f'SYSTEM """{SYSTEM_PROMPT}"""\n'
    )
    print(f"Wrote {MODELFILE.relative_to(PROJECT_ROOT)}")


def main() -> None:
    print(f"Loading base model {BASE_MODEL} ...")
    # float32 for the merge math itself, so adding the LoRA update loses no precision.
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)

    print(f"Attaching adapter from {ADAPTER_DIR.relative_to(PROJECT_ROOT)} ...")
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))

    print("Merging adapter into base weights ...")
    merged = model.merge_and_unload()
    merged = merged.to(torch.float16)  # Ollama quantizes from this to Q4_K_M

    # The base tokenizer, not the adapter copy: Unsloth edits special tokens such as
    # the pad token, and the vocabulary is identical either way.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"Saved merged model to {MERGED_DIR.relative_to(PROJECT_ROOT)}/")

    # Smoke test in plain transformers, before Ollama and quantization get involved.
    from prompts import build_messages

    for text in ["my card hasn't arrived yet", "I was charged twice for the same thing"]:
        ids = tokenizer.apply_chat_template(
            build_messages(text), add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )["input_ids"]
        with torch.no_grad():
            out = merged.generate(ids, max_new_tokens=30, do_sample=False)
        reply = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"  {text!r} -> {reply.strip()}")

    write_modelfile()


if __name__ == "__main__":
    main()
