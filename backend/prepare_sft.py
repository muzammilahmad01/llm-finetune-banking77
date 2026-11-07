"""Phase 4: build supervised fine-tuning (SFT) data in chat format.

Run:  python backend/prepare_sft.py

Reads   data/train_split.jsonl, data/val_split.jsonl   (from make_splits.py)
Writes  data/sft_train.jsonl, data/sft_val.jsonl

Each output line is one complete conversation the model learns from:
    {"messages": [system (77 labels), user (customer message), assistant ({"intent": ...})]}
This "messages" format is what TRL's SFTTrainer and Unsloth expect.
The test split is NOT converted: it is never used for training.
"""

import json
import statistics

from data import DATA_DIR, load_split  # (importing data also points HF_HOME at .cache/)
from eval_baseline import parse_response
from labels import LABELS
from prompts import build_messages, format_answer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # the Hugging Face original of Ollama's qwen2.5:0.5b


def render_chat(messages: list[dict]) -> str:
    """Qwen2.5's chat template: how a conversation becomes one string of text.

    The model never sees "roles"; it sees special tokens like <|im_start|> that
    mark who is speaking. Training and inference must use this same format.
    (On Colab, tokenizer.apply_chat_template() does this for us.)
    """
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)


def load_tokenizer():
    """Download only Qwen's tokenizer file (~7 MB), not the model weights."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    return Tokenizer.from_file(hf_hub_download(BASE_MODEL, "tokenizer.json"))


def to_conversation(row: dict) -> dict:
    answer = format_answer(LABELS[row["label"]])
    return {"messages": build_messages(row["text"]) + [{"role": "assistant", "content": answer}]}


def main() -> None:
    tokenizer = load_tokenizer()
    n_tokens = lambda s: len(tokenizer.encode(s, add_special_tokens=False).ids)

    for split in ("train_split", "val_split"):
        rows = load_split(split)
        convs = [to_conversation(r) for r in rows]

        # Sanity check: every target answer must score as correct under our own
        # eval parser. If not, we'd be training the model to fail our eval.
        for row, conv in zip(rows, convs):
            pred, json_valid, in_vocab = parse_response(conv["messages"][-1]["content"])
            assert json_valid and in_vocab and pred == LABELS[row["label"]], conv

        out_name = "sft_train" if split == "train_split" else "sft_val"
        path = DATA_DIR / f"{out_name}.jsonl"
        with path.open("w") as f:
            for conv in convs:
                f.write(json.dumps(conv) + "\n")

        # Token statistics on the fully rendered conversation.
        full = sorted(n_tokens(render_chat(c["messages"])) for c in convs)
        answer = [n_tokens(c["messages"][-1]["content"]) for c in convs]
        print(f"{out_name}: {len(convs):,} conversations -> data/{out_name}.jsonl")
        print(f"  tokens per conversation: min {full[0]}, mean {statistics.mean(full):.0f}, "
              f"p99 {full[int(0.99 * (len(full) - 1))]}, max {full[-1]}")
        print(f"  tokens in the answer (what the loss is computed on): "
              f"mean {statistics.mean(answer):.1f}, max {max(answer)}")

    # Break the prompt down: how much is fixed instructions vs. the customer's message?
    system_tokens = n_tokens(render_chat(build_messages("")[:1]))
    print(f"\nshared system prompt (77 labels): {system_tokens} tokens in every example")
    # Round the longest example up to a multiple of 64 for max_seq_length (GPU-friendly).
    longest = max(n_tokens(render_chat(json.loads(l)["messages"]))
                  for l in (DATA_DIR / "sft_train.jsonl").open())
    print(f"recommended max_seq_length for Phase 5: {-(-longest // 64) * 64} "
          f"(longest example is {longest} tokens; nothing gets cut off)")

    print("\nExample training conversation:")
    example = json.loads((DATA_DIR / "sft_train.jsonl").open().readline())
    for m in example["messages"]:
        content = m["content"] if m["role"] != "system" else m["content"][:110] + " ... (77 labels) ..."
        print(f"  [{m['role']}] {content!r}")


if __name__ == "__main__":
    main()
