# What 35 minutes of LoRA fine-tuning bought a 0.5B model

**The question:** can a tiny, local language model sort bank customer messages into 77 intents well
enough to be useful, and what does it take to get there?

**The answer:** zero-shot, it can't (25% accuracy). After 35 minutes of LoRA fine-tuning on a free GPU
it reaches 91%, at the same size and speed. With guardrails on top, it is right 95% of the time when it
answers, and hands the rest to a human.

## The starting point

Banking77 is a set of 13,083 real customer messages, each labelled with one of 77 intents such as
`card_arrival`, `pending_transfer` or `why_verify_identity`. Many intents are near-duplicates, which
makes it a hard task for a small model.

I gave `Qwen2.5-0.5B-Instruct` the full list of 77 labels in the prompt and asked for JSON such as
`{"intent": "card_arrival"}`. It scored **25.2%**. The error analysis showed why: it matched words in
the message to words in the label names. Anything containing "verify" became `verify_my_identity`, so its
siblings scored 0%. 20 intents scored 0%, and a quarter of its answers were labels it made up.

A better prompt doesn't fix this. A 0.5B model doesn't have the capacity to read 77 subtle definitions
and apply them, but it can learn the distinctions from examples.

## What fine-tuning changed

I trained a LoRA adapter on 8,997 labelled messages. LoRA is a method that freezes the original weights
and trains small extra matrices on top, here 8.8M parameters, or 1.75% of the model. The loss was computed
on the answer tokens only, so the model learned to answer rather than to repeat the prompt. I kept the
prompt, the eval script and the quantization (Q4_K_M) exactly the same as for the baseline, so the only
change was the weights.

| | Before | After |
|---|---|---|
| Accuracy | 25.2% | **90.7%** |
| Invented labels | 25.2% | **0.2%** |
| Intents scoring 0% | 20 | **0** |
| Latency | 0.146 s | 0.143 s |

The mistakes that remain are the kind a human would also make. `pending_transfer` and
`transfer_not_received_by_recipient` describe the same situation from two angles, and some test labels
look like noise in the dataset.

## What it cost

- **Compute:** one free Colab T4 session, 35 minutes of training.
- **Size and speed:** nothing. The adapter is 35 MB, and the merged and quantized model is 397 MB, the
  same as the base model. Latency is unchanged.
- **Time:** most of the effort went into everything *around* training, not the training itself:
  - A library update removed a helper I needed, so I wrote my own collator to mask the prompt tokens.
  - Exporting to GGUF inside Colab ran out of memory and crashed the runtime, losing the first trained model.
    The fix: download only the 35 MB adapter and do the merge and export on the laptop.
  - Ollama couldn't produce the same quantization as the baseline, so I used llama.cpp to keep the
    comparison fair.

## Making it safe to use

A classifier that is right 91% of the time is still confidently wrong 9% of the time, and it will give
*every* message a banking label, including "what's the weather in London?". I added two layers of checks.

- **Before the model:** block empty, overlong, small-talk and prompt-injection messages. Mask card numbers,
  IBANs and emails, so they never reach the model or the logs.
- **After the model:** accept only an exact one of the 77 labels. Repair one known formatting glitch, where
  the model drops the closing quote but has the right label. Answer `unknown` when the model's confidence
  (the probability of its reply, from the token log-probabilities) is below 0.6.

I chose the 0.6 threshold on validation data only and measured on untouched test sets:

- Confident mistakes fell from 9.3% to **4.4%**, while 7.1% of messages went to a human.
- **82.7%** of off-topic messages were caught, using 1,000 out-of-scope messages from CLINC150.
- **0 of 13,083** real customer messages were blocked by the input rules.

Two things changed from the original plan once I measured them. "Retry on invalid output" is useless at
temperature 0, because the model gives the same answer again. And the small-talk rule had to match whole
messages only, because 190 real questions start with "Hi" or "Hello".

## What I'd do next

1. **Teach the model to say "out of scope".** Add an `out_of_scope` class to training, using the CLINC150
   out-of-scope training data. This should catch most of the 17% of off-topic messages that still get a
   label, without relying on a threshold.
2. **Remove the label list from the prompt.** After fine-tuning, the model knows the labels, and the list is
   441 of the ~474 tokens in every request. Training without it would cut the work per request by more
   than 90%. That matters most on hardware without prompt caching.
3. **Find the source of the formatting glitch.** Compare the merged model before and after quantization,
   and against a merge into the 4-bit base the adapter was trained on.
4. **Compare against a large hosted model** to see how close 0.5B gets to the ceiling, and at what cost per
   message.
5. **Publish the adapter** on the Hugging Face Hub so the results can be reproduced without retraining.
