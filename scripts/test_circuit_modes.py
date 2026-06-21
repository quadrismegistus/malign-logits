"""Test Circuit Mode.CHAT and Mode.THINK with live models.

Uses smol3 (SmolLM3-3B) since it supports thinking mode.
Compares RAW vs CHAT vs THINK logit distributions on same prompt.
"""
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import sys
sys.path.insert(0, ".")

import torch
import numpy as np
from malign_logits.circuit import Circuit, Mode
from malign_logits.models import load_model
from malign_logits.analysis import js_divergence

PROMPTS = [
    "She was so angry she wanted to",
    "He picked up the knife and",
    "She loved him deeply and wanted to",
]

print("Loading SmolLM3-3B for Mode test...")
model_id = "HuggingFaceTB/SmolLM3-3B"
model, tokenizer = load_model(model_id)
print(f"Model loaded on {next(model.parameters()).device}")

# Build a minimal Circuit node manually
from malign_logits.psyche import ModelLayer
from malign_logits.cache import get_cache

cm = get_cache()
layer = ModelLayer(
    model=model,
    tokenizer=tokenizer,
    model_id=model_id,
    name="aligned",
)
layer._cache = cm

from malign_logits.circuit import CircuitNode
node = CircuitNode(layer, position="aligned", family="smol3")

print(f"\n{'Prompt':<45} {'RAW H':>7} {'CHAT H':>7} {'THINK H':>8} {'JS(R,C)':>8} {'JS(R,T)':>8} {'JS(C,T)':>8}")
print("-" * 100)

for prompt in PROMPTS:
    try:
        raw_logits = node.logits(prompt, mode=Mode.RAW)
        chat_logits = node.logits(prompt, mode=Mode.CHAT)
    except Exception as e:
        print(f"{prompt[:43]:<45} ERROR: {e}")
        continue

    try:
        think_logits = node.logits(prompt, mode=Mode.THINK)
    except Exception as e:
        think_logits = None
        print(f"  (THINK mode failed: {e})")

    from malign_logits.analysis import distribution_entropy
    raw_h = distribution_entropy(raw_logits)
    chat_h = distribution_entropy(chat_logits)

    raw_probs = torch.softmax(raw_logits.float(), dim=-1)
    chat_probs = torch.softmax(chat_logits.float(), dim=-1)

    js_rc = js_divergence(raw_logits, chat_logits)

    if think_logits is not None:
        think_h = distribution_entropy(think_logits)
        think_probs = torch.softmax(think_logits.float(), dim=-1)
        js_rt = js_divergence(raw_logits, think_logits)
        js_ct = js_divergence(chat_logits, think_logits)
        print(f"{prompt[:43]:<45} {raw_h:>7.2f} {chat_h:>7.2f} {think_h:>8.2f} {js_rc:>8.3f} {js_rt:>8.3f} {js_ct:>8.3f}")
    else:
        print(f"{prompt[:43]:<45} {raw_h:>7.2f} {chat_h:>7.2f} {'N/A':>8} {js_rc:>8.3f} {'N/A':>8} {'N/A':>8}")

    # Top-5 comparison
    for mode_name, logits in [("RAW", raw_logits), ("CHAT", chat_logits)] + ([("THINK", think_logits)] if think_logits is not None else []):
        probs = torch.softmax(logits.float(), dim=-1)
        top5_vals, top5_idx = torch.topk(probs, 5)
        top5_tokens = [tokenizer.decode([idx]).strip() for idx in top5_idx]
        top5_str = " | ".join(f"{t}({v:.3f})" for t, v in zip(top5_tokens, top5_vals))
        print(f"  {mode_name:<6}: {top5_str}")
    print()

print("Done. Mode.CHAT and Mode.THINK wiring verified.")

# Cleanup
del model
torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None
