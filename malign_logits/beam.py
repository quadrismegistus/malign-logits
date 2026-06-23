"""
beam.py — Beam search storyline extraction.

Replaces tree exploration for storyline analysis. Uses HuggingFace
generate() with batched beam search — 100 storylines in <1s on CUDA,
~3s on MPS.

    from malign_logits.beam import beam_storylines, annotate_beams

    # Base model's top 100 storylines
    stories = beam_storylines("allenai/OLMo-2-0425-1B", "anger", n=100)

    # Annotate: run same beams through aligned model
    annotated = annotate_beams("allenai/OLMo-2-0425-1B", "anger", n=100)
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Storyline:
    """One beam search result — a complete storyline."""
    text: str
    tokens: List[int]
    token_texts: List[str]
    path_prob: float
    log_prob: float
    # Per-annotator metrics (filled by annotate_beams)
    annotations: dict = field(default_factory=dict)


def beam_storylines(model_id: str, prompt: str, n: int = 100,
                    max_tokens: int = 10) -> List[Storyline]:
    """Extract top-N storylines by beam search.

    Returns list of Storyline objects sorted by path probability.
    ~1s for 100 storylines on CUDA, ~3s on MPS.
    """
    from .models import load_model
    from .probe import _resolve_prompt

    prompt_text = _resolve_prompt(prompt)
    model, tokenizer = load_model(model_id)
    device = next(model.parameters()).device

    ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    prompt_len = ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            ids,
            num_beams=n,
            num_return_sequences=n,
            max_new_tokens=max_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )

    storylines = []
    for i in range(len(out.sequences)):
        seq = out.sequences[i]
        new_tokens = seq[prompt_len:]
        token_ids = new_tokens.tolist()
        token_texts = [tokenizer.decode([tid]).strip() for tid in token_ids]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        log_prob = out.sequences_scores[i].item()

        storylines.append(Storyline(
            text=text,
            tokens=token_ids,
            token_texts=token_texts,
            path_prob=float(np.exp(log_prob)),
            log_prob=log_prob,
        ))

    del model
    import gc
    gc.collect()
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    return storylines


def annotate_beams(model_id: str, prompt: str, n: int = 100,
                   max_tokens: int = 10,
                   annotators: list = None) -> List[Storyline]:
    """Beam search + teacher-forcing through aligned models.

    For each storyline from the base model, teacher-forces through
    each annotator to compute per-token resistance.

    Returns Storyline objects with annotations dict:
        {annotator_short: {
            'log_prob': float,       # total log-prob under annotator
            'path_prob': float,      # exp(log_prob)
            'token_probs': [float],  # per-token P under annotator
            'token_resist': [float], # per-token self_resist
            'total_resist': float,   # sum of token_resist
            'mean_resist': float,    # mean of token_resist
        }}
    """
    from .models import load_model
    from .probe import _resolve_prompt
    from .registry import Registry
    from scipy.special import softmax
    import gc

    prompt_text = _resolve_prompt(prompt)
    reg = Registry()

    # Step 1: get base storylines
    storylines = beam_storylines(model_id, prompt, n=n, max_tokens=max_tokens)

    # Step 2: determine annotators
    if annotators is None:
        base_id = reg.base_of(model_id) or model_id
        annotators = reg.variants_of(base_id)
        if model_id != base_id:
            annotators = [base_id] + annotators
        annotators = [m for m in annotators if m != model_id]

    if not annotators:
        return storylines

    # Get base tokenizer for encoding
    from transformers import AutoTokenizer
    base_tok = AutoTokenizer.from_pretrained(model_id)

    # Step 3: for each annotator, teacher-force all storylines
    for ann_id in annotators:
        ann_short = ann_id.split("/")[-1].replace("-", "_")[:20]
        print(f"  Annotating with {ann_id.split('/')[-1]}...", end="", flush=True)

        try:
            ann_model, ann_tok = load_model(ann_id)
            ann_device = next(ann_model.parameters()).device

            for story in storylines:
                # Build full sequence: prompt + storyline tokens
                full_text = prompt_text + " " + story.text
                full_ids = base_tok.encode(full_text, return_tensors="pt").to(ann_device)
                prompt_ids = base_tok.encode(prompt_text, return_tensors="pt").to(ann_device)
                prompt_len = prompt_ids.shape[1]

                with torch.no_grad():
                    out = ann_model(full_ids)

                logits = out.logits[0].float().cpu().numpy()

                # Compute per-token probability and resistance
                token_probs = []
                token_resist = []
                total_log_prob = 0.0

                for pos in range(prompt_len - 1, min(full_ids.shape[1] - 1, prompt_len + len(story.tokens) - 1)):
                    token_idx = pos + 1
                    if token_idx >= full_ids.shape[1]:
                        break
                    target_id = full_ids[0, token_idx].item()
                    probs = softmax(logits[pos])

                    p_ann = float(probs[target_id])
                    token_probs.append(p_ann)
                    total_log_prob += float(np.log(max(p_ann, 1e-10)))

                    # Base prob for this token in the storyline
                    story_pos = pos - (prompt_len - 1)
                    if story_pos < len(story.tokens):
                        # Approximate base prob from overall path prob
                        p_base = story.path_prob ** (1.0 / len(story.tokens))
                        resist = -float(np.log2(max(p_ann, 1e-10))) - (-float(np.log2(max(p_base, 1e-10))))
                    else:
                        resist = 0.0
                    token_resist.append(resist)

                story.annotations[ann_short] = {
                    "log_prob": total_log_prob,
                    "path_prob": float(np.exp(total_log_prob)),
                    "token_probs": token_probs,
                    "token_resist": token_resist,
                    "total_resist": sum(token_resist),
                    "mean_resist": np.mean(token_resist) if token_resist else 0.0,
                }

            del ann_model
            gc.collect()
            try:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass
            print(f" {len(storylines)} storylines")

        except Exception as e:
            print(f" FAILED: {str(e)[:60]}")

    return storylines
