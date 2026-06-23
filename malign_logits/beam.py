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

    # Build batched input: encode all storylines, pad to same length
    from transformers import AutoTokenizer
    base_tok = AutoTokenizer.from_pretrained(model_id)
    if base_tok.pad_token_id is None:
        base_tok.pad_token_id = base_tok.eos_token_id

    prompt_ids = base_tok.encode(prompt_text)
    prompt_len = len(prompt_ids)

    all_ids = []
    all_lengths = []
    for story in storylines:
        full_text = prompt_text + " " + story.text
        ids = base_tok.encode(full_text)
        all_ids.append(ids)
        all_lengths.append(len(ids))

    max_len = max(all_lengths)
    padded = torch.full((len(storylines), max_len), base_tok.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(storylines), max_len, dtype=torch.long)
    for i, ids in enumerate(all_ids):
        padded[i, :len(ids)] = torch.tensor(ids)
        attention_mask[i, :len(ids)] = 1

    def _batched_teacher_force(model, device, batch_size=16):
        """Run batched forward pass, return per-token probs for each storyline."""
        all_probs = []
        for start in range(0, len(storylines), batch_size):
            end = min(start + batch_size, len(storylines))
            batch_ids = padded[start:end].to(device)
            batch_mask = attention_mask[start:end].to(device)

            with torch.no_grad():
                out = model(batch_ids, attention_mask=batch_mask)

            logits = out.logits.float().cpu()

            for bi in range(end - start):
                si = start + bi
                seq_len = all_lengths[si]
                token_probs = []
                for pos in range(prompt_len - 1, seq_len - 1):
                    target_id = padded[si, pos + 1].item()
                    p = torch.softmax(logits[bi, pos], dim=-1)
                    token_probs.append(float(p[target_id]))
                all_probs.append(token_probs)

        return all_probs

    # Step 2b: batched base model teacher-forcing
    print(f"  Base teacher-force ({len(storylines)} stories)...", end="", flush=True)
    base_model, _ = load_model(model_id)
    base_device = next(base_model.parameters()).device
    base_probs = _batched_teacher_force(base_model, base_device)
    for i, story in enumerate(storylines):
        story.base_token_probs = base_probs[i]
    del base_model
    gc.collect()
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    print(f" done")

    # Step 3: batched annotator teacher-forcing
    for ann_id in annotators:
        ann_short = ann_id.split("/")[-1].replace("-", "_")[:20]
        print(f"  {ann_id.split('/')[-1]} ({len(storylines)} stories)...", end="", flush=True)

        try:
            ann_model, _ = load_model(ann_id)
            ann_device = next(ann_model.parameters()).device
            ann_probs = _batched_teacher_force(ann_model, ann_device)

            for i, story in enumerate(storylines):
                ann_tp = ann_probs[i]
                base_tp = story.base_token_probs

                token_resist = []
                for j in range(min(len(ann_tp), len(base_tp))):
                    resist = (
                        -float(np.log2(max(ann_tp[j], 1e-10)))
                        - (-float(np.log2(max(base_tp[j], 1e-10))))
                    )
                    token_resist.append(resist)

                ann_mean = float(np.exp(np.mean(
                    [np.log(max(p, 1e-10)) for p in ann_tp]
                ))) if ann_tp else 0.0
                base_mean = float(np.exp(np.mean(
                    [np.log(max(p, 1e-10)) for p in base_tp]
                ))) if base_tp else 0.0

                story.annotations[ann_short] = {
                    "token_probs": ann_tp,
                    "token_resist": token_resist,
                    "mean_prob": ann_mean,
                    "base_mean_prob": base_mean,
                    "total_resist": sum(token_resist),
                    "mean_resist": float(np.mean(token_resist)) if token_resist else 0.0,
                }

            del ann_model
            gc.collect()
            try:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass
            print(f" done")

        except Exception as e:
            print(f" FAILED: {str(e)[:60]}")

    # Cache results
    from .probe import _get_cache
    cache = _get_cache()
    cache_key = {"model": model_id, "prompt": prompt_text,
                 "n_beams": n, "max_tokens": max_tokens,
                 "type": "beam_annotated_v1"}
    cache_data = [
        {
            "text": s.text,
            "tokens": s.tokens,
            "token_texts": s.token_texts,
            "path_prob": s.path_prob,
            "log_prob": s.log_prob,
            "base_token_probs": getattr(s, "base_token_probs", []),
            "annotations": s.annotations,
        }
        for s in storylines
    ]
    cache.set_derived(cache_key, cache_data)

    return storylines


def load_cached_beams(model_id: str, prompt: str, n: int = 100,
                      max_tokens: int = 10) -> Optional[List[Storyline]]:
    """Load cached beam results if available."""
    from .probe import _get_cache, _resolve_prompt

    prompt_text = _resolve_prompt(prompt)
    cache = _get_cache()
    cache_key = {"model": model_id, "prompt": prompt_text,
                 "n_beams": n, "max_tokens": max_tokens,
                 "type": "beam_annotated_v1"}
    data = cache.get_derived(cache_key)
    if data is None:
        return None

    storylines = []
    for d in data:
        s = Storyline(
            text=d["text"],
            tokens=d["tokens"],
            token_texts=d["token_texts"],
            path_prob=d["path_prob"],
            log_prob=d["log_prob"],
            annotations=d.get("annotations", {}),
        )
        s.base_token_probs = d.get("base_token_probs", [])
        storylines.append(s)
    return storylines
