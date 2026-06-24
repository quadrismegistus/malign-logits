from . import *


def discover_top_words(
    model,
    tokenizer,
    prompt,
    top_k_first=200,
    max_word_tokens=5,
    device=None,
    progress_callback=None,
):
    """
    Discover the model's most probable first words following a prompt.

    Strategy:
        1. Take the top-k most probable first tokens.
        2. For each, greedily continue until a word boundary.
        3. Accumulate probability mass per completed word.
        4. Filter to alphabetic words only.

    Args:
        model: A HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompt: Text string to complete.
        top_k_first: How many first-token paths to explore.
        max_word_tokens: Maximum subword tokens per word.
        device: Torch device override.

    Returns:
        dict mapping word (str) -> probability (float), sorted descending.
    """
    if device is None:
        device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        first_logits = model(input_ids).logits[0, -1, :]

    first_log_probs = torch.log_softmax(first_logits.float(), dim=-1)
    top_first = torch.topk(first_log_probs, top_k_first)

    word_scores = {}

    pairs = list(zip(top_first.values, top_first.indices))
    for idx, (first_lp, first_id) in enumerate(tqdm(pairs)):
        if progress_callback and idx % 10 == 0:
            progress_callback(idx, len(pairs))
        current_ids = torch.cat(
            [input_ids, first_id.unsqueeze(0).unsqueeze(0).to(device)], dim=-1
        )
        cumulative_lp = first_lp.item()

        for step in range(max_word_tokens - 1):
            generated_text = tokenizer.decode(
                current_ids[0][input_ids.shape[1]:], skip_special_tokens=True
            ).strip()

            # Stop if we've passed a word boundary
            if " " in generated_text and len(generated_text.split()[0]) > 0:
                break

            with torch.no_grad():
                next_logits = model(current_ids).logits[0, -1, :]
            next_lp = torch.log_softmax(next_logits.float(), dim=-1)
            best_next = next_lp.argmax()
            cumulative_lp += next_lp[best_next].item()
            current_ids = torch.cat(
                [current_ids, best_next.unsqueeze(0).unsqueeze(0).to(device)],
                dim=-1,
            )

        # Extract completed first word
        generated_text = tokenizer.decode(
            current_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        word = generated_text.split()[0] if generated_text else ""
        word = word.strip(".,;:!?\"'()[]{}—-–")

        if word and word.isalpha():
            prob = math.exp(cumulative_lp)
            word_scores[word] = word_scores.get(word, 0) + prob

    # Normalize
    total = sum(word_scores.values())
    if total > 0:
        word_scores = {w: p / total for w, p in word_scores.items()}

    return dict(sorted(word_scores.items(), key=lambda x: -x[1]))


def get_word_logprobs(model, tokenizer, prompt, candidate_words, device=None):
    """
    Compute exact log-probabilities for specific candidate words.

    For each candidate word, computes the joint probability of its tokens
    appearing after the prompt. Useful when you want precise comparisons
    over a controlled vocabulary.

    Args:
        model: A HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompt: Text string preceding the word.
        candidate_words: List of words to score.
        device: Torch device override.

    Returns:
        dict mapping word (str) -> probability (float), sorted descending.
    """
    if device is None:
        device = next(model.parameters()).device

    word_logprobs = {}

    for word in candidate_words:
        full_text = prompt + " " + word
        full_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_len = prompt_ids.shape[1]

        with torch.no_grad():
            outputs = model(full_ids)
            log_probs = torch.log_softmax(outputs.logits[0].float(), dim=-1)

        word_lp = 0.0
        for pos in range(prompt_len, full_ids.shape[1]):
            token_id = full_ids[0, pos]
            word_lp += log_probs[pos - 1, token_id].item()

        word_logprobs[word] = word_lp

    # Convert log-probs to normalized probabilities
    max_lp = max(word_logprobs.values())
    probs = {w: math.exp(lp - max_lp) for w, lp in word_logprobs.items()}
    total = sum(probs.values())
    probs = {w: p / total for w, p in probs.items()}

    return dict(sorted(probs.items(), key=lambda x: -x[1]))


def score_words_from_logits(logits, tokenizer, candidate_words):
    """Score candidate words using pre-computed logits (single forward pass).

    For single-token words, reads probability directly from the logits.
    Multi-token words are approximated using the first token's probability.

    Much faster than get_word_logprobs (which does one forward pass per word).

    Args:
        logits: Raw logits tensor at the last position (vocab_size,).
        tokenizer: Tokenizer for encoding words.
        candidate_words: List of words to score.

    Returns:
        dict mapping word (str) -> probability (float), sorted descending.
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    word_scores = {}

    for word in candidate_words:
        token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if not token_ids:
            continue
        tid = token_ids[0]
        if len(token_ids) > 1 and not tokenizer.decode([tid]).strip():
            tid = token_ids[1]
        word_scores[word] = log_probs[tid].item()

    if not word_scores:
        return {}

    max_lp = max(word_scores.values())
    probs = {w: math.exp(lp - max_lp) for w, lp in word_scores.items()}
    total = sum(probs.values())
    probs = {w: p / total for w, p in probs.items()}

    return dict(sorted(probs.items(), key=lambda x: -x[1]))


def hybrid_word_probs(beam_words, logits, tokenizer):
    """Combine beam word list with exact logit probabilities.

    Uses raw logit P(token) for single-token words (exact), falls back
    to beam path probability for multi-token words. No model loading needed.

    Args:
        beam_words: dict from beam_word_probs() — word → beam probability
        logits: numpy array of raw logits at position -1
        tokenizer: tokenizer for encoding words

    Returns:
        dict mapping word → probability, renormalized.
    """
    import numpy as np
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()

    hybrid = {}
    for word in beam_words:
        tids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(tids) == 1:
            hybrid[word] = float(probs[tids[0]])
        else:
            hybrid[word] = beam_words[word]

    total = sum(hybrid.values())
    if total > 0:
        hybrid = {w: p / total for w, p in hybrid.items()}

    return dict(sorted(hybrid.items(), key=lambda x: -x[1]))


def beam_word_probs(model, tokenizer, prompt, n_beams=1000, depth=3, device=None):
    """Word probabilities via beam search — accurate for multi-token words.

    Runs beam search at the given depth, aggregates sequence probabilities
    by first decoded word. Faster and more accurate than discover_top_words
    for multi-token words (4s vs 15s per prompt on 1B MPS).

    The probabilities are normalized across the beam set (not the full vocab),
    so they're correct for relative comparison across layers but don't match
    raw logit softmax values.

    Args:
        model: A HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompt: Text string to complete.
        n_beams: Number of beams (more = more words found, slower).
        depth: Token depth (2 captures most words, 3 for longer words).
        device: Torch device override.

    Returns:
        dict mapping word (str) -> probability (float), sorted descending.
    """
    if device is None:
        device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    out = model.generate(
        input_ids,
        num_beams=n_beams,
        num_return_sequences=n_beams,
        max_new_tokens=depth,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=pad_token_id,
    )

    scores = out.sequences_scores.float().cpu().numpy()
    probs = math.e ** scores
    total = probs.sum()
    if total > 0:
        probs = probs / total

    word_probs = {}
    prompt_len = input_ids.shape[1]
    for i, seq in enumerate(out.sequences):
        text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
        word = text.split()[0].strip(".,;:!?\"'()[]{}—-–") if text.split() else ""
        if not word or not word.isalpha() or len(word) < 2:
            continue
        word_probs[word] = word_probs.get(word, 0) + float(probs[i])

    return dict(sorted(word_probs.items(), key=lambda x: -x[1]))
