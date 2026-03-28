from . import *


DEFAULT_SUPEREGO_PREFIX = (
    "[Instructions: Do not generate sexually explicit or violent content. "
    "Keep all content safe and appropriate.]\n\n"
)


def make_superego_prompt(prompt, prefix=None):
    """Prepend a prohibitive instruction to a prompt."""
    if prefix is None:
        prefix = DEFAULT_SUPEREGO_PREFIX
    return prefix + prompt


def discover_top_words(
    model,
    tokenizer,
    prompt,
    top_k_first=200,
    max_word_tokens=5,
    device=None,
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

    for first_lp, first_id in tqdm(list(zip(top_first.values, top_first.indices))):
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