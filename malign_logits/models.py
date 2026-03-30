from . import *


def _load_tokenizer(model_name, revision=None, cache_dir=None):
    """Load tokenizer for a HuggingFace model."""
    from transformers import AutoTokenizer
    kwargs = {}
    if revision:
        kwargs["revision"] = revision
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return AutoTokenizer.from_pretrained(model_name, **kwargs)


def _load_causal_lm(model_name, quantization_config, device_map, dtype,
                     revision=None, cache_dir=None):
    from transformers import AutoModelForCausalLM
    kwargs = {}
    if revision:
        kwargs["revision"] = revision
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        dtype=dtype,
        **kwargs,
    )


def _platform_kwargs():
    """Return device_map, dtype, quantization_config for the current platform."""
    is_mac = platform.system() == "Darwin"
    if is_mac:
        return {
            "device_map": "mps",
            "dtype": torch.float16,
            "quantization_config": None,
        }
    else:
        print("Detected PC/Linux - using BitsAndBytes 4-bit quantization")
        from transformers import BitsAndBytesConfig
        return {
            "device_map": "auto",
            "dtype": None,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            ),
        }


def load_model(model_name, revision=None, cache_dir=None):
    """Load a single model and its tokenizer.

    Args:
        model_name: HuggingFace model ID.
        revision: Branch/tag/commit (e.g. "step1000").
        cache_dir: Custom HuggingFace cache directory.

    Returns:
        (model, tokenizer)
    """
    kwargs = _platform_kwargs()
    tokenizer = _load_tokenizer(model_name, revision=revision, cache_dir=cache_dir)
    label = f"{model_name}@{revision}" if revision else model_name
    print(f"Loading {label}...")
    model = _load_causal_lm(
        model_name,
        kwargs["quantization_config"],
        kwargs["device_map"],
        kwargs["dtype"],
        revision=revision,
        cache_dir=cache_dir,
    )
    return model, tokenizer


def load_models(
    base_name=BASE_MODEL_NAME,
    sft_name=SFT_MODEL_NAME,
    dpo_name=DPO_MODEL_NAME,
):
    """Load base, SFT, and DPO models (3-layer topology).

    All models share a single tokenizer (OLMo uses the same vocab
    across all checkpoints).

    Returns:
        (base_model, sft_model, dpo_model, tokenizer)
    """
    kwargs = _platform_kwargs()
    tokenizer = _load_tokenizer(base_name)

    print(f"Loading base model ({base_name})...")
    base_model = _load_causal_lm(base_name, **kwargs)

    print(f"Loading SFT model ({sft_name})...")
    sft_model = _load_causal_lm(sft_name, **kwargs)

    print(f"Loading DPO model ({dpo_name})...")
    dpo_model = _load_causal_lm(dpo_name, **kwargs)

    print("All models loaded.")
    return base_model, sft_model, dpo_model, tokenizer


def load_four_models(
    base_name=BASE_MODEL_NAME,
    sft_name=SFT_MODEL_NAME,
    dpo_name=DPO_MODEL_NAME,
    instruct_name=INSTRUCT_MODEL_NAME,
):
    """Load base, SFT, DPO, and RLVR/Instruct models (4-layer topology).

    Returns:
        (base_model, sft_model, dpo_model, instruct_model, tokenizer)
    """
    kwargs = _platform_kwargs()
    tokenizer = _load_tokenizer(base_name)

    print(f"Loading base model ({base_name})...")
    base_model = _load_causal_lm(base_name, **kwargs)

    print(f"Loading SFT model ({sft_name})...")
    sft_model = _load_causal_lm(sft_name, **kwargs)

    print(f"Loading DPO model ({dpo_name})...")
    dpo_model = _load_causal_lm(dpo_name, **kwargs)

    print(f"Loading RLVR/Instruct model ({instruct_name})...")
    instruct_model = _load_causal_lm(instruct_name, **kwargs)

    print("All models loaded.")
    return base_model, sft_model, dpo_model, instruct_model, tokenizer


def get_base_logits(model, tokenizer, prompt, device=None):
    """Get raw logits from a model for a prompt (for displacement/overdetermination)."""
    if device is None:
        device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :].cpu()
    return logits


def sequence_perplexity(model, tokenizer, prompt, device=None):
    """Compute sequence perplexity of a prompt under the model.

    Teacher-forced forward pass: for each token position, compute the
    negative log-probability of the actual next token. Returns the
    exponentiated mean (perplexity).

    Args:
        model: HuggingFace causal LM.
        tokenizer: Shared tokenizer.
        prompt: Input text.
        device: Override device.

    Returns:
        float: Perplexity (lower = more expected).
    """
    if device is None:
        device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    seq_len = input_ids.shape[1]
    if seq_len < 2:
        return float("nan")
    with torch.no_grad():
        logits = model(input_ids).logits  # (1, seq_len, vocab_size)
    # Shift: predict token t+1 from position t
    shift_logits = logits[0, :-1, :].float()  # (seq_len-1, vocab_size)
    shift_labels = input_ids[0, 1:]           # (seq_len-1,)
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
    mean_nll = -token_log_probs.mean().item()
    return math.exp(mean_nll)


def logit_lens(model, tokenizer, prompt, device=None):
    """Project each layer's hidden state to vocabulary space (logit lens).

    Single forward pass. For each of the model's hidden layers, applies
    the final layer norm and lm_head to produce a probability distribution,
    showing how the model's prediction evolves through the network.

    Returns:
        List of (vocab_size,) tensors, one per layer (layer 0 = embedding,
        layer 1..N = transformer layers). Length = num_hidden_layers + 1.
    """
    if device is None:
        device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden_dim)

        # Get norm and lm_head for projection
        norm = model.model.norm
        lm_head = model.lm_head

        layer_logits = []
        for hidden in hidden_states:
            normed = norm(hidden)
            logits = lm_head(normed)[0, -1, :].cpu()
            layer_logits.append(logits)

    return layer_logits


def logit_lens_words(model, tokenizer, prompt, words=None, top_k=5, device=None):
    """Track word probabilities at each network layer.

    Includes top-k predictions at each layer plus any explicitly
    requested words (ensuring tracked words are always visible even
    when they're not in the top-k).

    Args:
        model: HuggingFace causal LM.
        tokenizer: Shared tokenizer.
        prompt: Input prompt.
        words: List of words to always include (on top of top-k).
        top_k: Number of top predictions to include per layer.

    Returns:
        DataFrame with columns [layer, word, probability, logit, source].
        source is "top_k" or "tracked".
    """
    layer_logits = logit_lens(model, tokenizer, prompt, device)
    words = words or []

    # Encode tracked words with leading space (continuation tokens)
    word_token_ids = {}
    for word in words:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if ids:
            word_token_ids[word] = ids[0]

    rows = []
    for layer_idx, logits in enumerate(layer_logits):
        probs = torch.softmax(logits.float(), dim=0)

        # Top-k words at this layer
        topk = probs.topk(top_k)
        seen_words = set()
        for prob, tid in zip(topk.values, topk.indices):
            word = tokenizer.decode([tid]).strip()
            if not word or len(word) < 2:
                continue
            seen_words.add(word)
            rows.append({
                "layer": layer_idx,
                "word": word,
                "probability": round(float(prob), 8),
                "logit": round(float(logits[tid]), 4),
                "source": "top_k",
            })

        # Always include tracked words
        for word, tid in word_token_ids.items():
            if word not in seen_words:
                rows.append({
                    "layer": layer_idx,
                    "word": word,
                    "probability": round(float(probs[tid]), 8),
                    "logit": round(float(logits[tid]), 4),
                    "source": "tracked",
                })

    return pd.DataFrame(rows)


def get_embeddings(model):
    """Extract the input embedding matrix from a model."""
    return model.get_input_embeddings().weight.detach().cpu()
