from . import *


def _load_tokenizer(model_name, revision=None, cache_dir=None):
    """Load tokenizer for a HuggingFace model."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    kwargs = {}
    if revision:
        kwargs["revision"] = revision
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **kwargs)
    # Some models (e.g. DeepSeek LLM 7B) ship a slow LlamaTokenizer that
    # doesn't decode GPT-2 byte-pair markers (Ġ→space). Detect and fix.
    # add_special_tokens=False so a prepended BOS (Llama, Amber, ...) doesn't
    # make every such tokenizer look broken and trigger a needless reload.
    if tok.decode(tok.encode("a b", add_special_tokens=False)) != "a b":
        tok = PreTrainedTokenizerFast.from_pretrained(model_name, trust_remote_code=True, **kwargs)
    return tok


def _load_causal_lm(model_name, quantization_config, device_map, dtype,
                     revision=None, cache_dir=None):
    from transformers import AutoModelForCausalLM
    kwargs = {}
    if revision:
        kwargs["revision"] = revision
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=True,
            **kwargs,
        )
    except OSError:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=True,
            use_safetensors=False,
            **kwargs,
        )


def _platform_kwargs():
    """Return device_map, dtype, quantization_config for the current platform."""
    if platform.system() == "Darwin":
        return {
            "device_map": "mps",
            "dtype": torch.float16,
            "quantization_config": None,
        }
    elif torch.cuda.is_available():
        return {
            "device_map": "auto",
            "dtype": torch.float16,
            "quantization_config": None,
        }
    else:
        return {
            "device_map": "cpu",
            "dtype": torch.float32,
            "quantization_config": None,
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

        # Get norm and lm_head for projection (architecture-dependent)
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            norm = model.model.norm          # Llama, OLMo, Mistral, Qwen
        elif hasattr(model, 'gpt_neox'):
            norm = model.gpt_neox.final_layer_norm  # GPT-NeoX (Pythia)
        elif hasattr(model, 'transformer'):
            norm = model.transformer.ln_f    # GPT-2, GPT-Neo
        else:
            raise AttributeError(f"Cannot find final layer norm for {type(model).__name__}")
        lm_head = model.lm_head

        # THE LAST ENTRY IS ALREADY NORMED AND MUST NOT BE NORMED AGAIN.
        # HuggingFace appends hidden states INSIDE the decoder loop -- each one
        # the input to its layer, so pre-norm -- then applies the final norm
        # after the loop and appends THAT as the last element. `norm(norm(x))`
        # re-applies the learned weight to a vector that is already unit-scale,
        # which is not a rescaling but a different direction, and it lands on
        # the one layer every logit-lens claim is read off.
        #
        # LLM360/Amber, "She was so angry she wanted to":
        #     head(hidden[-1])         kill 0.119145   == the model's own logits
        #     head(norm(hidden[-1]))   kill 0.059886   maxdiff 0.244
        last = len(hidden_states) - 1
        layer_logits = []
        for i, hidden in enumerate(hidden_states):
            normed = hidden if i == last else norm(hidden)
            logits = lm_head(normed)[0, -1, :].cpu()
            layer_logits.append(logits)

        # AND THE CHECK THAT WOULD HAVE CAUGHT IT, WHICH COSTS NOTHING. The
        # final layer's projection IS the model's output; the forward pass has
        # already computed it, so disagreement means the projection is wrong for
        # this architecture and no layer from it is readable. This was invisible
        # for as long as nobody compared the two.
        ref = outputs.logits[0, -1, :].detach().float().cpu()
        gap = float((layer_logits[-1].detach().float() - ref).abs().max())
        if gap > 1e-2:
            raise AssertionError(
                "logit lens final layer does not reproduce the model's own "
                "logits (max abs diff %.4g) for %s. The projection is wrong "
                "for this architecture; do not read any layer from it."
                % (gap, type(model).__name__))

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

    # Encode tracked words with leading space (continuation tokens).
    #
    # `ids[0]` IS THE FIRST TOKEN, NOT THE WORD. For a multi-token word this
    # reads a fragment and labels it with the whole word: on LLM360/Amber
    # " scream" is ['sc', 'ream'], so the tracked row is the probability of
    # 'sc' -- shared with scare, scratch, scold -- under the name `scream`.
    # 31% of that model's movement-vocabulary mass is multi-token.
    #
    # The semantics are unchanged (callers depend on them) but `n_tokens` and
    # `first_token` now travel with every row, so a consumer can tell a word
    # probability from a prefix probability instead of having to know. For a
    # calibrated word trajectory use true_word_probs, which records `t1` per
    # word and gives p(word) at the output to license the prefix reading.
    word_token_ids, word_n_tokens = {}, {}
    for word in words:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if ids:
            word_token_ids[word] = ids[0]
            word_n_tokens[word] = len(ids)

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
                    #: 1 means `probability` IS the word's; >1 means it is the
                    #: probability of `first_token` and an upper bound on the word's.
                    "n_tokens": word_n_tokens[word],
                    "first_token": tokenizer.decode([tid]),
                })

    return pd.DataFrame(rows)


def get_embeddings(model):
    """Extract the input embedding matrix from a model."""
    return model.get_input_embeddings().weight.detach().cpu()
