from . import *

def _is_amber_model(model_name):
    return model_name.lower().startswith("llm360/amber")

def _load_tokenizer(model_name):
    """
    Load tokenizer with a graceful fallback and actionable error messages.
    """
    from transformers import AutoTokenizer, LlamaTokenizer

    def _looks_piecey(tok):
        """Detect tokenizers that decode with raw SentencePiece markers."""
        try:
            sample_ids = tok.encode("The quick brown fox", add_special_tokens=False)
            sample_text = tok.decode(sample_ids, skip_special_tokens=True)
            return "▁" in sample_text
        except Exception:
            return False

    if _is_amber_model(model_name):
        try:
            # print(f"Using LlamaTokenizer for {model_name}")
            return LlamaTokenizer.from_pretrained(model_name)
        except Exception as llama_error:
            print(f"LlamaTokenizer load failed for {model_name}; falling back. ({llama_error})")

    slow_error = None
    try:
        # Prefer slow tokenizer first for SentencePiece-backed models because
        # some fast conversion paths produce piece-level artifacts in decode.
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)
    except Exception as exc:
        slow_error = exc

    try:
        print("Slow tokenizer load failed; trying fast tokenizer backend.")
        tok = AutoTokenizer.from_pretrained(model_name)
        if getattr(tok, "is_fast", False) and _looks_piecey(tok):
            raise RuntimeError("Fast tokenizer decode appears to emit piece markers.")
        return tok
    except Exception as fast_error:
        combined_error = f"{slow_error}\n{fast_error}".lower()
        if "sentencepiece" in combined_error or "tokenizer.model" in combined_error:
            raise ImportError(
                "Failed to load tokenizer because SentencePiece support is missing. "
                "Install required deps with `pip install sentencepiece protobuf` "
                "and restart your Python runtime."
            ) from fast_error
        raise RuntimeError(
            f"Failed to load tokenizer for '{model_name}'. "
            "Tried both slow and fast tokenizer backends."
        ) from fast_error

def _load_causal_lm(
    model_name,
    quantization_config,
    device_map,
    dtype,
):
    from transformers import AutoModelForCausalLM, LlamaForCausalLM

    common_kwargs = {
        "quantization_config": quantization_config,
        "device_map": device_map,
        "dtype": dtype,
    }

    if _is_amber_model(model_name):
        try:
            # print(f"Using LlamaForCausalLM for {model_name}")
            return LlamaForCausalLM.from_pretrained(model_name, **common_kwargs)
        except Exception as llama_error:
            print(f"LlamaForCausalLM load failed for {model_name}; falling back. ({llama_error})")

    return AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)

def load_model(
    base_name="meta-llama/Llama-3.1-8B",
    load_in_4bit=True,
):
    """
    Load base and instruct models with platform-specific optimizations.

    On Mac (Darwin): Uses MPS (Metal Performance Shaders) with torch.float16
    On other platforms: Uses BitsAndBytes 4-bit quantization if available

    Returns:
        base_model, instruct_model, tokenizer

    Note: You must first accept Meta's license at huggingface.co and
    authenticate with `huggingface_hub.login()`.
    """
    # Detect platform for optimal loading strategy
    is_mac = platform.system() == "Darwin"

    if is_mac:
        # Mac optimization: Use MPS with float16
        # print("Detected Mac - using MPS (Metal Performance Shaders) with torch.float16")
        device_map = "mps"
        dtype = torch.float16
        quantization_config = None
        load_in_4bit = False  # Override - MPS doesn't support BitsAndBytes
    else:
        # PC/Linux optimization: Use BitsAndBytes 4-bit quantization
        print("Detected PC/Linux - using BitsAndBytes 4-bit quantization")
        from transformers import BitsAndBytesConfig

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        device_map = "auto"
        dtype = None

    tokenizer = _load_tokenizer(base_name)

    base_model = _load_causal_lm(
        base_name,
        quantization_config,
        device_map,
        dtype,
    )

    return base_model, tokenizer



def load_models(
    base_name="meta-llama/Llama-3.1-8B",
    instruct_name="meta-llama/Llama-3.1-8B-Instruct",
    load_in_4bit=True,
):
    """
    Load base and instruct models with platform-specific optimizations.

    On Mac (Darwin): Uses MPS (Metal Performance Shaders) with torch.float16
    On other platforms: Uses BitsAndBytes 4-bit quantization if available

    Returns:
        base_model, instruct_model, tokenizer

    Note: You must first accept Meta's license at huggingface.co and
    authenticate with `huggingface_hub.login()`.
    """
    # Detect platform for optimal loading strategy
    is_mac = platform.system() == "Darwin"

    if is_mac:
        # Mac optimization: Use MPS with float16
        # print("Detected Mac - using MPS (Metal Performance Shaders) with torch.float16")
        device_map = "mps"
        dtype = torch.float16
        quantization_config = None
        load_in_4bit = False  # Override - MPS doesn't support BitsAndBytes
    else:
        # PC/Linux optimization: Use BitsAndBytes 4-bit quantization
        print("Detected PC/Linux - using BitsAndBytes 4-bit quantization")
        from transformers import BitsAndBytesConfig

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        device_map = "auto"
        dtype = None

    tokenizer = _load_tokenizer(base_name)

    print("Loading base model...")
    base_model = _load_causal_lm(
        base_name,
        quantization_config,
        device_map,
        dtype,
    )

    print("Loading instruct model...")
    instruct_model = _load_causal_lm(
        instruct_name,
        quantization_config,
        device_map,
        dtype,
    )

    print("Both models loaded.")
    return base_model, instruct_model, tokenizer


def load_three_models(
    base_name=BASE_MODEL_NAME,
    instruct_name=INSTRUCT_MODEL_NAME,
    safe_name=SAFE_MODEL_NAME,
    load_in_4bit=True,
):
    """
    Load base, instruct (chat), and safe (superego) models.

    Returns:
        base_model, instruct_model, safe_model, base_tokenizer, instruct_tokenizer, safe_tokenizer
    """
    base_model, base_tokenizer = load_model(
        base_name=base_name,
        load_in_4bit=load_in_4bit,
    )
    instruct_model, instruct_tokenizer = load_model(
        base_name=instruct_name,
        load_in_4bit=load_in_4bit,
    )
    safe_model, safe_tokenizer = load_model(
        base_name=safe_name,
        load_in_4bit=load_in_4bit,
    )
    return (
        base_model,
        instruct_model,
        safe_model,
        base_tokenizer,
        instruct_tokenizer,
        safe_tokenizer,
    )


def get_base_logits(model, tokenizer, prompt, device=None):
    """Get raw logits from a model for a prompt (for displacement/overdetermination)."""
    if device is None:
        device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :].cpu()
    return logits


def get_embeddings(model):
    """Extract the input embedding matrix from a model."""
    return model.get_input_embeddings().weight.detach().cpu()