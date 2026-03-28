from . import *


def _load_tokenizer(model_name):
    """Load tokenizer for a HuggingFace model."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


def _load_causal_lm(model_name, quantization_config, device_map, dtype):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        dtype=dtype,
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


def load_model(model_name):
    """Load a single model and its tokenizer.

    Returns:
        (model, tokenizer)
    """
    kwargs = _platform_kwargs()
    tokenizer = _load_tokenizer(model_name)
    print(f"Loading {model_name}...")
    model = _load_causal_lm(
        model_name,
        kwargs["quantization_config"],
        kwargs["device_map"],
        kwargs["dtype"],
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


def get_embeddings(model):
    """Extract the input embedding matrix from a model."""
    return model.get_input_embeddings().weight.detach().cpu()
