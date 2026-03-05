import platform

import torch


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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Detect platform for optimal loading strategy
    is_mac = platform.system() == "Darwin"

    if is_mac:
        # Mac optimization: Use MPS with float16
        print("Detected Mac - using MPS (Metal Performance Shaders) with torch.float16")
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

    tokenizer = AutoTokenizer.from_pretrained(base_name)

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        quantization_config=quantization_config,
        device_map=device_map,
        dtype=dtype,
    )

    print("Loading instruct model...")
    instruct_model = AutoModelForCausalLM.from_pretrained(
        instruct_name,
        quantization_config=quantization_config,
        device_map=device_map,
        dtype=dtype,
    )

    print("Both models loaded.")
    return base_model, instruct_model, tokenizer


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