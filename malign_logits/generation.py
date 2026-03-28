from . import *

import textwrap

def print_wrapped(text, width=80):
    """Print text with word wrapping at the specified width (default 80)."""
    print(textwrap.fill(text, width=width))

def _decode_generated(tokenizer, token_ids):
    """Decode generated ids and normalize fallback SentencePiece marker output."""
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    # Some fast-tokenizer fallback paths can leak raw SentencePiece markers.
    if "▁" in text:
        text = text.replace("▁", " ")
        text = " ".join(text.split())
    return text

def _tokenize_for_generation(tokenizer, text, device):
    """Tokenize text and always provide an attention mask."""
    encoded = tokenizer(text, return_tensors="pt", padding=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def _build_superego_prompt(prompt, superego_prefix, use_prefix):
    if use_prefix:
        return superego_prefix + prompt
    return prompt

def model_generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=1.0,
    top_k=None,
    do_sample=True,
):
    """Simple generation for model continuations."""
    def _continue(model, input_ids, attention_mask):
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                pad_token_id=tokenizer.eos_token_id,
            )
        return _decode_generated(tokenizer, output_ids[0][input_ids.shape[1]:])

    print(f'Prompt:   {prompt}\n')
    base_input, base_mask = _tokenize_for_generation(tokenizer, prompt, model.device)
    text = _continue(model, base_input, base_mask)
    print_wrapped(text)
    print()
    return text


def generate(
    base_model,
    instruct_model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    superego_prefix=None,
    temperature=1.0,
    top_k=50,
    do_sample=True,
    superego_model=None,
    base_tokenizer=None,
    instruct_tokenizer=None,
    superego_tokenizer=None,
    superego_use_prefix=True,
    verbose=True,
):
    """Simple generation for base, ego, and superego continuations."""
    if superego_prefix is None:
        superego_prefix = DEFAULT_SUPEREGO_PREFIX
    if superego_model is None:
        superego_model = instruct_model
    base_tokenizer = base_tokenizer or tokenizer
    instruct_tokenizer = instruct_tokenizer or tokenizer
    superego_tokenizer = superego_tokenizer or instruct_tokenizer

    def _continue(model, layer_tokenizer, input_ids, attention_mask):
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                pad_token_id=layer_tokenizer.eos_token_id,
            )
        return _decode_generated(layer_tokenizer, output_ids[0][input_ids.shape[1]:])

    if verbose:
        print(f'Prompt:   {prompt}\n')
    base_input, base_mask = _tokenize_for_generation(base_tokenizer, prompt, base_model.device)
    ego_input, ego_mask = _tokenize_for_generation(instruct_tokenizer, prompt, instruct_model.device)
    superego_input, superego_mask = _tokenize_for_generation(
        superego_tokenizer,
        _build_superego_prompt(prompt, superego_prefix, superego_use_prefix),
        superego_model.device,
    )

    base_text = _continue(base_model, base_tokenizer, base_input, base_mask)
    if verbose:
        print_wrapped(f"[Base]\n{base_text}\n")
        print()

    ego_text = _continue(instruct_model, instruct_tokenizer, ego_input, ego_mask)
    if verbose:
        print_wrapped(f"[Ego]\n{ego_text}\n")
        print()
    
    superego_text = _continue(
        superego_model,
        superego_tokenizer,
        superego_input,
        superego_mask,
    )
    if verbose:
        print_wrapped(f"[Superego]\n{superego_text}")
        print()

    return {
        "prompt": prompt,
        "base": base_text,
        "ego": ego_text,
        "superego": superego_text,
    }


def generate_neurotic(
    base_model, instruct_model, tokenizer, prompt,
    max_new_tokens=100, superego_prefix=None, temperature=0.8,
    displacement_weight=0.3, hidden_layer=16, embedding_refresh_interval=5,
):
    """
    Neurotic generation using contextual embeddings, recomputing
    them every N tokens instead of every token.

    Args:
        displacement_weight: Controls neurotic intensity.
            1.0 = decompensating body-language.
            0.3 = obsessive intellectualisation.

    Note:
        This method assumes all compared logits share one tokenizer vocabulary.
        For heterogeneous tokenizers, use `generate(...)` without neurotic mode.
    """
    if superego_prefix is None:
        superego_prefix = DEFAULT_SUPEREGO_PREFIX

    # Generate base
    base_input, base_mask = _tokenize_for_generation(tokenizer, prompt, base_model.device)
    with torch.no_grad():
        base_out = base_model.generate(
            base_input, max_new_tokens=max_new_tokens,
            attention_mask=base_mask,
            do_sample=True, temperature=temperature, top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    base_text = _decode_generated(tokenizer, base_out[0][base_input.shape[1]:])
    print(f"Base:     {base_text}\n")

    # Generate ego and superego fast
    ego_input, ego_mask = _tokenize_for_generation(tokenizer, prompt, instruct_model.device)
    with torch.no_grad():
        ego_out = instruct_model.generate(
            ego_input, max_new_tokens=max_new_tokens,
            attention_mask=ego_mask,
            do_sample=True, temperature=temperature, top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    ego_text = _decode_generated(tokenizer, ego_out[0][ego_input.shape[1]:])
    print(f"Ego:      {ego_text}\n")

    sup_full, sup_mask = _tokenize_for_generation(
        tokenizer, superego_prefix + prompt, instruct_model.device
    )
    with torch.no_grad():
        sup_out = instruct_model.generate(
            sup_full, max_new_tokens=max_new_tokens,
            attention_mask=sup_mask,
            do_sample=True, temperature=temperature, top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    sup_text = _decode_generated(tokenizer, sup_out[0][sup_full.shape[1]:])

    print(f"Superego: {sup_text}\n")
    print("Generating neurotic text...")

    # Neurotic generation
    neurotic_ids, neurotic_mask = _tokenize_for_generation(
        tokenizer, prompt, instruct_model.device
    )
    base_ids, base_mask = _tokenize_for_generation(tokenizer, prompt, base_model.device)
    prompt_len = neurotic_ids.shape[1]

    # Cache for contextual embeddings
    cached_sim_matrix = None
    cached_repressed_indices = None
    cached_permitted_indices = None

    symptom_log = []

    for step in tqdm(list(range(max_new_tokens))):
        with torch.no_grad():
            ego_logits = instruct_model(
                neurotic_ids, attention_mask=neurotic_mask
            ).logits[0, -1, :].cpu().float()

            sup_prefix_ids, sup_prefix_mask = _tokenize_for_generation(
                tokenizer, superego_prefix, instruct_model.device
            )
            sup_full_ids = torch.cat([sup_prefix_ids, neurotic_ids], dim=-1)
            sup_full_mask = torch.cat([sup_prefix_mask, neurotic_mask], dim=-1)
            sup_logits = instruct_model(
                sup_full_ids, attention_mask=sup_full_mask
            ).logits[0, -1, :].cpu().float()

            base_logits = base_model(base_ids, attention_mask=base_mask).logits[0, -1, :].cpu().float()

        ego_probs = torch.softmax(ego_logits / temperature, dim=-1)
        sup_probs = torch.softmax(sup_logits / temperature, dim=-1)
        base_probs = torch.softmax(base_logits / temperature, dim=-1)

        # Identify repressed tokens
        repression = ego_probs - sup_probs
        repressed_mask = repression > 0.001
        permitted_mask = ~repressed_mask & (sup_probs > 1e-6)

        repressed_indices = repressed_mask.nonzero(as_tuple=True)[0]
        permitted_indices = permitted_mask.nonzero(as_tuple=True)[0]

        # Drive weighting
        mean_base = base_probs[base_probs > 0].mean()
        drive_weight = 1 + torch.log(1 + base_probs / (mean_base + 1e-10))
        effective_mass = torch.where(
            repressed_mask,
            repression * drive_weight * displacement_weight,
            torch.zeros_like(repression),
        )

        # Recompute contextual embeddings periodically
        needs_refresh = (
            step % embedding_refresh_interval == 0
            or cached_sim_matrix is None
        )

        if needs_refresh and len(repressed_indices) > 0 and len(permitted_indices) > 0:
            # print(f"  Step {step}: computing contextual embeddings...")

            current_prompt_text = tokenizer.decode(
                neurotic_ids[0], skip_special_tokens=True
            )
            device = instruct_model.device

            def get_ctx_emb(token_id):
                token_text = tokenizer.decode(token_id)
                text = current_prompt_text + " " + token_text
                ids = tokenizer.encode(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = instruct_model(ids, output_hidden_states=True)
                    hidden = out.hidden_states[hidden_layer]
                    return hidden[0, -1, :].cpu().float()

            # Only compute embeddings for top repressed and permitted tokens
            # to keep it tractable
            top_repressed = repressed_indices[
                effective_mass[repressed_indices].argsort(descending=True)[:20]
            ]
            top_permitted = permitted_indices[
                sup_probs[permitted_indices].argsort(descending=True)[:100]
            ]

            rep_embs = []
            for idx in top_repressed:
                rep_embs.append(get_ctx_emb(idx.item()))
            rep_embs = torch.stack(rep_embs)
            rep_embs = torch.nn.functional.normalize(rep_embs, dim=-1)

            perm_embs = []
            for idx in top_permitted:
                perm_embs.append(get_ctx_emb(idx.item()))
            perm_embs = torch.stack(perm_embs)
            perm_embs = torch.nn.functional.normalize(perm_embs, dim=-1)

            cached_sim_matrix = torch.clamp(rep_embs @ perm_embs.T, min=0)
            cached_repressed_indices = top_repressed
            cached_permitted_indices = top_permitted

        # Displace
        neurotic_probs = sup_probs.clone()
        neurotic_probs[repressed_mask] = 0

        if cached_sim_matrix is not None:
            for i, rep_idx in enumerate(cached_repressed_indices):
                if i >= cached_sim_matrix.shape[0]:
                    break
                mass = effective_mass[rep_idx].item()
                if mass < 1e-6:
                    continue
                sims = cached_sim_matrix[i]
                if sims.sum() > 0:
                    weights = sims / sims.sum()
                    neurotic_probs[cached_permitted_indices] += mass * weights

        neurotic_probs = torch.clamp(neurotic_probs, min=0)
        neurotic_probs = neurotic_probs / neurotic_probs.sum()

        # Sample
        neurotic_token = torch.multinomial(neurotic_probs, 1)
        token_id = neurotic_token.item()

        # Log symptoms
        gained = (neurotic_probs[token_id] - sup_probs[token_id]).item()
        if gained > 0.003:
            token_text = tokenizer.decode(token_id).strip()
            symptom_log.append({
                'position': step,
                'token': token_text,
                'gained': gained,
            })

        neurotic_ids = torch.cat([
            neurotic_ids, neurotic_token.unsqueeze(0).to(instruct_model.device)
        ], dim=-1)
        neurotic_mask = torch.cat(
            [neurotic_mask, torch.ones((1, 1), device=instruct_model.device, dtype=neurotic_mask.dtype)],
            dim=-1,
        )
        base_ids = torch.cat([
            base_ids, neurotic_token.unsqueeze(0).to(base_model.device)
        ], dim=-1)
        base_mask = torch.cat(
            [base_mask, torch.ones((1, 1), device=base_model.device, dtype=base_mask.dtype)],
            dim=-1,
        )

        if token_id == tokenizer.eos_token_id:
            break

    neurotic_text = _decode_generated(tokenizer, neurotic_ids[0, prompt_len:])

    print(f"\n{'='*60}")
    print(f"PROMPT:   {prompt}")
    print(f"BASE:     {base_text}")
    print(f"EGO:      {ego_text}")
    print(f"SUPEREGO: {sup_text}")
    print(f"NEUROTIC: {neurotic_text}")
    print(f"\nSymptoms ({len(symptom_log)}):")
    for s in symptom_log:
        print(f"  [{s['position']}] '{s['token']}' (gained {s['gained']:.4f})")

    return {
        'prompt': prompt,
        'base': base_text,
        'ego': ego_text,
        'superego': sup_text,
        'neurotic': neurotic_text,
        'symptom_log': symptom_log,
    }


# Backwards-compatible alias
generate_neurotic_contextual = generate_neurotic