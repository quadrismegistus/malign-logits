from . import *

import textwrap

def print_wrapped(text, width=80):
    """Print text with word wrapping at the specified width (default 80)."""
    print(textwrap.fill(text, width=width))

def _decode_generated(tokenizer, token_ids):
    """Decode generated ids."""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

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


def model_generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=1.0,
    top_k=None,
    do_sample=True,
):
    """Simple generation from a single model."""
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
    models,
    prompt,
    max_new_tokens=100,
    temperature=1.0,
    top_k=50,
    do_sample=True,
    verbose=True,
):
    """Generate continuations from multiple model layers.

    Args:
        models: dict mapping layer name -> (model, tokenizer) tuple.
            e.g. {"base": (base_model, tok), "ego": (sft_model, tok), ...}
        prompt: Text string to complete.
        max_new_tokens: Length of each continuation.
        temperature: Sampling temperature.
        verbose: Print results.

    Returns:
        dict mapping layer name -> generated text.
    """
    if verbose:
        print(f'Prompt:   {prompt}\n')

    results = {"prompt": prompt}

    for name, (model, tokenizer) in models.items():
        input_ids, mask = _tokenize_for_generation(
            tokenizer, prompt, model.device,
        )
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = _decode_generated(tokenizer, output_ids[0][input_ids.shape[1]:])
        results[name] = text

        if verbose:
            print_wrapped(f"[{name.capitalize()}]\n{text}\n")
            print()

    return results


def generate_neurotic(
    base_model, sft_model, dpo_model, tokenizer, prompt,
    max_new_tokens=100, temperature=0.8,
    displacement_weight=0.3, hidden_layer=16, embedding_refresh_interval=5,
):
    """
    Neurotic generation using contextual embeddings, recomputing
    them every N tokens instead of every token.

    Compares SFT (ego) and DPO (superego) logits at each step.
    Base model provides drive weighting.

    Args:
        base_model: Base model (drive energy).
        sft_model: SFT model (ego).
        dpo_model: DPO model (superego).
        tokenizer: Shared tokenizer.
        prompt: Text to continue.
        displacement_weight: Controls neurotic intensity.
            1.0 = decompensating body-language.
            0.3 = obsessive intellectualisation.
    """
    # Generate base, ego, superego for comparison
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

    ego_input, ego_mask = _tokenize_for_generation(tokenizer, prompt, sft_model.device)
    with torch.no_grad():
        ego_out = sft_model.generate(
            ego_input, max_new_tokens=max_new_tokens,
            attention_mask=ego_mask,
            do_sample=True, temperature=temperature, top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    ego_text = _decode_generated(tokenizer, ego_out[0][ego_input.shape[1]:])
    print(f"Ego:      {ego_text}\n")

    sup_input, sup_mask = _tokenize_for_generation(tokenizer, prompt, dpo_model.device)
    with torch.no_grad():
        sup_out = dpo_model.generate(
            sup_input, max_new_tokens=max_new_tokens,
            attention_mask=sup_mask,
            do_sample=True, temperature=temperature, top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    sup_text = _decode_generated(tokenizer, sup_out[0][sup_input.shape[1]:])
    print(f"Superego: {sup_text}\n")
    print("Generating neurotic text...")

    # Neurotic generation — token-by-token displacement
    neurotic_ids, neurotic_mask = _tokenize_for_generation(
        tokenizer, prompt, sft_model.device
    )
    base_ids, base_gen_mask = _tokenize_for_generation(tokenizer, prompt, base_model.device)
    dpo_ids, dpo_mask = _tokenize_for_generation(tokenizer, prompt, dpo_model.device)
    prompt_len = neurotic_ids.shape[1]

    cached_sim_matrix = None
    cached_repressed_indices = None
    cached_permitted_indices = None

    symptom_log = []

    for step in tqdm(list(range(max_new_tokens))):
        with torch.no_grad():
            # Ego logits from SFT model
            ego_logits = sft_model(
                neurotic_ids, attention_mask=neurotic_mask
            ).logits[0, -1, :].cpu().float()

            # Superego logits from DPO model
            sup_logits = dpo_model(
                dpo_ids, attention_mask=dpo_mask
            ).logits[0, -1, :].cpu().float()

            # Base logits for drive weighting
            base_logits = base_model(
                base_ids, attention_mask=base_gen_mask
            ).logits[0, -1, :].cpu().float()

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
            current_prompt_text = tokenizer.decode(
                neurotic_ids[0], skip_special_tokens=True
            )
            device = sft_model.device

            def get_ctx_emb(token_id):
                token_text = tokenizer.decode(token_id)
                text = current_prompt_text + " " + token_text
                ids = tokenizer.encode(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = sft_model(ids, output_hidden_states=True)
                    hidden = out.hidden_states[hidden_layer]
                    return hidden[0, -1, :].cpu().float()

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

        # Advance all sequences with the same token
        neurotic_ids = torch.cat([
            neurotic_ids, neurotic_token.unsqueeze(0).to(sft_model.device)
        ], dim=-1)
        neurotic_mask = torch.cat(
            [neurotic_mask, torch.ones((1, 1), device=sft_model.device, dtype=neurotic_mask.dtype)],
            dim=-1,
        )
        base_ids = torch.cat([
            base_ids, neurotic_token.unsqueeze(0).to(base_model.device)
        ], dim=-1)
        base_gen_mask = torch.cat(
            [base_gen_mask, torch.ones((1, 1), device=base_model.device, dtype=base_gen_mask.dtype)],
            dim=-1,
        )
        dpo_ids = torch.cat([
            dpo_ids, neurotic_token.unsqueeze(0).to(dpo_model.device)
        ], dim=-1)
        dpo_mask = torch.cat(
            [dpo_mask, torch.ones((1, 1), device=dpo_model.device, dtype=dpo_mask.dtype)],
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
