import torch
from tqdm import tqdm

from .core import DEFAULT_SUPEREGO_PREFIX

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
    """
    if superego_prefix is None:
        superego_prefix = DEFAULT_SUPEREGO_PREFIX

    # Generate base
    base_input = tokenizer.encode(prompt, return_tensors="pt").to(base_model.device)
    with torch.no_grad():
        base_out = base_model.generate(
            base_input, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    base_text = tokenizer.decode(base_out[0][base_input.shape[1]:], skip_special_tokens=True)
    print(f"Base:     {base_text}\n")

    # Generate ego and superego fast
    ego_input = tokenizer.encode(prompt, return_tensors="pt").to(instruct_model.device)
    with torch.no_grad():
        ego_out = instruct_model.generate(
            ego_input, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    ego_text = tokenizer.decode(ego_out[0][ego_input.shape[1]:], skip_special_tokens=True)
    print(f"Ego:      {ego_text}\n")

    sup_full = tokenizer.encode(
        superego_prefix + prompt, return_tensors="pt"
    ).to(instruct_model.device)
    with torch.no_grad():
        sup_out = instruct_model.generate(
            sup_full, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    sup_text = tokenizer.decode(sup_out[0][sup_full.shape[1]:], skip_special_tokens=True)

    print(f"Superego: {sup_text}\n")
    print("Generating neurotic text...")

    # Neurotic generation
    neurotic_ids = tokenizer.encode(prompt, return_tensors="pt").to(instruct_model.device)
    base_ids = tokenizer.encode(prompt, return_tensors="pt").to(base_model.device)
    prompt_len = neurotic_ids.shape[1]

    # Cache for contextual embeddings
    cached_sim_matrix = None
    cached_repressed_indices = None
    cached_permitted_indices = None

    symptom_log = []

    for step in tqdm(list(range(max_new_tokens))):
        with torch.no_grad():
            ego_logits = instruct_model(neurotic_ids).logits[0, -1, :].cpu().float()

            sup_prefix_ids = tokenizer.encode(
                superego_prefix, return_tensors="pt"
            ).to(instruct_model.device)
            sup_full_ids = torch.cat([sup_prefix_ids, neurotic_ids], dim=-1)
            sup_logits = instruct_model(sup_full_ids).logits[0, -1, :].cpu().float()

            base_logits = base_model(base_ids).logits[0, -1, :].cpu().float()

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
        base_ids = torch.cat([
            base_ids, neurotic_token.unsqueeze(0).to(base_model.device)
        ], dim=-1)

        if token_id == tokenizer.eos_token_id:
            break

    neurotic_text = tokenizer.decode(
        neurotic_ids[0, prompt_len:], skip_special_tokens=True
    )

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