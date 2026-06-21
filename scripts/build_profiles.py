"""Build CircuitProfile CSVs for all families from cached data.

Loads each base model once for embedding axes, reads cached logits,
computes node + edge + mode profiles, writes to data/profiles/.

Usage:
    python scripts/build_profiles.py              # all families with cached logits
    python scripts/build_profiles.py --family olmo # single family
"""
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import sys
sys.path.insert(0, ".")

import argparse
import torch
import numpy as np
from malign_logits import MODEL_FAMILIES
from malign_logits.cache import get_cache
from malign_logits.analysis import js_divergence
from malign_logits.circuit import Circuit
from malign_logits.profile import (
    CircuitProfile, FamilyMetadata, NodeProfile, EdgeProfile, ModeProfile,
    PROMPTS, build_axes,
)
from malign_logits.models import load_model

cm = get_cache()

FAMILY_META = {
    "olmo": ("7B", "Dolma", "Tulu 3", "SFT+DPO+RLVR", "US", "Allen AI", True, True),
    "olmo-think": ("7B", "Dolma", "Tulu 3 Think", "Think-SFT+Think-DPO", "US", "Allen AI", True, True),
    "olmo-tiny": ("1B", "Dolma", "Tulu 3", "SFT+DPO+RLVR", "US", "Allen AI", False, True),
    "llama": ("8B", "undisclosed", "undisclosed", "SFT+DPO", "US", "Meta", False, False),
    "tulu": ("8B", "undisclosed", "Tulu 3", "SFT+DPO+RLVR", "US", "Allen AI", True, True),
    "qwen": ("7B", "undisclosed", "undisclosed", "SFT+DPO", "China", "Alibaba", True, False),
    "qwen3": ("8B", "undisclosed", "undisclosed", "SFT+DPO", "China", "Alibaba", True, False),
    "amber": ("7B", "RefinedWeb+others", "HH-RLHF", "SFT+DPO", "US", "LLM360", True, True),
    "zephyr": ("7B", "undisclosed (Mistral)", "UltraFeedback", "SFT+DPO", "US", "HuggingFace", True, True),
    "deepseek-7b": ("7B", "undisclosed", "undisclosed", "SFT+DPO", "China", "DeepSeek", True, False),
    "pythia": ("6.9B", "The Pile", "HH-RLHF", "SFT+DPO", "US", "EleutherAI+lomahony", False, True),
    "smol": ("360M", "FineWeb", "undisclosed", "DPO", "EU", "HuggingFace", False, False),
    "qwen-tiny": ("0.5B", "undisclosed", "undisclosed", "SFT+DPO", "China", "Alibaba", True, False),
    "smol3": ("3B", "FineWeb", "undisclosed", "APO", "EU", "HuggingFace", False, False),
}


def build_family_profile(family_key):
    fam = MODEL_FAMILIES[family_key]
    meta_tuple = FAMILY_META.get(family_key)
    if not meta_tuple:
        print(f"  {family_key}: no metadata, skipping")
        return None

    scale, corpus, al_data, al_method, country, org, open_w, open_d = meta_tuple

    layers = ["base"]
    checkpoints = {"base": fam.base}
    if fam.ego:
        layers.append("ego")
        checkpoints["ego"] = fam.ego
    if fam.superego:
        layers.append("superego")
        checkpoints["superego"] = fam.superego
    if fam.reinforced_superego:
        layers.append("rlvr")
        checkpoints["rlvr"] = fam.reinforced_superego

    metadata = FamilyMetadata(
        family=family_key, scale=scale, base_corpus=corpus,
        alignment_data=al_data, alignment_method=al_method,
        n_layers=len(layers), layer_names=layers,
        has_chat_template=bool(meta_tuple[6]),
        country=country, org=org, open_weight=open_w, open_data=open_d,
    )

    # Check if we have cached logits
    has_logits = False
    for cp_name, model_id in checkpoints.items():
        for pk in PROMPTS:
            if cm.get_logits(model_id, PROMPTS[pk]) is not None:
                has_logits = True
                break
        if has_logits:
            break

    if not has_logits:
        print(f"  {family_key}: no cached logits, skipping")
        return None

    # Load base model for embedding axes
    print(f"  {family_key}: loading {fam.base} for axes...", flush=True)
    model, tokenizer = load_model(fam.base)
    embed = model.get_input_embeddings().weight.detach().cpu().float()
    vocab_size = embed.shape[0]
    violence_axis, proc_axis = build_axes(embed, tokenizer)
    embed_v = (embed @ violence_axis).numpy()
    embed_p = (embed @ proc_axis).numpy()
    del model
    torch.mps.empty_cache() if hasattr(torch.mps, "empty_cache") else None

    profile = CircuitProfile(metadata=metadata)

    # Build node profiles
    logits_cache = {}
    for cp_name, model_id in checkpoints.items():
        for pk, prompt_text in PROMPTS.items():
            logits = cm.get_logits(model_id, prompt_text)
            if logits is None:
                continue
            n = min(len(logits), vocab_size)
            logits_t = torch.tensor(logits[:n]).float()
            probs = torch.softmax(logits_t, dim=-1)

            h = -(probs * torch.log(probs + 1e-10)).sum().item()
            eff = min(int(np.exp(h)), n)
            top_vals, top_idx = probs.topk(10)
            top_k = [(tokenizer.decode([idx]).strip(), float(val))
                     for idx, val in zip(top_idx, top_vals)]

            ev = float((probs.numpy() * embed_v[:n]).sum())
            ep = float((probs.numpy() * embed_p[:n]).sum())

            profile.nodes.append(NodeProfile(
                checkpoint=cp_name, prompt=pk, entropy=round(h, 4),
                effective_vocab=eff, top_k=top_k,
                violence_loading=round(ev, 5),
                procedural_loading=round(ep, 5),
                argmax_token=top_k[0][0], argmax_prob=round(top_k[0][1], 4),
            ))

            logits_cache[(cp_name, pk)] = logits_t

    # Build edge profiles
    for i in range(len(layers) - 1):
        from_cp = layers[i]
        to_cp = layers[i + 1]
        for pk in PROMPTS:
            if (from_cp, pk) not in logits_cache or (to_cp, pk) not in logits_cache:
                continue
            from_logits = logits_cache[(from_cp, pk)]
            to_logits = logits_cache[(to_cp, pk)]
            n = min(len(from_logits), len(to_logits))

            js = float(js_divergence(from_logits[:n], to_logits[:n]))

            from_probs = torch.softmax(from_logits[:n].float(), dim=-1)
            to_probs = torch.softmax(to_logits[:n].float(), dim=-1)
            delta = (to_probs - from_probs).numpy()

            top_gain_idx = np.argsort(delta)[-10:][::-1]
            top_lose_idx = np.argsort(delta)[:10]
            gainers = [(tokenizer.decode([idx]).strip(), float(delta[idx]))
                      for idx in top_gain_idx if delta[idx] > 0.001]
            losers = [(tokenizer.decode([idx]).strip(), float(delta[idx]))
                     for idx in top_lose_idx if delta[idx] < -0.001]

            from_node = profile.node(from_cp, pk)
            to_node = profile.node(to_cp, pk)
            argmax_change = f"{from_node.argmax_token} → {to_node.argmax_token}" if from_node and to_node else "?"

            # Classify signature
            base_top1 = profile.node("base", pk).argmax_token if profile.node("base", pk) else None
            if to_node and base_top1:
                is_blank = any(c in to_node.argmax_token for c in ("_", "▁")) or to_node.argmax_token.strip() in ("", "nan", "None", "?")
                base_blank = any(c in str(base_top1) for c in ("_", "▁")) or str(base_top1).strip() in ("", "nan", "None", "?")
                if is_blank:
                    signature = "foreclosure"
                elif base_blank and not is_blank:
                    signature = "de_foreclosure"
                elif to_node.argmax_token == base_top1:
                    signature = "transparent"
                elif to_node.argmax_token != from_node.argmax_token:
                    signature = "repression"
                else:
                    signature = "repression"
            else:
                signature = "unknown"

            from_h = from_node.entropy if from_node else 0
            to_h = to_node.entropy if to_node else 0

            profile.edges.append(EdgeProfile(
                from_checkpoint=from_cp, to_checkpoint=to_cp, prompt=pk,
                js_divergence=round(js, 5), argmax_change=argmax_change,
                top_gainers=gainers[:10], top_losers=losers[:10],
                signature=signature, sft_share=None,
                delta_entropy=round(to_h - from_h, 4),
            ))

    print(f"  {family_key}: {len(profile.nodes)} nodes, {len(profile.edges)} edges", flush=True)
    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str)
    args = parser.parse_args()

    if args.family:
        families = [args.family]
    else:
        families = list(FAMILY_META.keys())

    profiles = {}
    for fam in families:
        if fam not in MODEL_FAMILIES:
            continue
        p = build_family_profile(fam)
        if p:
            p.to_csv("data/profiles")
            profiles[fam] = p

    # Print worker table
    print(f"\n{'='*70}")
    print("WORKER TABLE (from profiles)")
    print(f"{'='*70}")
    print(f"{'Family':<14} {'Base':>8} → {'Aligned':>8}  {'Mechanism':<20} {'Proc loading':>12}")
    print("-" * 70)
    for fam, p in profiles.items():
        ws = p.worker_summary()
        if ws:
            print(f"{ws['family']:<14} {ws['base_argmax']:>8} → {ws['aligned_argmax']:>8}  "
                  f"{ws['mechanism']:<20} {ws['procedural_loading']:>+12.4f}")

    print(f"\nSaved {len(profiles)} profiles to data/profiles/")


if __name__ == "__main__":
    main()
