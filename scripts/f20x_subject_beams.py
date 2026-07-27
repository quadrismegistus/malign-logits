"""F20 expansion, Tier 1: the conditions of the "I".

Spec: TheoryMachines/agents/lacan/f20-expansion-prereg.md (revision 3).
Written by the lacan seat at RH's request. Namespaced f20x_* to avoid
collision with malign's paths.

Tier 1 asks whether first-person self-predication requires a discursive
position, by beam search at shallow depth in two modes across both arms.

TEMPLATE RULE (registered): each family's ALIGNED-arm chat template is applied
to its BASE arm. Not universal ChatML. The template is rendered to a STRING and
passed to both arms as raw text, so each model tokenises it as it can -- the
aligned arm resolves its own special tokens, the base arm sees the pattern as
pieces, which is the same condition under which ChatML appears in pretraining.

Loads each model once and runs all cells, rather than calling beam_storylines
per cell, which reloads the model every time. The generate call is replicated
exactly: length_penalty=0.0 so sequences_scores is the raw summed logprob.

Usage:
    uv run python scripts/f20x_subject_beams.py --smoke          # one tiny family
    uv run python scripts/f20x_subject_beams.py                  # full roster
    uv run python scripts/f20x_subject_beams.py --families llama,olmo
"""
import argparse
import gc
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from malign_logits import MODEL_FAMILIES
from malign_logits.models import load_model

PROMPTS = {
    "who_are_you": "Who are you?",
    "what_are_you": "What are you?",
    "introduce": "Introduce yourself.",
    "tell_about": "Tell me about yourself.",
}

# Non-identity controls, form-matched to the identity prompts. Added to TIER 1
# before any roster data existed, on malign's advice after its F28 scale-up died
# on a control it could have run first. If P_self is as high here, the template
# installs a first-person speaking habit rather than a self-predicating
# position, and H1's interpretation does not survive.
# ANSWERABLE non-identity, form-matched. This is the real specificity control.
CONTROLS = {
    "who_wrote_hamlet": "Who wrote Hamlet?",        # matches who_are_you
    "what_is_photo": "What is photosynthesis?",     # matches what_are_you
    "describe_weather": "Describe the weather.",    # matches introduce
    "tell_about_france": "Tell me about France.",   # matches tell_about
}

# UNANSWERABLE deictics, kept as a separate reported condition. In the smoke
# test these scored narrow P_self 0.652 and 0.486 against 0.000-0.008 for the
# answerable controls: with no antecedent the model has nothing to say and
# falls back to self-description. That is a real behaviour and a bad control.
UNANSWERABLE = {
    "who_is_she": "Who is she?",
    "what_is_that": "What is that?",
}
ALL_PROMPTS = {**PROMPTS, **CONTROLS, **UNANSWERABLE}


def prompt_class(k):
    if k in PROMPTS:
        return "identity"
    return "control" if k in CONTROLS else "unanswerable"


IS_IDENTITY = {k: (k in PROMPTS) for k in ALL_PROMPTS}

# Registered patterns. Broad is primary; narrow subtracts non-identity predicates.
BROAD = re.compile(r"^\s*(I am|I'm|My name is|This is)\b", re.I)
STOPLIST = ("not sure", "sorry", "afraid", "glad", "here to", "happy to", "confused")

EXCLUDE = {"olmo-32b", "llama-70b"}  # will not fit / too slow for now
N_BEAMS = 100
DEPTH = 10  # matches the existing storylines corpus
OUT_BEAMS = "data/f20x_beams.parquet"
OUT_SUMMARY = "data/f20x_summary.csv"


def self_predicates(text):
    """(broad, narrow) — does this continuation self-predicate?"""
    if not BROAD.match(text):
        return False, False
    head = text.strip()[:60].lower()
    return True, not any(s in head for s in STOPLIST)


def templated(prompt, aligned_model_id):
    """Render the ALIGNED arm's chat template to a string. None if it has none."""
    from malign_logits.models import _load_tokenizer
    tok = _load_tokenizer(aligned_model_id)
    if not getattr(tok, "chat_template", None):
        return None
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return None


CHATML = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"

# THE LADDER (RH's design). ChatML's special tokens do two things at once: mark a
# turn position, AND cue an AI-assistant transcript, since that is the only place
# ChatML occurs in a corpus. These plain-text frames separate the two. Each rung
# adds exactly one thing, so the difference between rungs is interpretable.
#
#   dyad_me_you   position + person-reversal, no roles named. ME/YOU are shifters
#                 that reverse with the speech act, so this also tests whether the
#                 model works out that it is the one addressed as "you".
#   dyad_qa       turn structure with no persons at all
#   dyad_user_ast roles named, plain text (no special tokens)
#   dyad_human_ai explicitly AI, plain text
#   chatml        AI-associated SPECIAL TOKENS
#   chat          plus the family's asserted identity content
#
# NOT cue-free: plain dialogue frames are all over the corpus in interviews,
# screenplays and forum posts. Rung 1 cues DIALOGUE, not AI-dialogue, which is
# the contrast wanted -- but it is not a neutral baseline and is not reported as
# one. All rungs are family-independent, so they share cache across families
# with a common base, like chatml and raw.
LADDER = {
    "dyad_me_you":   "ME: {q}\nYOU:",
    "dyad_qa":       "Q: {q}\nA:",
    "dyad_user_ast": "User: {q}\nAssistant:",
    "dyad_human_ai": "Human: {q}\nAI:",
}


def templated_nosys(prompt, aligned_model_id):
    """Family template with the system block suppressed via an empty system msg.

    Works for ChatML-derived templates (collapses them to bare ChatML); llama
    ignores it and returns its system block unchanged. Callers compare against
    the native render and skip when identical.
    """
    from malign_logits.models import _load_tokenizer
    tok = _load_tokenizer(aligned_model_id)
    if not getattr(tok, "chat_template", None):
        return None
    try:
        return tok.apply_chat_template(
            [{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return None


def beams_for(model, tokenizer, text, n=N_BEAMS, depth=DEPTH):
    """Replicates beam.beam_storylines exactly; model stays loaded across cells.

    Returns payloads shape-identical to what annotate_beams stashes, so the
    entries are indistinguishable from normally-produced storylines.
    """
    from scipy.special import softmax as _softmax
    device = next(model.parameters()).device
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    plen = ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            ids, num_beams=n, num_return_sequences=n, max_new_tokens=depth,
            output_scores=True, return_dict_in_generate=True,
            length_penalty=0.0,  # sequences_scores = raw sum logprob
        )

    # per-beam per-position entropy, as in beam.beam_storylines
    beam_entropies = []
    if getattr(out, "scores", None) and getattr(out, "beam_indices", None) is not None:
        bidx = out.beam_indices.cpu().numpy()
        pos_h = []
        for pos in range(len(out.scores)):
            pr = _softmax(out.scores[pos].float().cpu().numpy(), axis=-1)
            pos_h.append(-np.sum(pr * np.log(pr + 1e-30), axis=-1))
        for i in range(len(out.sequences)):
            beam_entropies.append([
                round(float(pos_h[pos][bidx[i, pos]]), 3)
                for pos in range(len(out.scores))
            ])

    rows = []
    for i in range(len(out.sequences)):
        lp = out.sequences_scores[i].item()
        new = out.sequences[i][plen:]
        tids = new.tolist()
        rows.append({
            "text": tokenizer.decode(new, skip_special_tokens=True),
            "tokens": tids,
            "token_texts": [tokenizer.decode([t]).strip() for t in tids],
            "log_prob": lp,
            "path_prob": float(np.exp(lp)),
            "base_token_probs": [],
            "entropy": beam_entropies[i] if beam_entropies else [],
            "annotations": {},
        })
    return rows


def load_stashed(model_id, prompt_text, mode, n=N_BEAMS, depth=DEPTH, tmpl_src=None):
    """Return stashed rows for this cell, or None.

    tmpl_src disambiguates family-specific renders. The house convention keys on
    {model, prompt, mode}, which assumes ONE template per model. That is false
    here: llama, tulu and every tulu-sft variant share Llama-3.1-8B and feed it
    DIFFERENT templates under mode='chat'. Verified byte-identical beams before
    this fix -- the second family silently read the first's result. raw and
    chatml are family-independent and carry no tmpl field, so they keep sharing
    correctly, which is the duplication we DO want.
    """
    from malign_logits.probe import _get_cache
    key = {"model": model_id, "prompt": prompt_text,
           "n_beams": n, "max_tokens": depth,
           "type": "beam_annotated_v1"}
    if mode != "raw":
        key["mode"] = mode
    if tmpl_src:
        key["tmpl"] = tmpl_src
    return _get_cache().get_beams(key)


def stash_beams(model_id, prompt_text, mode, rows, n=N_BEAMS, depth=DEPTH, tmpl_src=None):
    """Write to the beams stash under the LOGITS convention (cache.py:133-150).

    The beams stash has no mode field: every one of its ~11.6k entries is raw,
    and a chat-mode write would silently overwrite the raw entry of the same
    prompt. That is a latent bug in beam_storylines, not a convention. malign
    mapped three different project conventions (generations folds mode into the
    model string; beam_words carries an explicit field; beams carries nothing)
    and the logits one is the house pattern because it is backward-compatible:

        key = {model, prompt}; if mode != "raw": key["mode"] = mode

    Only non-raw modes add the field, so existing raw entries keep their exact
    address. Adding it unconditionally would orphan all ~11.6k.

    Note the prompt keyed is the ORIGINAL QUESTION, not the rendered template,
    so a later load_cached_beams("Who are you?", mode="chat") finds it.
    """
    from malign_logits.probe import _get_cache
    key = {"model": model_id, "prompt": prompt_text,
           "n_beams": n, "max_tokens": depth,
           "type": "beam_annotated_v1"}
    if mode != "raw":
        key["mode"] = mode
    if tmpl_src:
        key["tmpl"] = tmpl_src   # see load_stashed
    _get_cache().set_beams(key, rows)


def run_arm(family, arm, model_id, cells, sink, tmpl_src):
    """Load the model ONLY if some cell is missing from the stash.

    The model load dominates runtime, so checking the stash after loading
    (as an earlier version did) made resume worthless.
    """
    live = [c for c in cells if c[2] is not None]
    cached = {(m, q): load_stashed(model_id, q, m,
                                   tmpl_src=(tmpl_src if m in ("chat", "chat_nosys") else None))
              for m, _, _, q in live}   # ladder+chatml+raw are family-independent
    missing = [c for c in live if cached[(c[0], c[3])] is None]
    print(f"  {family}/{arm}: {model_id} "
          f"[{len(live) - len(missing)}/{len(live)} cached]")

    model = tok = None
    if missing:
        model, tok = load_model(model_id)
    try:
        for mode, pkey, text, question in live:
            rows = cached[(mode, question)]
            if rows is None:
                rows = beams_for(model, tok, text)
                stash_beams(model_id, question, mode, rows,
                            tmpl_src=(tmpl_src if mode in ("chat", "chat_nosys") else None))
            for b in rows:
                broad, narrow = self_predicates(b["text"])
                sink.append({
                    "family": family, "arm": arm, "model_id": model_id,
                    "stage": stage_hint(model_id) if arm != "base" else "base",
                    "mode": mode, "prompt": pkey,
                    "is_identity": IS_IDENTITY[pkey],
                    "pclass": prompt_class(pkey),
                    "text": b["text"], "path_prob": b["path_prob"],
                    "log_prob": b["log_prob"],
                    "self_broad": broad, "self_narrow": narrow,
                })
    finally:
        if model is not None:
            del model
        gc.collect()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


SLOTS = ("base", "ego", "superego", "reinforced_superego")


def stage_hint(model_id):
    """Best-effort stage from the model ID. The registry SLOT is unreliable --
    tulu-sft-full keeps an SFT checkpoint in its `superego` slot -- so the slot
    is recorded separately and this is only a hint. model_id is the truth."""
    n = model_id.lower()
    for key, lab in (("rlvr", "rlvr"), ("-dpo", "dpo"), ("dpo", "dpo"),
                     ("-ppo", "ppo"), ("ppo", "ppo"), ("safe", "safety"),
                     ("-sft", "sft"), ("sft", "sft"),
                     ("instruct", "instruct"), ("chat", "chat"), ("-it", "instruct")):
        if key in n:
            return lab
    return "unknown"


def arms_for(fam):
    """Every distinct stage the family exposes, not just one collapsed 'aligned'.

    21 of 48 families have BOTH ego (SFT) and superego (DPO); the previous
    `superego or ego` silently dropped the SFT arm for all of them, including
    amber -- whose AmberChat-vs-AmberSafe contrast is the load-bearing evidence
    for the safety-data-style gradient, and the only way to answer H5.
    """
    out, seen = [], set()
    for slot in SLOTS:
        mid = getattr(fam, slot, None)
        if mid and mid not in seen:
            seen.add(mid)
            out.append((slot, mid))
    return out


def is_cached(model_id):
    """True only if weights are already local. Prevents a runaway download."""
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(model_id, local_files_only=True,
                          allow_patterns=["config.json"])
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="one tiny family only")
    ap.add_argument("--families", help="comma-separated family keys")
    ap.add_argument("--allow-download", action="store_true",
                    help="off by default; without it, uncached families are skipped")
    args = ap.parse_args()

    roster, skipped = [], []
    for key, fam in MODEL_FAMILIES.items():
        base = getattr(fam, "base", None)
        aligned = getattr(fam, "superego", None) or getattr(fam, "ego", None)
        if not (base and aligned):
            continue
        if key in EXCLUDE:
            skipped.append((key, "excluded by size"))
            continue
        if not args.allow_download and not (is_cached(base) and is_cached(aligned)):
            skipped.append((key, "weights not in local cache"))
            continue
        roster.append((key, base, aligned))
    if args.families:
        want = set(args.families.split(","))
        roster = [r for r in roster if r[0] in want]
    if args.smoke:
        tiny = [r for r in roster if r[0] in ("qwen-tiny", "smol")]
        roster = tiny[:1] or roster[:1]

    # ---- priority order: most informative first, so provisional results land early ----
    FIRST = "llama"   # RH: the May 11 talk used Llama, so it leads the queue
    H5 = {"amber", "beaver", "olmo", "olmo-tiny", "tulu"}   # carry the stage decomposition
    def size(mid):
        m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", mid)
        return float(m.group(1)) if m else 8.0
    seen_bases, ordered = set(), []
    scored = []
    for key, base, aligned in roster:
        fam = MODEL_FAMILIES[key]
        n_stages = len(arms_for(fam)) - 1
        scored.append((key, base, aligned, n_stages, size(base)))
    # tier A: H5 families (stage decomposition); tier B: new base model (base diversity);
    # tier C: other decomposable; tier D: rest. Within tier, smallest first.
    def tier(key, base, n_stages):
        if key == FIRST:
            return -1
        if key in H5:
            return 0
        if base not in seen_bases:
            return 1
        return 2 if n_stages > 1 else 3
    remaining = list(scored)
    while remaining:
        best = min(remaining, key=lambda r: (tier(r[0], r[1], r[3]), r[4]))
        remaining.remove(best)
        seen_bases.add(best[1])
        ordered.append(best[:3])
    roster = ordered
    print(f"roster: {len(roster)} families, priority-ordered")
    print("  first ten: " + ", ".join(k for k, _, _ in roster[:10]))
    sink = []
    for key, base, aligned in roster:
        tmpl = {p: templated(txt, aligned) for p, txt in ALL_PROMPTS.items()}
        if all(v is None for v in tmpl.values()):
            skipped.append((key, "aligned arm ships no chat template"))
            print(f"  SKIP {key}: no chat template on aligned arm")
            continue
        nosys = {p: templated_nosys(txt, aligned) for p, txt in ALL_PROMPTS.items()}
        nosys_dup = all(nosys.get(p) == tmpl.get(p) for p in ALL_PROMPTS)
        if nosys_dup:
            skipped.append((key, "chat_nosys identical to chat (no system block)"))
        # (mode, prompt_key, text_fed_to_model, original_question_for_stash_key)
        cells = ([("raw", p, txt, txt) for p, txt in ALL_PROMPTS.items()]
                 + [("chat", p, tmpl[p], ALL_PROMPTS[p]) for p in ALL_PROMPTS]
                 + [("chatml", p, CHATML.format(q=ALL_PROMPTS[p]), ALL_PROMPTS[p])
                    for p in ALL_PROMPTS]
                 + [(m, p, f.format(q=ALL_PROMPTS[p]), ALL_PROMPTS[p])
                    for m, f in LADDER.items() for p in ALL_PROMPTS])
        if not nosys_dup:
            cells += [("chat_nosys", p, nosys[p], ALL_PROMPTS[p]) for p in ALL_PROMPTS]
        for arm, mid in arms_for(MODEL_FAMILIES[key]):
            try:
                run_arm(key, arm, mid, cells, sink, tmpl_src=key)
            except Exception as e:
                skipped.append((f"{key}/{arm}", repr(e)[:120]))
                print(f"  FAIL {key}/{arm}: {repr(e)[:120]}")
        pd.DataFrame(sink).to_parquet(OUT_BEAMS, index=False, compression="zstd")  # incremental

    df = pd.DataFrame(sink)
    if df.empty:
        print("no rows")
        return
    df.to_parquet(OUT_BEAMS, index=False, compression="zstd")

    # P_self = share of the RETAINED BEAM SET, never an absolute probability.
    g = df.groupby(["family", "arm", "mode", "prompt", "pclass"])
    summary = g.apply(lambda d: pd.Series({
        "n_beams": len(d),
        "P_self_broad": d.loc[d.self_broad, "path_prob"].sum() / d["path_prob"].sum(),
        "P_self_narrow": d.loc[d.self_narrow, "path_prob"].sum() / d["path_prob"].sum(),
    }), include_groups=False).reset_index()
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"\nwrote {OUT_BEAMS} ({len(df)} rows), {OUT_SUMMARY} ({len(summary)} cells)")
    if skipped:
        print("\nskipped / failed:")
        for k, why in skipped:
            print(f"  {k}: {why}")
    print("\nP_self (broad), pooled over prompts:")
    piv = summary.pivot_table(index=["family", "arm"], columns=["mode", "pclass"],
                              values="P_self_narrow", aggfunc="mean")
    print(piv.round(3).to_string())
    print("\nEXCESS P_self (narrow, chat mode) = identity minus answerable control")
    w = summary[summary["mode"] == "chat"].pivot_table(
        index=["family", "arm"], columns="pclass", values="P_self_narrow", aggfunc="mean")
    w["excess"] = w.get("identity", 0) - w.get("control", 0)
    print(w.round(3).to_string())


if __name__ == "__main__":
    main()
