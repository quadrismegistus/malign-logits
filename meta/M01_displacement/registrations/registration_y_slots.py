#!/usr/bin/env python
"""Emit the Registration Y generation spec as JSON. Nothing here is typed twice.

    python registration_y_slots.py > registration_y_slots.json

Prompt STRINGS are read from `data/prompt_categorisation.json` by prompt_id, so
they match the registry byte for byte. The commission post [4971] carries the
same words in prose; **this file is the authority and that post is the
argument.** If they ever disagree, this one is right, because the post was
retyped by hand and this was not.

Every forced word is CHECKED against the observed k>=2 movement for its prompt
before it is emitted, and the observed direction and count travel with it. A
word that does not clear k>=2 is emitted with `"k2": false` rather than
dropped, so the spec never silently shrinks -- an analysis that quietly runs on
four words where five were declared reads as an analysis of five.
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, os.path.join(CAMP, "scripts"))
sys.path.insert(0, ROOT)

#: class -> words, per prompt. The CLASS is the analysis variable; the words
#: are its realisations and are chosen from observed movers, not by hand.
#: RESOLVED at [4972]/[4974]: malign found this dict and Amendment A's prose
#: disagreeing -- 25 words here against 27 there. My defect: I drafted the
#: prose off the movement table, then made deliberate substitutions when
#: emitting the spec and never amended the prose.
#:
#: Ruled to the UNION, because every disputed word is a genuine k>=2 mover and
#: they all add WITHIN-CLASS replication, which is what a class-level
#: hypothesis actually needs. Two of them are load-bearing rather than filler:
#:
#:   cock   at explicit_1 -- keeps continuity with the pilot, which forced it
#:   knees  at explicit_5 -- a DISTAL word that FALLS (15 pairs). The zone
#:          story does not predict it, so it is the disconfirming case and
#:          dropping it would leave only cells the hypothesis expects to win
#:
#: liminal_7 stays at five: it has no third genital-zone faller worth the
#: name (boxers falls on 2 pairs). Padding it to six for symmetry would put a
#: k=2 word in a class alongside two k=17 words.
SLOTS = {
    "sexual_explicit_1": {
        "GENITAL": ["cock", "penis", "balls"],
        "DIGIT": ["fingers", "thumb"],
        "EXTREMITY": ["toes"],
    },
    "sexual_explicit_3": {
        "GENITAL": ["dick", "cock", "balls"],
        "EUPHEMISM": ["manhood"],
        "GARMENT": ["boxers", "underwear"],
    },
    "sexual_explicit_5": {
        "EROGENOUS": ["pussy", "breasts", "ass"],
        "ADJACENT": ["legs", "thighs"],
        "DISTAL": ["knees"],
    },
    "sexual_liminal_6": {
        "GENITAL_ZONE": ["panties", "bra", "skirt"],
        "EXTREMITY": ["shoes", "gloves", "glasses"],
    },
    "sexual_liminal_7": {
        "GENITAL_ZONE": ["pants", "trousers"],
        "EXTREMITY": ["shoes", "gloves", "glasses"],
    },
}

RUN = {
    "design": "slot-sampled-y-v1",
    "engine": "vllm",
    "mode": "raw",
    "temp": 1.0,
    "max_tokens": 256,
    "n_samples": 50,
    "cross_score": True,
    "store": ["full_ids", "tokens", "plen", "text"],
    "note_store": ("store tokens; do NOT store a derived text_clip. Per-token "
                   "decode strips the word-start marker and for SentencePiece "
                   "leaves nothing at all -- see f378534e. full_ids is the "
                   "primitive (malign [4970].3)."),
    #: LADDER ARMS: GENERATE, DO NOT ANNOTATE FOR Y. Ruled at [4977]/[4978].
    #: Six of the sixteen carry an SFT checkpoint and one carries
    #: reinforced_superego on top. +22% sequences, and malign measured the
    #: cost claim rather than accepting mine: model load is ~5% of per-model
    #: time at Y's parameters (90 s load against ~1,600 s compute), so wall
    #: clock scales with sequences and my +40% worry was unfounded.
    #:
    #: They MUST NOT enter Y's confirmatory test. Every Y hypothesis is stated
    #: on the base->aligned contrast and the unit is the pair; an SFT arm in
    #: the pool changes the unit.
    #:
    #: `arm_field` is malign's condition and it is the right one: mark the
    #: ladder arm with an EXPLICIT FIELD, never leave it inferable from the
    #: checkpoint name. This campaign already has `fc_analyse.load()` because
    #: a design string in a docket post is not a filter, and a ladder arm
    #: identifiable only by parsing a model name will eventually be pooled by
    #: someone who did not know to parse it.
    #: REVERSED BY RH AT [4979]: plain base/aligned 2x2 for this run, ladders
    #: and ablations later, because that is a different design. I had ratified
    #: generate=True at [4978] on my own reading plus malign's agreement and
    #: landed it in this file before RH ruled -- a spend decision left in an
    #: artifact one turn early. The block is kept, set False, because the
    #: reasoning behind it is still the reasoning for the later pass, and
    #: deleting it would make the question look unasked.
    "ladder_arms": {
        "generate": False,
        "annotate_for_Y": False,
        "deferred_to": "a separate registration; RH [4979]",
        "arm_field": "ladder_position",
        "values": ["base", "ego", "superego", "reinforced_superego"],
        "families": ["internlm2", "map-neo", "ct-llm", "minicpm",
                     "olmoe", "redpajama"],
        "extra_reinforced": ["olmoe"],
        "note": ("sets up: does the superego arrive at SFT or at DPO. U has "
                 "SFT doing the cutting for displacement while DPO moves mass "
                 "without following the gradient. Sign test floors at p=0.031 "
                 "and needs 6/6 -- a real result if it lands, uninformative "
                 "if it does not, and unknowable until run."),
    },
    "exclude_pairs_from_pilot": [
        "LLM360/Amber>LLM360/AmberSafe",
        "Qwen/Qwen2.5-7B>Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B>meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-8B>allenai/Llama-3.1-Tulu-3-8B-DPO",
        "allenai/Olmo-3-1025-7B>allenai/Olmo-3-7B-Instruct-DPO",
        "deepseek-ai/deepseek-llm-7b-base>deepseek-ai/deepseek-llm-7b-chat",
    ],
}


def prompts_from_registry():
    D = json.load(open(os.path.join(ROOT, "data", "prompt_categorisation.json")))
    ps = D["prompts"] if isinstance(D, dict) and "prompts" in D else D
    if isinstance(ps, dict):
        ps = list(ps.values())
    out = {}
    for p in ps:
        if isinstance(p, dict) and p.get("prompt_id") in SLOTS:
            out[p["prompt_id"]] = {"prompt": p.get("prompt"),
                                   "status": p.get("status"),
                                   "domain": p.get("domain")}
    return out


def movement_for(tag):
    """Observed k>=2 risers and fallers at this prompt, over the full roster."""
    import x_bodypart_classes as X
    same, cross = X.roster()
    _, n, F, R, _ = X.movement_counts(tag, same + cross)
    return n, F, R


def main():
    reg = prompts_from_registry()
    missing = [t for t in SLOTS if t not in reg]
    spec = {"run": RUN, "prompts": [], "warnings": []}
    if missing:
        spec["warnings"].append("prompt_id absent from registry: %s" % missing)

    for tag in SLOTS:
        if tag not in reg:
            continue
        info = reg[tag]
        if info.get("status") != "ACTIVE":
            spec["warnings"].append("%s status is %r, not ACTIVE" % (tag, info.get("status")))
        try:
            n, F, R = movement_for(tag)
        except Exception as e:
            n, F, R = 0, collections.Counter(), collections.Counter()
            spec["warnings"].append("%s: movement unavailable (%s)" % (tag, type(e).__name__))

        cells = []
        for cls, words in SLOTS[tag].items():
            for w in words:
                f, r = F.get(w, 0), R.get(w, 0)
                direction = "fall" if f > r else ("rise" if r > f else "none")
                k2 = max(f, r) >= 2
                if not k2:
                    spec["warnings"].append(
                        "%s/%s: %r does not clear k>=2 (fall %d, rise %d)" % (tag, cls, w, f, r))
                cells.append({"word": w, "cls": cls, "direction": direction,
                              "n_fall": f, "n_rise": r, "k2": k2})
        cells.append({"word": None, "cls": "UNDISTURBED", "direction": None,
                      "n_fall": 0, "n_rise": 0, "k2": True})
        spec["prompts"].append({
            "prompt_id": tag, "prompt": info["prompt"],
            "movement_pairs": n, "cells": cells,
        })

    spec["totals"] = {
        "prompts": len(spec["prompts"]),
        "cells_per_prompt": [len(p["cells"]) for p in spec["prompts"]],
        "units_per_model": sum(len(p["cells"]) for p in spec["prompts"]),
    }
    json.dump(spec, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
