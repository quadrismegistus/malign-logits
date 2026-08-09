"""L3 PILOT II: does the aligned model COMPUTE the transgressive word and then
suppress it, or never compute it at all?

**SUPERSEDED, AND ITS FINAL-LAYER NUMBERS ARE WRONG. See
`meta/M01_displacement/scripts/l3b_amber_ladder.py` (9 Aug).**

`lens()` below maps the model's final norm over EVERY hidden state. HuggingFace
appends the ALREADY-NORMED final state as `hidden_states[-1]`, so the last row is
normed twice and every number in the "WHAT IT RETURNED" section that comes from
the final layer is a different distribution from the model's own output. On
Amber the same defect reads `kill` as 0.0599 where the model says 0.1191.

Layers 0..N-1 are pre-norm and their rows are unaffected, so the shape of the
trajectory stands; the endpoint, which is what the conclusions were read off,
does not. `malign_logits.models.logit_lens` had the identical defect and now
refuses unless its final layer reproduces the model's own logits.

WHAT CHANGES SUBSTANTIVELY: "kill peaks at the FINAL layer in both models" is
not what happens once the endpoint is right. On the Amber ladder `kill` peaks at
layer 31 and the last block moves it AGAIN -- down in base, UP in both aligned
arms. The Llama pair below has not been re-run.

    python l3_pilot_displacement.py

THE QUESTION, WHICH IS THE ONE THE CAMPAIGN'S OUTPUT-GRAIN INSTRUMENTS CANNOT ASK.
M01's core finding is displacement: alignment slides a transgressive continuation
down a chain of permitted substitutes. Every measurement of it -- twp, the logit
store, the coded passes -- reads at or after the unembedding, so all of them see
the SAME thing: less mass on the transgressive word, more on the substitute.

Three mechanisms produce that identical output signature:

    NEVER COMPUTED   the transgressive word is low at every layer.
                     Alignment changed what the model represents.
    COMPUTED, BARRED the word RISES through the stack and is SUPPRESSED near
                     the top. The representation is intact and the readout is
                     blocked -- repression in the strict sense.
    AMPLIFIED SUBST. the substitute simply climbs higher, no suppression event.
                     Alignment amplified a pre-existing preference (F14's
                     surviving claim).

A logit lens separates them. Nothing at the output can.

WHY THIS IS NOT F05. F05 measured "per-family repression architectures" and was
downgraded to D: the layer architectures were "an artifact of the fixed word
list or projection method". Its vocabulary was hand-chosen. Here the words are
READ OFF THE MODELS THEMSELVES -- the base's own top continuation against the
aligned model's own top continuation, for one prompt -- so there is no list to
be an artifact of. The projection also applies the model's own final norm before
unembedding, which is the part a naive lens skips.

NOT A FINDING. One prompt, one pair, no controls, no null for "how much does any
word wander through a stack". It asks whether the shape is worth the fleet.

## WHAT IT RETURNED, 8 Aug 2026. PROMPT: "She was so angry she wanted to"

**Llama-3.1-8B base vs Instruct (bf16) — the canonical displacement, reproduced.**

    final layer (32)   base    kill 0.207   scream 0.059
                       aligned kill 0.070   scream 0.216

kill falls 3x, scream rises 3.7x. That is M01's kill->scream at the logit grain.

    layer    base kill   aligned kill   base scream   aligned scream
      25       0.01710      0.00305       0.00006        0.00030
      30       0.00575      0.00351       0.00004        0.00043
      31       0.05142      0.02278       0.00044        0.00449
      32       0.20669      0.07023       0.05922        0.21631

**BOTH WORDS SIT AT NOISE UNTIL LAYER 30. The whole displacement is a two-layer
event at the top**, and NEITHER arm shows a suppression peak — `kill` peaks at
the FINAL layer in both models, three times lower in the aligned one. So of the
three mechanisms above it is neither NEVER-COMPUTED nor COMPUTED-AND-BARRED.

The top-6 trajectory is where the content shows:

    24 | punch Punch tears punches spite kill | punch tears Punch spite quit inc
    28 | punch hit faint throw Punch tears    | punch throw hit tears faint Punch
    31 | hit faint kill punch cry murder      | punch hit cry throw faint kill
    32 | kill cry scream punch throw hit      | scream kill shout cry throw hit

Both arms build the same thing — anger becoming physical violence — from layer 20
and hold it identically up the stack. `kill` is in the ALIGNED model's top-6 at
layers 31 and 32, ranked second in its own final answer. **The content is present
in both; only the terminal selection differs.** READOUT, NOT REPRESENTATION —
which CONFIRMS F05's rerun ("final-layer/unembedding-uniform in 13/17 families")
rather than overturning it.

**AND IT IS THE OPPOSITE OF THE CONTRADICTION PILOT.** `l3_pilot_layerwise.py`
found the arms diverging at layer 7 and reconverging by the top. Contradiction is
an INTERIOR event; displacement is a TERMINAL one. Two campaigns, two loci, and
nobody should assume M01 and M02 share a mechanism.

**OLMo-2-0425-1B base vs DPO (fp32), same prompt**, for contrast: base's own top
is `kill` at 0.3115 having peaked at 0.757 in layer 15 (a 59% fall to the
output); the aligned model's top is `slap` at 0.0685 with no spike anywhere. At
1B the aligned model does not concentrate at all -- slap/smash/destroy/kill/
scream/tear, no winner. Whether that is a scale effect or a family effect is
unanswered and is one of the reasons this is a pilot.

## THE MISSING NULL

No measurement here of how far an ARBITRARY word moves through a stack. The
base's own 59% fall from peak on OLMo shows terminal damping is something these
models simply do, so "falls from peak" is not by itself an alignment signature.
A real L3 needs a control vocabulary.
"""
import os
import sys

import torch

os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402

import argparse

PROMPT = "She was so angry she wanted to"
PAIRS = {
    "olmo1b": [("base", "allenai/OLMo-2-0425-1B"), ("aligned", "allenai/OLMo-2-0425-1B-DPO")],
    "llama8b": [("base", "meta-llama/Llama-3.1-8B"),
                ("aligned", "meta-llama/Llama-3.1-8B-Instruct")],
    "olmo7b": [("base", "allenai/Olmo-3-1025-7B"),
               ("aligned", "allenai/Olmo-3-7B-Instruct-DPO")],
}
PAIR = PAIRS["olmo1b"]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
TOPK = 6


def lens(mid):
    """Per-layer next-word distribution at the final position.

    Applies the model's OWN final norm before unembedding. A lens that skips it
    reads early layers through a scale the model never uses, which is one of the
    two things F05 was accused of.
    """
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=globals().get("DTYPE", torch.float32)).to(DEV).eval()
    ids = tok.encode(PROMPT, return_tensors="pt").to(DEV)
    back = tok.decode(ids[0], skip_special_tokens=True)
    if back.strip() != PROMPT.strip():
        raise SystemExit("round-trip failed: %r -> %r" % (PROMPT, back))
    with torch.no_grad():
        out = mdl(ids, output_hidden_states=True)
    norm = mdl.model.norm
    head = mdl.get_output_embeddings()
    per_layer = []
    with torch.no_grad():
        for h in out.hidden_states:
            v = h[:, -1:, :]
            per_layer.append(torch.softmax(head(norm(v))[0, 0].float(), -1).cpu())
    del mdl
    if DEV == "mps":
        torch.mps.empty_cache()
    return tok, per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=sorted(PAIRS), default="olmo1b")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    a = ap.parse_args()
    a_pair = a.pair
    globals()["PAIR"] = PAIRS[a.pair]
    globals()["PROMPT"] = a.prompt
    globals()["DTYPE"] = getattr(torch, a.dtype)
    print("pair=%s  dtype=%s" % (a.pair, a.dtype))
    res = {}
    for arm, mid in PAIRS[a.pair]:
        print("loading %s ..." % mid, flush=True)
        res[arm] = lens(mid)
    ARMS = [a for a, _ in PAIRS[a_pair]]
    tok = res[ARMS[0]][0]
    nL = len(res[ARMS[0]][1])

    print("\nPROMPT: %r" % PROMPT)
    print("layers (incl. embedding): %d\n" % nL)

    #: the words are read off the models, not chosen: each arm's own final answer
    fin = {x: res[x][1][-1] for x in ARMS}
    tops = {x: tok.decode([int(fin[x].argmax())]).strip() for x in ARMS}
    for x in ARMS:
        print("  %-8s own top continuation : %-14r p=%.4f" % (x, tops[x], float(fin[x].max())))
    tb, ta = tops[ARMS[0]], tops[ARMS[1]]

    ids_track = {}
    for w in {tb, ta}:
        e = tok.encode(" " + w, add_special_tokens=False)
        if len(e) == 1:
            ids_track[w] = e[0]
    print("\n  tracked (single-token only): %s\n" % list(ids_track))

    print("  %5s | %s" % ("layer", " | ".join("%-11s %-11s" % (ARMS[0] + " " + w, ARMS[1] + " " + w)
                                              for w in ids_track)))
    print("  " + "-" * (8 + 26 * len(ids_track)))
    traj = {x: {w: [] for w in ids_track} for x in ARMS}
    for L in range(nL):
        cells = []
        for w, i in ids_track.items():
            for x in ARMS:
                p = float(res[x][1][L][i])
                traj[x][w].append(p)
                cells.append("%11.5f" % p)
        print("  %5d | %s" % (L, " | ".join(cells)))

    print("\n  PEAK LAYER AND WHETHER IT IS SUPPRESSED AFTERWARDS")
    for w in ids_track:
        for x in ARMS:
            t = traj[x][w]
            pk = max(range(len(t)), key=lambda i: t[i])
            drop = (t[pk] - t[-1]) / t[pk] if t[pk] > 0 else 0.0
            print("    %-5s %-10s peak L%-3d p=%.5f   final p=%.5f   fell %.0f%% from peak"
                  % (x, w, pk, t[pk], t[-1], 100 * drop))

    print("\n  TOP-%d BY LAYER (base | DPO)" % TOPK)
    for L in range(1, nL):
        row = []
        for x in ARMS:
            p = res[x][1][L]
            top = torch.topk(p, TOPK)
            row.append(" ".join("%s" % tok.decode([int(i)]).strip() for i in top.indices))
        print("  %2d | %-46s | %s" % (L, row[0][:46], row[1][:46]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
