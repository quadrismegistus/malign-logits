"""REGRET DISCRIMINATOR — is `I should have` the stem, or is it English?

Commissioned at [1980], from lacan's [1978].3.

    .venv/bin/python scripts/m03_regret_discriminator.py

THE QUESTION. Three individual stems put over half their continuation mass on
` have` — C4 0.621, U3 0.588, C1 0.555. **`I should HAVE [done X]` is
RETROSPECTIVE REGRET: the speaker reproaching a past failure rather than
deciding what to do next. If that is the dominant reading, THE DELIBERATION IS
FORECLOSED and there is nothing for alignment to steer** — a foreclosure of a
different kind from the one the floor check was built for, and one neither
instrument was designed to see.

**A FREQUENCY EFFECT AND A FORECLOSURE LOOK IDENTICAL AT THIS RESOLUTION.**
`should have` is a very common bigram. What separates them is whether the mass
sits on ` have` because of the STEM or because of ENGLISH.

THE DISCRIMINATOR, and it is lacan's: **the INSTITUTIONAL arm of the same
scenario. Same bigram frequency, same marker, different past act.**

    institutional `have` mass ≈ individual     -> ENGLISH. The bigram carries
                                                  it and the stems are innocent.
    institutional `have` mass << individual    -> THE STEM. The individual
                                                  arm's grievance structure is
                                                  producing the regret reading.

**PAIRED BY SCENARIO**, because the arms share the scenario, the frame and the
marker and differ in the thing under test — the same pairing the register
probes used, and for the same reason: the unpaired comparison mixes the arm
contrast with which scenarios sit in each.

WHY THE PAST ACT IS SUSPECTED, AND IT IS A CONSTRAINT COLLISION ([1978].3).
Constraint (iii) needs the SPEAKER INSIDE THE CLAUSE so there is something to
pluralise; the natural way to put them there is a PAST ACT — *the complaint I
filed*, *the evidence I sent* — **which is exactly what licenses `I should HAVE
filed it sooner`.** And a grievance needs a prior attempt or there is no
conflict yet. **So the past act is doubly forced, by the construct and by
(iii).** Third collision involving (iii), after (i)'s challenger-as-agent and
the evaluative-stative slot.

ON MARKING THE NINE. Lacan counts nine of eighteen individual stems containing
a completed act by the speaker. **This file does NOT assert that count. It
extracts `I <past-verb>` spans mechanically AND PRINTS THEM, so the marking is
checkable rather than taken** — and it prints its own disagreement with the
hand count instead of quietly matching it. **A mechanical marker over English
prose is a heuristic; the spans are the evidence and the count is not.**
"""

import argparse
import os
import re
import statistics
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "allenai/Olmo-3-1025-7B"

#: Speaker + past-tense verb. Regular `-ed` plus the irregulars that actually
#: occur in grievance prose. Printed, never trusted as a count.
IRREG = ("sent", "wrote", "made", "gave", "paid", "kept", "took", "put",
         "told", "left", "lost", "got", "brought", "spoke", "went", "had")
PAST_ACT = re.compile(
    r"\b(?:I|we)\s+((?:\w+ed)|(?:" + "|".join(IRREG) + r"))\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mk", os.path.join(ROOT, "meta/M03_proceduralization/m03_kernel.py"))
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    ALL = mk.KERNELS + mk.CONVERSIONS + getattr(mk, "UNANCHORED", [])

    print(f"REGRET DISCRIMINATOR — {args.model}\n")
    print("DECLARED READOUT ([1978].3, before this ran):")
    print("  institutional ~ individual   -> ENGLISH, the bigram carries it")
    print("  institutional << individual  -> THE STEM, the grievance structure "
          "does\n")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16).to(device).eval()
    have = tok(" have", add_special_tokens=False).input_ids[0]

    def p_have(text):
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            p = torch.softmax(model(ids).logits[0, -1].float(), dim=-1).cpu()
        return p[have].item()

    rows = []
    for k in ALL:
        c = mk.build(k)
        spans = [m.group(0) for m in PAST_ACT.finditer(k["indiv_sg"])]
        rows.append((k["id"], k["domain"], p_have(c["indiv_I_final"]),
                     p_have(c["inst_I_final"]), spans))

    print("PAST-ACT SPANS — extracted, PRINTED, and not asserted as a count.")
    for sid, dom, _, _, spans in rows:
        print(f"  {sid:<10}{dom:<12}{spans if spans else '—'}")
    n_mark = sum(1 for r in rows if r[4])
    print(f"\n  mechanical count {n_mark} of {len(rows)}; lacan's hand count is 9.")
    if n_mark != 9:
        print(f"  *** THE TWO DISAGREE BY {abs(n_mark - 9)}. The spans above are "
              f"the evidence;\n      neither count is authoritative and the "
              f"marking is a heuristic over prose.\n")
    else:
        print("  They agree, which is weak evidence — one regex and one reader "
              "can\n  agree by sharing an assumption about what counts.\n")

    print(f"  {'scenario':<10}{'domain':<12}{'indiv':>9}{'inst':>9}"
          f"{'indiv-inst':>12}   past-act")
    for sid, dom, iv, it, spans in sorted(rows, key=lambda x: -(x[2] - x[3])):
        print(f"  {sid:<10}{dom:<12}{iv:>9.4f}{it:>9.4f}{iv - it:>12.4f}"
              f"   {'yes' if spans else 'no'}")

    d = [r[2] - r[3] for r in rows]
    pos = sum(1 for x in d if x > 0)
    print(f"\nPAIRED n={len(d)}: mean {statistics.mean(d):+.4f}  "
          f"median {statistics.median(d):+.4f}  {pos} of {len(d)} positive")
    print(f"  individual mean {statistics.mean([r[2] for r in rows]):.4f}   "
          f"institutional mean {statistics.mean([r[3] for r in rows]):.4f}")
    for tag, sub in (("past-act stems", [r for r in rows if r[4]]),
                     ("no past act", [r for r in rows if not r[4]])):
        if sub:
            print(f"  {tag:<16} n={len(sub):<3} indiv {statistics.mean([r[2] for r in sub]):.4f}"
                  f"   inst {statistics.mean([r[3] for r in sub]):.4f}"
                  f"   diff {statistics.mean([r[2] - r[3] for r in sub]):+.4f}")

    out = os.path.join(ROOT, "data", "m03_regret_discriminator.csv")
    with open(out, "w") as f:
        f.write("scenario_id,domain,indiv_have,inst_have,diff,past_act\n")
        for sid, dom, iv, it, spans in rows:
            f.write(f"{sid},{dom},{iv},{it},{iv - it},{bool(spans)}\n")
    print(f"\nwrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
