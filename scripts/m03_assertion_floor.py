"""ASSERTION FLOOR CHECK — does any individual stem foreclose a forceful reply?

Commissioned at [1966]. Declared at [1967] before running; authorized at [1965].

    .venv/bin/python scripts/m03_assertion_floor.py

WHY THIS ONE CONSTRAINT GETS A CHECK AT ALL. Of M03's six drafting
constraints, five protect against a NULL — violated, they cost power or
interpretability. **THE SIXTH, `the individual prompt leaves assertion live`,
PROTECTS AGAINST A FALSE POSITIVE: violated, it MANUFACTURES the finding out of
the stem, in the arm the finding is about** ([1962].3, [1963].1). A validity
class, not a caveat class.

F21's origin ([1960].1) is a user report: the model *"explains why the
institution might be justified, coaches tone, and steers from asserting a right
to politely asking"* — *"you go in ready to push back on an unfair billing
charge and come out drafting a polite inquiry."* **If a stem admits only a
deferential continuation, the steering is in the prompt.**

TWO INSTRUMENTS, BECAUSE THEY FAIL DIFFERENTLY.

**(A) CONTINUATION ENTROPY** — the pen's preferred route ([1965].2), and it
needs no vocabulary at all. A stem that forecloses has few live continuations.
**It cannot distinguish `everything is foreclosed` from `assertion is
foreclosed`** — fifty live deferential continuations and no forceful one is
high-entropy and maximally foreclosed for this purpose.

**(B) SPEECH-ACT FORCE MASS** — and the axis is NOT assertive-vs-procedural.
`File a complaint with the regulator` is PROCEDURAL AND HIGH-FORCE; `reach out
and ask them to reconsider` is NON-PROCEDURAL AND LOW-FORCE. **An
assertive/procedural split misclassifies both, and it would score M03's OUTCOME
VARIABLE while claiming to check the stem** — the conflation constraint (ii)
exists to prevent. The axis is the FORCE OF THE SPEECH ACT: demand against
request, assert a right against ask a favour.

**LIST PROVENANCE, which is the whole risk.** Written from the user report's
own wording, by custody and not by the seat that drafted the stems ([1965].2 —
a list written by the drafter is a list written to pass), **not read off any
model output, and not derived from the stems' scores.** That is C2's
derivation-bound defect avoided by construction ([1704].1: a set defined as
words-that-rise cannot fail on its derivation population). **This one is
defined from the HYPOTHESIS and it can fail.** Declared in full at [1967]
before the run. Heuristic, never targeted, **never reused as an outcome
instrument downstream.**

**COVERAGE IS PRINTED BEFORE ANY RATIO** ([1969].2/[1970].1). The lists total
37 words; the most probable continuations after a grievance stem are `file`,
`contact`, `call`, `write`, `send`, `take` — **and none is on either list,
because `file a complaint` is HIGH force and `file the paperwork` is LOW: the
force is carried by the OBJECT, not the verb, so it is unlistable at the first
content token.** **A NEAR-ZERO HIGH SCORE OVER MASS THE INSTRUMENT CANNOT SEE
MEANS THE LIST MISSED, NOT THAT THE STEM FORECLOSED**, and only the coverage
number separates those. The top continuations are printed per stem so the
missed vocabulary is visible rather than inferred.

**STEM EXCLUDED BY CONSTRUCTION.** Scoring reads the next-token distribution
only; the stem is never matched against. `appeal`, `dispute` and `stop` occur
in the drafted stems in other senses, and matching by token id rather than
substring means the `file`-inside-`issued` family cannot fire here.

**THE LIST IS NOT GROWN TOWARD COVERAGE** ([1970].1): a list grown to cover the
mass becomes a proceduralization lexicon, which is the thing the axis
correction forbids. Small, hypothesis-derived, plus its own coverage number.

FLAG = UNION, NOT INTERSECTION, and the asymmetry sets the rule: **a false flag
costs the drafter twenty minutes and a missed one manufactures the finding.**
**Low coverage can only ADD flags, never suppress them** ([1970].2), so the cut
survives it: ambiguous stems get inspected with the coverage number beside them.

NO THRESHOLD IS PROPOSED HERE. A cut chosen by the seat reporting the result is
a selection rule wearing a floor's name. **The full ranked distribution of all
18 is printed on both statistics and the cut is the pen's.** A flagged stem is
flagged for INSPECTION, not convicted ([1965].2); the conviction and the
rewrite are the drafter's.

WHAT IT CANNOT DO. It measures whether the BASE MODEL finds forceful
continuations available. **It does not measure whether a HUMAN would find
assertion sensible**, and the two come apart wherever the assertive reading
rests on world knowledge the base model carries weakly. **It nominates; it
cannot discharge RH's construct read.**
"""

import argparse
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "allenai/Olmo-3-1025-7B"

#: Declared in full at [1967] BEFORE the run, so both seats could attack the
#: list rather than the result. Force of the speech act, not bureaucratic form.
HIGH_FORCE = ("demand", "insist", "refuse", "dispute", "challenge", "contest",
              "assert", "push", "escalate", "sue", "report", "withhold", "deny",
              "object", "tell", "stop", "complain", "appeal", "threaten")
LOW_FORCE = ("ask", "request", "inquire", "clarify", "understand", "reach",
             "follow", "politely", "apologise", "wait", "hope", "try", "see",
             "consider", "check", "confirm", "explain", "discuss")


def first_token_ids(tok, words):
    """id -> word, for the FIRST token of ' word'. Collisions are reported, not
    silently merged: two words sharing a first token would double-count."""
    out = {}
    for w in words:
        i = tok(" " + w, add_special_tokens=False).input_ids[0]
        out.setdefault(i, []).append(w)
    return out


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

    stems = [(k["id"], k["domain"], mk.build(k)["indiv_I_final"]) for k in ALL]
    print(f"ASSERTION FLOOR CHECK — {args.model}")
    print(f"{len(stems)} INDIVIDUAL stems, marker-final. Base model only.\n")
    print("NO THRESHOLD IS PROPOSED. Full ranked distribution on both "
          "statistics;\nthe cut is the pen's, and a flagged stem is flagged "
          "for INSPECTION.\n")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16).to(device).eval()

    hi, lo = first_token_ids(tok, HIGH_FORCE), first_token_ids(tok, LOW_FORCE)
    clash = set(hi) & set(lo)
    print(f"list -> first tokens: {len(hi)} high ids for {len(HIGH_FORCE)} words, "
          f"{len(lo)} low ids for {len(LOW_FORCE)} words")
    if clash:
        print(f"  *** {len(clash)} id(s) in BOTH lists — reported, not merged: "
              f"{[(hi[i], lo[i]) for i in clash]}")
    print()

    rows = []
    for sid, dom, stem in stems:
        ids = tok(stem, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()
        p = torch.softmax(logits, dim=-1).cpu()
        ent = -(p[p > 0] * p[p > 0].log()).sum().item()
        h = sum(p[i].item() for i in hi)
        l = sum(p[i].item() for i in lo)
        top = torch.topk(p, 8)
        rows.append((sid, dom, ent, h, l,
                     h / (h + l) if (h + l) > 0 else float("nan"), h + l,
                     [(tok.decode([i]).strip(), round(v.item(), 3))
                      for v, i in zip(top.values, top.indices)]))

    print("COVERAGE FIRST ([1969].2) — what fraction of the continuation the "
          "instrument\ncan see at all. A near-zero HIGH over mass the lists "
          "cannot see means the\nLIST MISSED, not that the stem foreclosed.\n")
    print(f"  {'scenario':<10}{'coverage':>10}   top continuations (mass)")
    for r in sorted(rows, key=lambda x: -x[6]):
        tops = "  ".join(f"{w}:{v}" for w, v in r[7][:6])
        print(f"  {r[0]:<10}{r[6]:>10.4f}   {tops}")
    cov = [r[6] for r in rows]
    print(f"\n  coverage range {min(cov):.4f} - {max(cov):.4f}, "
          f"median {sorted(cov)[len(cov)//2]:.4f}\n")

    for label, key in (("(A) CONTINUATION ENTROPY — lowest first", 2),
                       ("(B) HIGH-FORCE MASS — lowest first", 3)):
        print(f"{label}")
        print(f"  {'scenario':<10}{'domain':<12}{'entropy':>9}{'high':>10}"
              f"{'low':>10}{'share':>8}{'coverage':>10}")
        for r in sorted(rows, key=lambda x: x[key]):
            print(f"  {r[0]:<10}{r[1]:<12}{r[2]:>9.3f}{r[3]:>10.5f}"
                  f"{r[4]:>10.5f}{r[5]:>8.3f}{r[6]:>10.4f}")
        print()

    out = os.path.join(ROOT, "data", "m03_assertion_floor.csv")
    with open(out, "w") as f:
        f.write("scenario_id,domain,entropy,high_force_mass,low_force_mass,"
                "share,coverage,top_continuations\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]},"
                    f"\"{' '.join(w for w, _ in r[7])}\"\n")
    print(f"wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
