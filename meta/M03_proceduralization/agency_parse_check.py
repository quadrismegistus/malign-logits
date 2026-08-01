"""FAILED INSTRUMENT, KEPT BECAUSE IT FAILS. A dependency parse cannot measure agency.

Commissioned at [1898].6 as "the base model's agency score on the 36 marker-free
cells — blind-safe, minutes of compute." Withdrawn at [1900] because the
instrument it assumed does not exist. This is the attempt, kept so the next seat
reaching for the obvious cheap route does not spend the same twenty minutes.

    .venv/bin/python meta/M03_proceduralization/agency_parse_check.py

THE QUESTION IT WAS BUILT FOR. [1897] classified the 18 M03 scenarios by hand and
found agent-role and institutionality covarying in 15 of 18 — an attribution limit
on the SPEAKER contrast, and one that bears on F21's "agency RISES in every family"
because M03 is the registered test of the reading that rests on it. Lacan asked for
an independent check, correctly noting that the classification was one reader's and
that the reader had an interest.

WHAT THIS BUILDS. A mechanical version: parse each cell, score the speaker pronoun
as AGENT if it holds nsubj/poss and PATIENT if it holds dobj/pobj/nsubjpass. No
model, no annotation, no word list — which is what an independent check of a hand
classification should be.

WHAT IT RETURNS, AND WHY THE NUMBER IS WORTHLESS.

    arm       agent  patient   both  absent    n
    indiv        12        2      2       2   18
    inst         13        0      1       4   18
                                    separation +1

Against a hand reading of 15 of 18, that looks like a refutation. IT IS NOT A
DISAGREEMENT — IT IS A DIFFERENT QUANTITY, and one cell shows why:

    "The officer slammed me ... even though I never resisted."
        'me'  dep=dobj    head='slammed'
        'I'   dep=nsubj   head='resisted'    <- NEGATED

`I never resisted` parses as nsubj, so this scores the cell AGENTIVE. Semantically
the speaker is precisely the one things were DONE TO, and the only clause where
they hold subject position is a NEGATED NON-ACTION.

**GRAMMATICAL SUBJECTHOOD IS NOT AGENCY. Negation and stative predicates make them
orthogonal and no dependency relation distinguishes them.**

WHY THERE IS NO CHEAP CORRECT VERSION. Agency is semantic. The project's only
agency instrument is F21's tagger — an aligned LLM scoring a property that
alignment is hypothesised to install — and [1737].1 forbids an annotator from any
family under test. So the options are an annotator outside the roster with the
kindred-system scope line declared, or the limit stated unmeasured. [1900] took the
second, and the clause carries its hand-classification provenance on its face.

AND THE SAME CONSTRAINT RETURNS LATER, WHICH IS WHY THIS FILE SAYS SO ([1900].4):
the run's agency readout — base-model agency as the grammatical floor, with only
movement above it attributable to alignment — needs the same annotator question
answered when that arm is specified. It does not gate the freeze. It should not be
rediscovered.
"""

import collections
import os
import re
import sys

#: THREE levels: this file sits at meta/M03_proceduralization/, so two dirnames
#: land on meta/ and not on the repo root. The first version had two and failed
#: on a path that read plausibly -- meta/pair_drafts/ -- which is why it was not
#: obvious from the traceback.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPEAKER = {"i", "we", "me", "us", "my", "our"}
AGENT_DEPS = {"nsubj", "poss"}
PATIENT_DEPS = {"dobj", "pobj", "nsubjpass", "dative", "iobj"}


def role(nlp, text):
    """The speaker's grammatical relation. NOT its semantic role — see docstring.

    The trailing stance clause ("... . I" / "... and I") is stripped: it is
    identical across both arms by construction and would score every cell
    agentive on a token the design holds constant.
    """
    t = re.sub(r"[,.]?\s*(and\s+)?(I|we)\s*$", "", text.strip(), flags=re.I)
    doc = nlp(t)
    ag = pat = 0
    for tok in doc:
        if tok.text.lower() not in SPEAKER:
            continue
        if tok.dep_ in AGENT_DEPS:
            ag += 1
        elif tok.dep_ in PATIENT_DEPS:
            pat += 1
    if ag and not pat:
        return "agent"
    if pat and not ag:
        return "patient"
    return "both" if ag else "absent"


def main():
    import spacy
    nlp = spacy.load("en_core_web_sm")
    rows = []
    for name in ("m03_scenarios_A.yaml", "m03_scenarios_B.yaml"):
        txt = open(os.path.join(ROOT, "pair_drafts", name)).read()
        for m in re.finditer(r'^\s+(indiv_I_absent|inst_I_absent):\s*"(.*)"', txt, re.M):
            rows.append((m.group(1).split("_")[0], m.group(2)))

    tab = collections.Counter()
    for arm, text in rows:
        tab[(arm, role(nlp, text))] += 1

    kinds = ("agent", "patient", "both", "absent")
    print("AGENCY PARSE CHECK — 36 marker-free cells. THE NUMBER BELOW IS NOT AGENCY.\n")
    print(f"  {'arm':<8}" + "".join(f"{k:>10}" for k in kinds) + f"{'n':>6}")
    for arm in ("indiv", "inst"):
        n = sum(tab[(arm, k)] for k in kinds)
        print(f"  {arm:<8}" + "".join(f"{tab[(arm, k)]:>10}" for k in kinds) + f"{n:>6}")

    sep = tab[("inst", "agent")] - tab[("indiv", "agent")]
    print(f"\n  separation {sep:+d} scenarios, against a hand reading of 15 of 18.")
    print("\n  DO NOT REPORT THIS AS A DISAGREEMENT. The counterexample:")
    print("    \"The officer slammed me ... even though I never resisted.\"")
    print("      'me' dobj on 'slammed'   'I' nsubj on 'resisted'  <- NEGATED")
    print("    Scored AGENTIVE while the speaker is the one things were done to.")
    print("\n  GRAMMATICAL SUBJECTHOOD IS NOT AGENCY. This instrument measures the")
    print("  first and the question is about the second. See the module docstring.")


if __name__ == "__main__":
    sys.exit(main())
