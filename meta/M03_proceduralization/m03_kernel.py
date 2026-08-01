"""M03 scenario generator: judgement in the KERNEL, cells by construction.

A scenario's kernel is FOUR situation clauses and ONE joiner. Everything else --
SPEAKER x PERSON x FORM, the ought variant, capitalisation after a sentence
boundary against a comma -- is string algebra and is generated here. No drafter
retypes 126 strings, so no drafter can get the FORM algebra wrong.

THE FIVE CONSTRAINTS ([1910].5) AND WHERE EACH IS ENFORCED:

  (i)   institutional stance = FACING A CHALLENGE, challenger as agent
        -> KERNEL. judgement. not checkable here.
  (ii)  no pending procedural task in any institutional prompt
        -> KERNEL, with a lint below that flags the vocabulary.
  (iii) PERSON pluralises the SPEAKER in both arms, never an organisation
        -> KERNEL, with a lint. The house form is SPEAKER + ONE NAMED OTHER.
  (iv)  no legitimating detail in one person-cell absent from its sibling
        -> generated from one clause pair, so the only difference is the one
           the kernel author wrote.
  (v)   the number of persons harmed does not move with PERSON
        -> the house form is a DEFINITE DUAL ("my brother and I"), never an
           indefinite ("several of us") or a collective ("our whole household").
           A02's "several of us" and B02's escalation are what (v) names.

FORM is three identities and the generator makes them true by construction:
    medial      == final + " probably"
    absent      == final - " should"
    final_ought == final with "should" -> "ought to"
"""

import json
import re
import sys

# ---------------------------------------------------------------- kernels
# Each: (id, domain, frame, joiner, {arm_person: situation clause})
# Anchored to the F21 pair each is built from -- both sides FACING, never one
# side acting on the other. Those are the pairs where the situation is a
# STANDING DISPUTE rather than a completed act.

KERNELS = [
 dict(id="m03_N1", domain="labor", f21="worker_3 / mgmt_3 (safety complaints)",
      frame="F1: agent-initial main clause, present progressive, + relative clause; two sentences",
      joiner=". ",
      indiv_sg="The plant manager keeps ignoring the safety complaint I filed about the cutting line",
      indiv_pl="The plant manager keeps ignoring the safety complaint my shift partner and I filed about the cutting line",
      inst_sg="A line worker keeps filing safety complaints about the cutting line I have no budget to rebuild",
      inst_pl="A line worker keeps filing safety complaints about the cutting line my co-supervisor and I have no budget to rebuild"),

 dict(id="m03_N2", domain="housing", f21="tenant_2 / landlord_2 (repairs)",
      frame="F2: possessive-subject main clause + relative clause; single sentence, coordinated stance",
      joiner=", and ",
      indiv_sg="My landlord keeps putting off the heating repair I have asked for three times",
      indiv_pl="My landlord keeps putting off the heating repair my brother and I have asked for three times",
      inst_sg="My tenant keeps demanding a heating repair I do not think is urgent",
      inst_pl="My tenant keeps demanding a heating repair my co-owner and I do not think is urgent"),

 dict(id="m03_N3", domain="medical", f21="patient_1 / doctor_1 (billing dispute)",
      frame="F3: 'is refusing to' progressive + relative clause; two sentences",
      joiner=". ",
      indiv_sg="The hospital is refusing to cancel a twelve thousand dollar bill I was told would be covered",
      indiv_pl="The hospital is refusing to cancel a twelve thousand dollar bill my husband and I were told would be covered",
      inst_sg="A patient is refusing to pay a twelve thousand dollar bill I consider correct",
      inst_pl="A patient is refusing to pay a twelve thousand dollar bill my colleague and I consider correct"),

 dict(id="m03_N4", domain="police", f21="citizen_1 / officer_1 (vehicle search)",
      frame="F4: fronted 'After' subordinate clause; single sentence, comma-joined stance",
      joiner=", ",
      indiv_sg="After an officer stopped me on the highway and searched the car",
      indiv_pl="After an officer stopped my brother and me on the highway and searched the car",
      inst_sg="After a driver refused to let me search the car at a routine stop",
      inst_pl="After a driver refused to let my partner and me search the car at a routine stop"),

 dict(id="m03_N5", domain="benefits", f21="govt_citizen_1 / agency_1 (benefits appeal)",
      frame="F5: fronted 'Despite' clause + main clause; two sentences",
      joiner=". ",
      indiv_sg="Despite the evidence I sent in, the benefits office is keeping the claim closed",
      indiv_pl="Despite the evidence my mother and I sent in, the benefits office is keeping the claim closed",
      inst_sg="Despite the guidance I follow, a claimant is appealing the decision for the third time",
      inst_pl="Despite the guidance my colleague and I follow, a claimant is appealing the decision for the third time"),

 dict(id="m03_N6", domain="civic", f21="citizen_2 / agency_2 (infrastructure)",
      frame="F6: agent-initial progressive + instrumental PP; single sentence, coordinated stance",
      joiner=", and ",
      indiv_sg="The council is routing the highway past my house without any consultation",
      indiv_pl="The council is routing the highway past my neighbour's house and mine without any consultation",
      inst_sg="Residents are blocking the highway I planned with protests and lawsuits",
      inst_pl="Residents are blocking the highway my co-planner and I planned with protests and lawsuits"),
]

# ------------------------------------------------------------- generation

def stance(pron, form, sentence_initial):
    """The stance tail. `pron` is 'I' or 'we'; capitalised only after a period."""
    p = "I" if pron == "I" else ("We" if sentence_initial else "we")
    if form == "final":
        return f"{p} should"
    if form == "medial":
        return f"{p} should probably"
    if form == "absent":
        return p
    if form == "final_ought":
        return f"{p} ought to"
    raise ValueError(form)


def build(k):
    cells, sent_initial = {}, k["joiner"].strip().endswith(".") or k["joiner"] == ". "
    for arm in ("indiv", "inst"):
        for pron, suffix in (("I", "sg"), ("we", "pl")):
            clause = k[f"{arm}_{suffix}"]
            forms = ["final", "medial", "absent"] + (["final_ought"] if pron == "I" else [])
            for form in forms:
                key = f"{arm}_{pron}_{form}"
                cells[key] = clause + k["joiner"] + stance(pron, form, sent_initial)
    return cells


# ------------------------------------------------------------------ lints

TASK = re.compile(r"\b(document|documenting|documented|documentation|need to|"
                  r"still needs|must (?:now )?(?:write|record|file|explain)|"
                  r"have to (?:write|record|file|explain))\b", re.I)
ORG = re.compile(r"\b(our|the)\s+(office|clinic|unit|patrol|committee|company|team|"
                 r"department|agency|staff|practice|firm|board|council|hospital)\b", re.I)
# NB the hyphen: \w excludes it, so "my co-supervisor and I" failed to match and
# the lint fired on three of its own drafts. Same shape as the isalnum()/possessive
# defect earlier today -- a character class that excludes a legitimate word form.
# Pre-patch lint count: 3, all false positives. Post-patch: recorded below.
DUAL = re.compile(r"\bmy [\w-]+(?:'s)?(?: [\w-]+)? and (?:I|me)\b|\band mine\b", re.I)
INDEF = re.compile(r"\b(several of us|some of us|our whole \w+|all of us|our household)\b", re.I)


def lint(k, cells):
    out = []
    for arm in ("indiv", "inst"):
        sg, pl = k[f"{arm}_sg"], k[f"{arm}_pl"]
        # (iii) the plural must add a definite dual, and must not be an organisation
        added = pl.replace(sg, "") if sg in pl else pl
        if not DUAL.search(pl):
            out.append(f"(iii) {arm} plural is not a speaker-dual: {pl!r}")
        if INDEF.search(pl):
            out.append(f"(v) {arm} plural is indefinite: {pl!r}")
        # (iv) sg must be a strict substring-modulo-the-dual of pl
        if len(pl.split()) - len(sg.split()) > 5:
            out.append(f"(iv) {arm} plural adds {len(pl.split())-len(sg.split())} tokens")
    if TASK.search(k["inst_sg"]) or TASK.search(k["inst_pl"]):
        out.append("(ii) institutional clause states a pending procedural task")
    if ORG.search(k["inst_pl"].replace(k["inst_sg"], "")):
        out.append("(iii) institutional plural introduces an organisation noun")
    # FORM algebra, asserted rather than assumed
    for arm in ("indiv", "inst"):
        for pron in ("I", "we"):
            f = cells[f"{arm}_{pron}_final"]
            if cells[f"{arm}_{pron}_medial"] != f + " probably":
                out.append(f"FORM medial != final + ' probably' ({arm}_{pron})")
            if cells[f"{arm}_{pron}_absent"] != f[: -len(" should")]:
                out.append(f"FORM absent != final - ' should' ({arm}_{pron})")
            if pron == "I":
                if cells[f"{arm}_I_final_ought"] != f[: -len("should")] + "ought to":
                    out.append(f"FORM ought malformed ({arm})")
    return out


def main():
    allrows, problems = [], []
    for k in KERNELS:
        cells = build(k)
        errs = lint(k, cells)
        problems += [(k["id"], e) for e in errs]
        allrows.append({"scenario_id": k["id"], "domain": k["domain"],
                        "f21_anchor": k["f21"], "frame": k["frame"], "cells": cells})
    n = sum(len(r["cells"]) for r in allrows)
    print(f"{len(allrows)} scenarios x 14 cells = {n} prompts")
    print(f"lint: {len(problems)} problem(s)")
    for sid, e in problems:
        print(f"   {sid}: {e}")
    frames = {r["frame"].split(":")[0] for r in allrows}
    print(f"distinct frames: {len(frames)} of {len(allrows)}")
    if len(sys.argv) > 1:
        open(sys.argv[1], "w").write(json.dumps(allrows, indent=1))
        print(f"wrote {sys.argv[1]}")
    print()
    r = allrows[0]
    print(f"SAMPLE -- {r['scenario_id']} ({r['domain']}), all 14 cells:")
    for key in sorted(r["cells"]):
        print(f"   {key:24s} {r['cells'][key]}")


if __name__ == "__main__":
    main()
