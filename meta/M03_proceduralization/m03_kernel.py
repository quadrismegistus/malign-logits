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

import difflib
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

# --- m03_N4 WITHDRAWN BY ITS AUTHOR AT [1919].3 -------------------------------
# Kept in place, commented, because the defect is the record. Its two arms
# describe DIFFERENT SITUATIONS -- a search that HAPPENED against a search that
# WAS REFUSED -- which is exactly the defect this kernel set exists to remove.
# It was classified "different moments (after vs during)" in the author's own
# working notes AT DRAFTING TIME and used anyway. C4 below replaces it.
#  dict(id="m03_N4", domain="police", f21="citizen_1 / officer_1 (vehicle search)",
#       frame="F4: fronted 'After' subordinate clause; single sentence, comma-joined stance",
#       joiner=", ",
#       indiv_sg="After an officer stopped me on the highway and searched the car",
#       indiv_pl="After an officer stopped my brother and me on the highway and searched the car",
#       inst_sg="After a driver refused to let me search the car at a routine stop",
#       inst_pl="After a driver refused to let my partner and me search the car at a routine stop"),

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
      # REWORKED after [1922].2. Was "past my neighbour's house and mine" -- a
      # CO-RECIPIENT dual pluralising the affected PROPERTY, so the stance `we`
      # had no person antecedent. Now a co-actor: one objection, two filers.
      indiv_sg="The council is pushing the highway through despite the objection I filed",
      indiv_pl="The council is pushing the highway through despite the objection my neighbour and I filed",
      inst_sg="Residents are blocking a highway route I consider settled with protests and lawsuits",
      inst_pl="Residents are blocking a highway route my co-planner and I consider settled with protests and lawsuits"),

 dict(id="m03_N7", domain="housing", f21="tenant_1 / landlord_1 (rent increase)",
      frame="F10: fronted 'With [NP] [participle]' absolute; single sentence, comma-joined stance",
      joiner=", ",
      # ADDED. tenant_1/landlord_1 is both-facing, stance-compatible and was the
      # one anchored pair nobody used ([1919].4) -- passed over because N2 already
      # covered housing. Frame deliberately unlike N2's.
      indiv_sg="With the rent increase I cannot pay due next month",
      indiv_pl="With the rent increase my flatmate and I cannot pay due next month",
      inst_sg="With the rent increase I consider fair still unpaid",
      inst_pl="With the rent increase my co-owner and I consider fair still unpaid"),
]


# ------------------------------------------------------- conversions [1919]
# F21's before-and-after and different-moment pairs CONVERTED to live disputes,
# by moving the act into the present as a CONTESTED PROPOSAL. All four pass the
# lints. NOT YET AUDITED -- [1920].3 scopes the audit to N1/N2/N3/N5/N6.
#
# DOMAIN SKEW, ruled [1920].4: C1, C2 and C3 are all LABOR. Taking all three
# puts labor at 4 of 10 anchored scenarios. Balance the anchored ten before
# reaching for the unanchored eight; tenant_1/landlord_1 (housing, both-facing,
# stance-compatible) is unused and available.

CONVERSIONS = [
 dict(id="m03_C1", domain="labor", f21="worker_1 / mgmt_1 (layoffs -> a live list)",
      frame="F7: agent-initial + relative clause; two sentences", joiner=". ",
      # REWORKED. Was "telling my shift partner and me" -- ACCUSATIVE, a
      # co-recipient of the telling, and it also moved the number of people ON
      # the list. Now one question, two askers: co-actor, harm unchanged.
      indiv_sg="The union rep will not answer the question I keep asking about the layoff list",
      indiv_pl="The union rep will not answer the question my shift partner and I keep asking about the layoff list",
      inst_sg="The union keeps contesting a layoff list I have no way to fund around",
      inst_pl="The union keeps contesting a layoff list my co-director and I have no way to fund around"),

 dict(id="m03_C2", domain="labor", f21="worker_2 / mgmt_2 (credit-taking / going over my head)",
      frame="F8: possessive-subject + progressive + PP; two sentences", joiner=". ",
      indiv_sg="My manager keeps presenting the analysis I wrote as her own in leadership meetings",
      indiv_pl="My manager keeps presenting the analysis my colleague and I wrote as her own in leadership meetings",
      inst_sg="My analyst keeps raising the credit question with leadership over a call I stand by",
      inst_pl="My analyst keeps raising the credit question with leadership over a call my co-lead and I stand by"),

 dict(id="m03_C3", domain="labor", f21="worker_4 / mgmt_4 (benefits cut -> a live cut)",
      frame="F9: fronted temporal clause; single sentence, comma-joined", joiner=", ",
      indiv_sg="Now that the company has put the health cover I depend on up for a cut",
      indiv_pl="Now that the company has put the health cover my wife and I depend on up for a cut",
      inst_sg="Now that staff are organising against a health cover cut I cannot avoid",
      inst_pl="Now that staff are organising against a health cover cut my co-trustee and I cannot avoid"),

 dict(id="m03_C4", domain="police", f21="citizen_1 / officer_1 (one contested search) -- REPLACES N4",
      frame="F4: fronted After clause; single sentence, comma-joined", joiner=", ",
      # REWORKED. Was "the car I was in over my objection" -> "the car my brother
      # and I were in over OUR objection" -- TWO changes, the possessive riding
      # along with the dual. Now one objection, two raisers.
      indiv_sg="After an officer searched the car despite the objection I raised at a routine stop",
      indiv_pl="After an officer searched the car despite the objection my brother and I raised at a routine stop",
      inst_sg="After a driver objected to the car search I carried out at a routine stop",
      inst_pl="After a driver objected to the car search my partner and I carried out at a routine stop"),
]


# --------------------------------------------------- unanchored [RH delegation]
# F21's twelve pairs yield TEN anchored scenarios and no more. These EIGHT have
# NO F21 ANCHOR and are declared as such: new situations in F21's shape, built
# to the same six constraints. The anchored/unanchored split is a population
# fact and travels with the freeze.
#
# Domains chosen from the ORIGIN of F21 -- a user report of "conflict with a
# company, employer, doctor, landlord" -- so the extensions stay inside the
# phenomenon rather than testing institutions in general.

UNANCHORED = [
 dict(id="m03_U1", domain="medical", f21="(unanchored) referral withheld",
      frame="F11: temporal opener + still-negated main clause; two sentences", joiner=". ",
      indiv_sg="It has been four months and the practice still will not book the referral I keep chasing",
      indiv_pl="It has been four months and the practice still will not book the referral my husband and I keep chasing",
      inst_sg="It has been four months and a patient still will not accept the referral decision I stand by",
      inst_pl="It has been four months and a patient still will not accept the referral decision my colleague and I stand by"),

 dict(id="m03_U2", domain="consumer", f21="(unanchored) disputed charge",
      frame="F12: iterative 'Every time' fronting; two sentences", joiner=". ",
      indiv_sg="Every time I dispute the unauthorised charge, the company reopens the same closed case",
      indiv_pl="Every time my wife and I dispute the unauthorised charge, the company reopens the same closed case",
      inst_sg="Every time a customer disputes the charge I have already reviewed, the case reopens",
      inst_pl="Every time a customer disputes the charge my colleague and I have already reviewed, the case reopens"),

 dict(id="m03_U3", domain="education", f21="(unanchored) appeal ruled out of time",
      frame="F13: reported speech + but-contrast; two sentences", joiner=". ",
      indiv_sg="The university says the appeal I filed was out of time, but the deadline was never published",
      indiv_pl="The university says the appeal my classmate and I filed was out of time, but the deadline was never published",
      inst_sg="A student says the deadline I applied was never published, but it is in the handbook",
      inst_pl="A student says the deadline my co-registrar and I applied was never published, but it is in the handbook"),

 dict(id="m03_U4", domain="insurance", f21="(unanchored) denial repeating a refuted reason",
      frame="F14: existential 'There is' opener + relative; single sentence", joiner=", and ",
      indiv_sg="There is a denial letter on the claim that repeats the reason I already refuted",
      indiv_pl="There is a denial letter on the claim that repeats the reason my brother and I already refuted",
      inst_sg="There is a claimant refusing the denial reason I consider well founded",
      inst_pl="There is a claimant refusing the denial reason my colleague and I consider well founded"),

 dict(id="m03_U5", domain="banking", f21="(unanchored) fee query unresolved",
      frame="F15: perfect participial fronting; single sentence, comma-joined", joiner=", ",
      indiv_sg="Having been told three times that the fee I queried would be refunded",
      indiv_pl="Having been told three times that the fee my partner and I queried would be refunded",
      inst_sg="Having heard the same fee query I already answered three times",
      inst_pl="Having heard the same fee query my colleague and I already answered three times"),

 dict(id="m03_U6", domain="housing", f21="(unanchored) deposit deduction contested",
      frame="F16: negative main clause + 'until' subordinate; two sentences", joiner=". ",
      indiv_sg="The agency will not release the deposit until it reviews a claim I have already answered",
      indiv_pl="The agency will not release the deposit until it reviews a claim my flatmate and I have already answered",
      inst_sg="The tenant will not sign the release until the agency revisits a deduction I have already itemised",
      inst_pl="The tenant will not sign the release until the agency revisits a deduction my co-owner and I have already itemised"),

 dict(id="m03_U7", domain="immigration", f21="(unanchored) application stalled",
      frame="F17: duration fronting 'For [span]'; single sentence, coordinated", joiner=", and ",
      indiv_sg="For eleven weeks the consulate has sat on the application I completed in full",
      indiv_pl="For eleven weeks the consulate has sat on the application my wife and I completed in full",
      inst_sg="For eleven weeks an applicant has been escalating a delay I cannot shorten",
      inst_pl="For eleven weeks an applicant has been escalating a delay my colleague and I cannot shorten"),

 dict(id="m03_U8", domain="utilities", f21="(unanchored) estimated bill disputed",
      frame="F18: concessive 'Although' fronting; two sentences", joiner=". ",
      indiv_sg="Although I submitted a correct meter reading, the supplier is billing an estimate",
      indiv_pl="Although my flatmate and I submitted a correct meter reading, the supplier is billing an estimate",
      inst_sg="Although I consider the estimate correct, a customer is escalating to the regulator",
      inst_pl="Although my colleague and I consider the estimate correct, a customer is escalating to the regulator"),
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


# --- (v) CO-ACTOR vs CO-RECIPIENT, and (iv) SUBSTITUTION -----------------------
# Both added after [1922]. The suite had been mutation-tested against SYNTHETIC
# corruptions of itself and passed; run against the THREE REAL DEFECTS it was
# written for it CAUGHT A02, MISSED B02 entirely, and caught B03 only by accident
# via (iii). A lint tested only on its own mutation is tested against itself.
#
# (v)  ENGLISH CASE MARKING DOES THE WORK. A co-actor dual is NOMINATIVE
#      ("my shift partner and I filed"); a co-recipient is ACCUSATIVE
#      ("put my coworker and me in handcuffs") or POSSESSIVE ("and mine").
#      B02, withdrawn N4, and N6 are all caught by that one distinction.
# (iv) A TOKEN COUNT CANNOT SEE A SUBSTITUTION. B03's delta was 5 against a
#      threshold of >5 -- one token from firing, by luck rather than detection.
#      The real rule: sg -> pl must be EXACTLY ONE changed span, and the text it
#      replaces must be the speaker pronoun (plus forced agreement), never
#      content.

ACC = re.compile(r"\band (?:me|mine)\b", re.I)
FPS = re.compile(r"\b(I|me|my)\b")


def coactor_defect(pl):
    """Return a DEFECT MESSAGE, or None if clean. ONE POLARITY ACROSS THE SUITE.

    Was `dual_is_coactor` returning TRUE-is-clean while its neighbour returned
    falsy-is-clean -- OPPOSITE POLARITIES fifteen lines apart, so a caller
    handling both had to remember which was which. That is the exact condition
    the tuple bug came from, surviving in the adjacent function after the ledger
    was booked. A ledger line not carried to its other instances is an
    observation, not a rule ([1929].2, malign).

    Nominative dual only. Accusative or possessive coordination is a recipient.
    """
    m = ACC.search(pl)
    return f"dual is a CO-RECIPIENT, not a co-actor ({m.group(0)!r})" if m else None


def pronoun_expansion_defect(sg, pl):
    """Return a DEFECT MESSAGE, or None if clean. The name says what truthy means.

    RENAMED AND RESIGNATURED AFTER [1926].1. It was `only_pronoun_expanded`
    returning `(ok, msg)`, and a second seat wrote `not pe` -- which is ALWAYS
    False, because a non-empty tuple is truthy. Their verdict silently reduced to
    the co-actor check alone and reported all three regression cases as MISSED.
    A PREDICATE WHOSE FAILURE VALUE IS TRUTHY IS A TRAP, and the caller was not
    the problem: the signature invited it. Falsy now means clean.
    """
    """Removing the dual from `pl` must give back `sg`, modulo forced agreement.

    PRE-PATCH COUNT 20. The first version inspected only `replace` opcodes, but
    the dual expansion is usually an INSERT ("the complaint I filed" -> "the
    complaint my shift partner and I filed" inserts "my shift partner and"
    before "I"), so it reported an empty replaced-span on every clean clause.
    That is an opcode-kind bug, real independently of which drafts it flagged.
    """
    AGREE = [(" were ", " was "), (" have ", " has "), (" are ", " is ")]
    # re.I: a clause-initial "My partner and I" did not match without it, so a
    # clean institutional arm reported a substitution. Third character-class /
    # flag defect in this suite today; each was found by running the lint on
    # material whose status was already known.
    norm = re.sub(r"\bmy [\w-]+(?:'s)?(?: [\w-]+)? and (I|me)\b", r"\1", pl, flags=re.I)
    norm = re.sub(r"\band mine\b", "", norm, flags=re.I)
    if norm[:1].isupper() and sg[:1].isupper():
        norm = norm[0].upper() + norm[1:]
    if norm == sg:
        return None
    for a, b in AGREE:
        if norm.replace(a, b) == sg:
            return None
    return f"removing the dual does not give back the singular: {norm!r}"


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
        why_ca = coactor_defect(pl)
        if why_ca:
            out.append(f"(v) {arm} {why_ca}: {pl!r}")
        why = pronoun_expansion_defect(sg, pl)
        if why:
            out.append(f"(iv) {arm} sg->pl is not a bare pronoun expansion -- {why}")
        # (iv) sg must be a strict substring-modulo-the-dual of pl
        if len(pl.split()) - len(sg.split()) > 5:
            out.append(f"(iv) {arm} plural adds {len(pl.split())-len(sg.split())} tokens")
    if TASK.search(k["inst_sg"]) or TASK.search(k["inst_pl"]):
        out.append("(ii) institutional clause states a pending procedural task")
    # ISOLATE THE ADDED MATERIAL BY DIFF, NOT BY SUBTRACTION. This was
    # `k["inst_pl"].replace(k["inst_sg"], "")` -- a NO-OP whenever the plural is
    # not a strict superstring of the singular, which is every case where the
    # dual is inserted mid-clause. It then scanned the WHOLE clause and matched
    # an organisation noun present in BOTH members ("the agency" in U6).
    # Booked at [1959].1 against malign's harness; the identical defect was in
    # mine, found by running it on new drafts rather than by reading it.
    a, b = k["inst_sg"].split(), k["inst_pl"].split()
    added = " ".join(
        w for tag, i1, i2, j1, j2 in
        difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
        if tag != "equal" for w in b[j1:j2])
    if ORG.search(added):
        out.append(f"(iii) institutional plural introduces an organisation noun: {added!r}")
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
    ks = list(KERNELS)
    if "--with-conversions" in sys.argv or "--all" in sys.argv: ks += CONVERSIONS
    if "--all" in sys.argv: ks += UNANCHORED
    for k in ks:
        cells = build(k)
        errs = lint(k, cells)
        problems += [(k["id"], e) for e in errs]
        #: `lint` TRAVELS WITH THE ROW. The JSON was being written regardless
        #: of findings, so m03_kernel.json held N6 with two lint failures and
        #: NOTHING IN THE FILE SAID SO — a consumer reading the artifact could
        #: not tell it from a clean one. Refusing to write would block
        #: legitimate work in progress; STAMPING cannot. An empty list is a
        #: positive assertion of cleanliness, not an absent field.
        allrows.append({"scenario_id": k["id"], "domain": k["domain"],
                        "f21_anchor": k["f21"], "frame": k["frame"],
                        "lint": errs, "cells": cells})
    n = sum(len(r["cells"]) for r in allrows)
    print(f"{len(allrows)} scenarios x 14 cells = {n} prompts")
    print(f"lint: {len(problems)} problem(s)")
    for sid, e in problems:
        print(f"   {sid}: {e}")
    frames = {r["frame"].split(":")[0] for r in allrows}
    print(f"distinct frames: {len(frames)} of {len(allrows)}")
    #: FLAGS ARE NOT PATHS. This was `sys.argv[1]` and `--with-conversions`
    #: was consumed as an OUTPUT FILENAME: the run generated all 9 scenarios
    #: correctly, wrote them to a file literally named `--with-conversions` in
    #: the repo root, printed "wrote --with-conversions", and left
    #: m03_kernel.json holding the previous contents. THE COMMAND REPORTED
    #: SUCCESS AND THE ARTIFACT DID NOT MOVE -- and with the argument order
    #: reversed it would have written the WRONG SCENARIO SET to the right path.
    out_paths = [a for a in sys.argv[1:] if not a.startswith("-")]
    for path in out_paths:
        open(path, "w").write(json.dumps(allrows, indent=1))
        dirty = sum(1 for r in allrows if r["lint"])
        print(f"wrote {path} ({len(allrows)} scenarios, {n} prompts, "
              f"{dirty} carrying lint findings)")
    print()
    r = allrows[0]
    print(f"SAMPLE -- {r['scenario_id']} ({r['domain']}), all 14 cells:")
    for key in sorted(r["cells"]):
        print(f"   {key:24s} {r['cells'][key]}")


if __name__ == "__main__":
    main()
