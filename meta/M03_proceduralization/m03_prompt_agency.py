"""M03 PROMPT-SIDE agency classification, made reproducible.

[1897].1 classified 18 scenarios by hand: the INDIVIDUAL speaker is the PATIENT
of the wrong, the INSTITUTIONAL speaker is its AGENT. [1898].6 commissioned a
check. That commission is the CONTINUATION-side measure (F21's DeepSeek tagger on
base generations) and it needs generation + API custody, which is not this seat.

THIS producer does the other half: it turns MY hand classification of the PROMPT
into a rule anyone can re-run. It does NOT measure agency of model output and is
not a substitute for the commissioned check.

THE RULE, on the marker-free cells only (`indiv_I_absent`, `inst_I_absent`), on
the text BEFORE the trailing stance pronoun:

    PATIENT   the speaker is object/oblique of the event verb  ("slammed ME",
              "dropped ME", "took credit for MY analysis"), or the event is a
              PASSIVE with the speaker as subject ("after BEING PASSED OVER"),
              or the event's subject is a THIRD PARTY ("the county dropped",
              "my landlord kept")
    AGENT     the speaker is the subject of an ACTIVE event verb ("I used a
              takedown", "I dropped a voter", "After PROMOTING the newer hire")
    NEITHER   agentless nominal or passive with no speaker role

Every classification prints its trigger so a reader can disagree with a specific
string rather than with a count.
"""

import re
import glob

SCEN = "/Users/rj416/github/malign-logits/pair_drafts/m03_scenarios_*.yaml"

# speaker as object of a transitive event verb
OBJ = re.compile(r"\b(\w+ed|took|put|slammed|moved|kept|dropped|denied|found|flagged|"
                 r"rejected|searched|seized|says|say)\b[^.,]{0,30}\b(me|my|mine)\b", re.I)
# passive with the speaker implicated as the undergoer
PASS = re.compile(r"\b(being|been|was|were|is|are)\s+\w+(ed|en)\b", re.I)
# speaker as subject of an active verb, or gerund-fronted act by the speaker
AGT = re.compile(r"\bI\s+(\w+ed|used|put|took|kept|moved|dropped|searched|"
                 r"seized|flagged|rejected|denied|sent|had|keep|say)\b", re.I)
AGT_GER = re.compile(r"^(After|Before|Despite|By)\s+\w+ing\b", re.I)
THIRD = re.compile(r"^(The|My|Our|They|A)\s+[\w' ]{2,30}?\b(\w+ed|took|put|says|say|"
                   r"kept|moved|dropped|denied|flagged|rejected|searched|slammed)\b", re.I)


def strip_stance(s):
    """Remove the trailing stance pronoun the cell ends on."""
    return re.sub(r"[,.]?\s*(I|We|and I|and we|, and I|, and now I)\s*$", "", s.strip())


def classify(text, speaker_first_person_act):
    t = strip_stance(text)
    trig = []
    agent = False
    if AGT.search(t):
        agent = True
        trig.append("speaker subject of active verb")
    if AGT_GER.match(t) and re.search(r"^(After|Before|By)\s+(promoting|denying|"
                                      r"rescheduling|matching|sending)", t, re.I):
        agent = True
        trig.append("gerund-fronted act by speaker")
    patient = False
    if OBJ.search(t):
        patient = True
        trig.append("speaker is object of event verb")
    if PASS.search(t) and (re.search(r"\b(me|my)\b", t, re.I) or not AGT.search(t)):
        patient = True
        trig.append("passive, speaker undergoer")
    if THIRD.match(t) and re.search(r"\b(me|my)\b", t, re.I):
        patient = True
        trig.append("third-party subject acting on speaker")
    if agent and not patient:
        return "AGENT", trig
    if patient and not agent:
        return "PATIENT", trig
    if agent and patient:
        return "MIXED", trig
    return "NEITHER", trig


def main():
    rows = []
    for f in sorted(glob.glob(SCEN)):
        txt = open(f).read()
        for blk in txt.split("\n- scenario_id: ")[1:]:
            sid = blk.split("\n")[0].strip()
            def cell(k):
                m = re.search(rf'^\s+{k}: "(.*?)"\s*$', blk, re.M)
                return m.group(1) if m else None
            rows.append((sid, cell("indiv_I_absent"), cell("inst_I_absent")))

    inverted = 0
    print(f"{len(rows)} scenarios, marker-free cells\n")
    for sid, iv, it in rows:
        civ, tiv = classify(iv, False)
        cit, tit = classify(it, True)
        flag = ""
        if civ == "PATIENT" and cit == "AGENT":
            inverted += 1
            flag = "  <- INVERTED"
        print(f"{sid}  indiv={civ:7s}  inst={cit:7s}{flag}")
        print(f"    indiv trigger: {tiv or ['none']}")
        print(f"    inst  trigger: {tit or ['none']}")
    print(f"\nPATIENT(indiv) -> AGENT(inst): {inverted} of {len(rows)}")
    print("Hand classification at [1897] was 15 of 18. A disagreement here is a")
    print("disagreement about specific strings, which is the point of the rule.")


if __name__ == "__main__":
    main()
