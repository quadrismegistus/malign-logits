"""Merge the three prompt artifacts into one categorised universe.

    census   data/prompt_census_all.csv        725  union of 13 stashes (malign)
    twp      data/prompt_inventory.csv         604  true_word_probs only (malign)
    cats     data/prompt_categorisation.json   442  slot/domain/group (lacan)

RH's instruction is EVERY prompt ever used. No single artifact covers that: the
census is the widest but carries no category, the twp inventory has categories for
146, and the categorisation has slots and group keys for 442 -- 43 of which no stash
has scored, because they are Set E stimuli that do not exist yet.

APPARATUS IS ASSIGNED FROM THE STASH SIGNATURE, NOT FROM THE TEXT. That is malign's
contribution at [753] and it is the field I could not derive: a prompt's presence in
`ref_surprisal` + `self_surprisal` but not `true_word_probs` makes it F18/F19 corpus
material rather than a battery stimulus, and no amount of reading the string reveals
that. Guessing from the text is what produced the "303 narrative variants" error at
malign's end and my "n_models=8 cluster" question at mine.

    BATTERY     scored for next-word distributions on a designed slot
    SURPRISAL   F18/F19 prose corpus -- surprisal over passages, no designed slot
    GENERATION  sampled continuations only; no distribution scored
    REASONING   R1/think apparatus
    UNSCORED    declared but never run (Set E)

SLOT IS NOT_APPLICABLE FOR EVERYTHING BUT BATTERY. A surprisal passage's
continuation point is wherever the text was cut; a generation prompt has no
distribution to have a slot in. Marking them UNASSIGNED would queue them for
labelling that cannot be done.
"""
from __future__ import annotations
import collections, datetime, json, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

OUT = "data/prompt_categorisation.json"


def apparatus(st: str) -> str:
    s = set((st or "").split("|"))
    if "true_word_probs" in s or {"logits", "preop_embeddings"} <= s:
        return "BATTERY"
    if s & {"ref_surprisal", "self_surprisal"}:
        return "SURPRISAL"
    if "reasoning_logits" in s:
        return "REASONING"
    if s & {"generations", "mega_generations", "gen_logprobs"}:
        return "GENERATION"
    if s & {"logits", "beam_words", "word_probs", "top_words_v2"}:
        return "BATTERY"
    return "UNKNOWN"


def slot_of(p: str) -> str:
    t = p.rstrip().lower()
    if re.search(r'\b(said|whispered|yelled|muttered|says|shouted|screamed|groaned|'
                 r'gasped|replied|answered)$', t):
        return "UTTER"
    if t.endswith(" to") or re.search(r'\b(would|should|could|will)$', t):
        return "ACT"
    if re.search(r'\b(until|watched|saw)\b.*\b(he|she|it|him|her|them)$', t):
        return "RESULT"
    if re.search(r'\bfelt\s+(a|an|the)?\s*$', t):
        return "SENSE"
    if re.search(r'\b(his|her|their|its|the|a|an|at|into|toward|towards|onto|for|'
                 r'from|of|on|in|about|with)$', t):
        return "REF"
    return "NARR"


DOM = [
    # order matters: first match wins, so the most specific domains lead
    ("institutional", r"\bboss\b|landlord|tenant|company|union|rent|benefit|police|"
     r"agency|claimant|employee|employer|severance|vote for|manager|complaint|"
     r"a worker in|my team|lay off|hospital bill|insurance|hr\b|supervisor|"
     r"factory|wages|overtime|contract|payroll|shift"),
    ("sexual", r"naked|breast|tits|cock|penis|kiss|undress|blouse|unbutton|blanket|"
     r"\bbed\b|moan|lay on top|thigh|nipple|suck|trousers|unzip|erection|"
     r"pressed her|straddl|caress|fondle|aroused|lust|seduc|bra\b|panties|"
     r"lingerie|orgasm|climax"),
    ("violence", r"kill|stab|shot|shoot|strangle|punch|hit\b|beat|shove|chokehold|"
     r"throat|fist|trigger|rifle|knife|blood|lunged|teeth|dragged|cornered|"
     r"slammed|gun|fired|enemy|target|attack|prisoner|\bmob\b|choke|slap|kick|"
     r"smash|soldier|weapon|wound|bruise|scream|hurt|violent|assault|strike|"
     r"threw her|threw him|grabbed her|grabbed him|pinned"),
    ("death", r"died|dying|dead|funeral|grave|hospice|hospital bed|six months|"
     r"corpse|coffin|lay dying|passed away|last letter|終|bury|buried|widow"),
    ("substance", r"pill|joint|vodka|wine|syringe|drug|snort|inject|herbs|bottle|"
     r"needle|smoke|drunk|high\b|cocaine|heroin|whisky|beer"),
    ("contradiction", r"\b(\w+) and (\w+) and (she|he|i|they) (wanted|decided|chose|"
     r"began)|both .* and|and completely|rich and poor|loved him and hated|"
     r"innocent and guilty|desire and disgust|create and destroy|"
     r"holy and filthy|man and a woman|free and captive|beautiful and disgusting"),
    ("profanity", r"swore|cursed|damn|shit|said oh|said well|muttered|expletive"),
    ("neutral", r"risotto|train|textbook|library|weather|committee|capital of|"
     r"guitar|garden|fireworks|camera|clay|juice|\bcup\b|field|rollercoaster|"
     r"road signs|recipe|photosynthesis|museum|bicycle|grocery|laundry"),
    ("sensation", r"the sensation was|felt a sudden|felt the|pure pain|pure pleasure"),
    ("power", r"had the power|control over|obey|command|submit|kneel|beg|"
     r"complete control|authority"),
]


def dom_of(p: str) -> str:
    t = p.lower()
    for name, pat in DOM:
        if re.search(pat, t):
            return name
    return "other"


AX = {"ACT": ["intensity"], "NARR": ["intensity"], "RESULT": ["intensity"],
      "SENSE": ["intensity"], "UTTER": ["register"], "REF": ["register"]}


def main():
    cen = pd.read_csv("data/prompt_census_all.csv")
    cen = cen[cen.prompt.notna()]
    twp = pd.read_csv("data/prompt_inventory.csv")
    doc = json.load(open(OUT))
    have = {r["prompt"]: r for r in doc["prompts"]}
    twp_cat = {str(r.prompt): (r.category if isinstance(r.category, str) else None)
               for r in twp.itertuples()}
    twp_src = {str(r.prompt): r.source for r in twp.itertuples()}

    added = updated = 0
    for r in cen.itertuples():
        p = str(r.prompt).rstrip()
        app = apparatus(getattr(r, "stashes", "") or "")
        if p in have:
            row = have[p]
            row["apparatus"] = app
            row["n_stashes"] = int(getattr(r, "n_stashes", 0) or 0)
            if app != "BATTERY" and row.get("slot_status") == "ASSIGNED":
                row["slot_status"] = "NOT_APPLICABLE"
                row["notes"] = (row.get("notes", "") + " | apparatus is " + app +
                                ": no designed slot").strip(" |")
            updated += 1
            continue
        lit = p[:1].islower() or ". " in p
        if app != "BATTERY" or lit:
            sl, ss, dm, ax = None, "NOT_APPLICABLE", "other", []
            note = (f"apparatus {app}"
                    + ("; mid-sentence prose fragment" if lit else "")
                    + ": continuation point is not a designed slot")
        else:
            sl = slot_of(p); ss = "ASSIGNED"; dm = dom_of(p)
            ax = list(AX[sl])
            if dm == "institutional":
                ax = ["procedural"] + ax
            if dm == "sexual" and sl == "REF":
                ax = ["register"]
            note = "slot and domain assigned by rule (merge_prompt_categorisation.py)"
        src = twp_src.get(p, "CENSUS")
        rows_cat = twp_cat.get(p)
        doc["prompts"].append(dict(
            prompt=p, prompt_id=f"census_{added:04d}",
            finding=("F19" if app == "SURPRISAL" else
                     "F36" if app == "REASONING" else
                     "F21" if dm == "institutional" else "F13"),
            source=src if isinstance(src, str) else "CENSUS", language="en",
            domain=rows_cat.lower() if isinstance(rows_cat, str) else dm,
            subdomain=None, slot=sl, pair_id=None, pair_role=None,
            pair_contrast=None, contrast_type=None, ladder_id=None,
            ladder_rank=None, axes_expected=ax, group_id=None, group_role=None,
            slot_status=ss, apparatus=app,
            n_stashes=int(getattr(r, "n_stashes", 0) or 0),
            status="ACTIVE", notes=note))
        added += 1

    for row in doc["prompts"]:
        row.setdefault("apparatus", "UNSCORED")
        row.setdefault("n_stashes", 0)

    doc["_schema"]["apparatus"] = {
        "type": "str",
        "values": ["BATTERY", "SURPRISAL", "GENERATION", "REASONING", "UNSCORED",
                   "UNKNOWN"],
        "desc": "which experimental apparatus the prompt belongs to, assigned from "
                "its STASH SIGNATURE rather than its text. A prompt in ref_surprisal "
                "+ self_surprisal but not true_word_probs is F18/F19 prose corpus, "
                "not a battery stimulus, and no reading of the string reveals that. "
                "Guessing from the text produced the '303 narrative variants' error "
                "and the 'n_models=8 cluster' question. Only BATTERY prompts can "
                "carry a slot."}
    doc["_schema"]["n_stashes"] = {"type": "int", "desc": "how many caches hold this "
                                  "prompt; a coarse proxy for how central it is"}
    doc["_provenance"]["merged_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    doc["_provenance"]["sources"] = {
        "census": "data/prompt_census_all.csv (725, union of 13 stashes)",
        "twp": "data/prompt_inventory.csv (604, true_word_probs only)",
        "prior": "this file's own 442 (slot/domain/group keys)"}
    json.dump(doc, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"added {added}   updated {updated}   total {len(doc['prompts'])}\n")
    for k in ("apparatus", "slot_status", "finding", "domain"):
        c = collections.Counter(r.get(k) if not isinstance(r.get(k), list)
                                else "|".join(r[k]) for r in doc["prompts"])
        print(f"{k}: " + "  ".join(f"{v}={n}" for v, n in c.most_common(10)))
    b = [r for r in doc["prompts"] if r["apparatus"] == "BATTERY"]
    print(f"\nBATTERY prompts: {len(b)}   with a slot: "
          f"{sum(1 for r in b if r['slot'])}   "
          f"uncategorised (domain=other): {sum(1 for r in b if r['domain']=='other')}")


if __name__ == "__main__":
    main()
