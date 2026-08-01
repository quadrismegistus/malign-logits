"""Build `data/prompt_categorisation.json` -- the fields a prompt needs before any
stratified statistic can be computed on it.

WHY EACH FIELD EXISTS. Every one traces to a specific failure this campaign hit, and
the `why` is carried in the file's own `_schema` block so that a later reader cannot
drop a field without seeing what it was for.

    finding          F11's `captive / free` and F13's `fist / voice` are mechanically
                     identical -- one token apart, both scored, both filed UNMAPPED --
                     and they answer different questions. F11's pairs use `chose to`
                     and `decided to`, a frame that FORCES a choice between poles, and
                     test superposition against frame-exit. F13's use open
                     continuations after a scene and test displacement. Pooling them
                     measures neither. Nothing in the existing inventory separates
                     them.

    slot             The largest effect in the whole study (referent slots 0.55
                     paradigmatic against narration 0.17) and it was never a column,
                     only a hand map inside one draw script. 458 of 604 prompts had no
                     slot, so the eligibility machinery could not see them.

    domain/subdomain Violence attenuates, institutional escalates, sexual moves on
                     register. Pooled they cancel to zero -- which is exactly what the
                     first (B) test reported. `liminal` vs `explicit` matters too:
                     violence_liminal is the top-agreeing stratum of 22 and
                     violence_explicit is mid-pack.

    pair_id          A contrast a statistic cannot see is not a contrast. 230
                     one-token pairs are scored and none of them is grouped.

    pair_role        NOT `transgressive`, because the marked member is not always the
                     harmful one: for a negation flip it is the negated frame, for an
                     F11 pole pair neither pole is transgressive, and Set D's `blanket`
                     was filed as transgressive and produced `covered` and `wrapped`
                     while its twin produced `kissed`.

    contrast_type    What the one changed token actually manipulates. `helped/shoved`
                     swaps the act; `clay/man` swaps the TARGET while holding the act;
                     `fist/voice` swaps the instrument and with it the slot's
                     admissible fillers. Those are three different experiments and
                     they were all called "minimal pairs".

    ladder_id/rank   `wanted to [cry|hit|punch|kill|shoot|stab|strangle]` is a graded
                     series, not a pair, and its rank is the manipulation. A pair-only
                     schema loses the ordering, which is the whole design.

    axes_expected    The ranking instrument has three axes and each stratum moves on
                     one. Recording which axes SHOULD populate makes the coder's output
                     auditable against a pre-declared expectation instead of being
                     interpreted after the fact.

    status/notes     `blanket` is scored, drawn, and should not be used. Without a
                     status field the only way to know that is to have read the
                     conversation in which it was found.

DELIBERATELY ABSENT: n_models, n_cells, scored. Those are properties of a LIVE store
that grew from 2,649 to 5,947 entries inside one session, and a static file that
carries them becomes a lie the moment the roster advances. Read them from
`prompt_inventory.csv`, which stamps its read time and says to rebuild rather than
quote.

    uv run .venv/bin/python scripts/build_prompt_categorisation.py
"""
from __future__ import annotations
import collections, datetime, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import malign_logits.taxonomy as T
from scripts.f13_draw_relation_items import TYPE_OF

OUT = "data/prompt_categorisation.json"

SCHEMA = {
    "prompt": {"type": "str", "desc": "verbatim prompt text, rstripped; the join key "
               "against true_word_probs"},
    "prompt_id": {"type": "str", "desc": "stable slug; never derived from the text at "
                  "read time, because a truncated display string was once used as a "
                  "filter and matched nothing"},
    "finding": {"type": "str", "values": ["F01", "F11", "F13", "F21", "F36", "none"],
                "desc": "which finding's design this prompt belongs to. Determines "
                        "what question a contrast answers."},
    "source": {"type": "str", "values": ["DEFAULT", "INSTITUTIONAL", "CHINESE",
                                         "SETD", "SETE", "F36", "LITERARY", "OTHER"]},
    "language": {"type": "str", "values": ["en", "zh"]},
    "domain": {"type": "str", "values": ["violence", "sexual", "death", "power",
               "profanity", "substance", "institutional", "neutral", "coherence",
               "contradiction", "class", "other"]},
    "subdomain": {"type": "str|null", "desc": "liminal|explicit for violence and "
                  "sexual; worker|mgmt|tenant|landlord|citizen|agency|patient|doctor|"
                  "officer|party for institutional; null otherwise"},
    "slot": {"type": "str", "values": ["ACT", "NARR", "REF", "RESULT", "SENSE",
             "UTTER"], "desc": "grammar of the continuation slot. IMPORTED from "
             "f13_draw_relation_items.TYPE_OF for canonical prompts -- one source of "
             "truth, never re-derived"},
    "pair_id": {"type": "str|null", "desc": "groups the members of a minimal pair or "
                "series"},
    "pair_role": {"type": "str|null", "values": ["MARKED", "UNMARKED"],
                  "desc": "which member carries the manipulation. NOT a claim about "
                          "harm."},
    "pair_contrast": {"type": "str|null", "desc": "the tokens that differ, "
                      "marked/unmarked, e.g. 'shoved/helped'"},
    "contrast_type": {"type": "str|null",
                      "values": ["transgressive_swap", "target_swap",
                                 "instrument_swap", "pole_swap", "negation_flip",
                                 "register_swap", "grammar_swap", "role_swap",
                                 "gender_swap", "intensity_ladder", "grievance_swap",
                                 "channel_swap"],
                      "desc": "what the changed token manipulates. Three designs were "
                              "all being called 'minimal pairs'."},
    "ladder_id": {"type": "str|null", "desc": "graded series this belongs to"},
    "ladder_rank": {"type": "int|null", "desc": "position in the series, 1 = mildest"},
    "axes_expected": {"type": "list[str]", "values": ["intensity", "register",
                      "procedural"], "desc": "which ranking axes should populate. "
                      "Pre-declared so coder output is auditable."},
    "group_id": {"type": "str|null", "desc": "the DESIGN GROUP this prompt belongs "
                 "to. Supersedes pair_id for anything larger than two: F11's design is "
                 "A / B / A-AND-B ('rich', 'poor', 'rich and poor') and the both-poles "
                 "member is the critical cell, because that is where superposition is "
                 "tested against frame-exit. A pair-only schema cannot express it and "
                 "one-token pair detection MISSES it, since the both member differs "
                 "from each pole by more than one token."},
    "group_role": {"type": "str|null", "values": ["POLE_A", "POLE_B", "BOTH",
                   "MARKED", "UNMARKED", "LADDER"],
                   "desc": "position in the design group. POLE_A/POLE_B/BOTH for "
                           "contradiction triples; MARKED/UNMARKED for two-member "
                           "contrasts; LADDER for graded series."},
    "slot_status": {"type": "str", "values": ["ASSIGNED", "UNASSIGNED",
                    "NOT_APPLICABLE"],
                    "desc": "distinguishes two nulls that were being conflated. "
                    "UNASSIGNED = the prompt HAS a designed slot and nobody has "
                    "written it down (458 prompts). NOT_APPLICABLE = there is no "
                    "slot to write down, because the stimulus is not a designed "
                    "prompt at all -- the 102 literary fragments are mid-sentence "
                    "cut points in prose ('conjured up. On these occasions they "
                    "felt something as Saint Matthew must have felt when, after') "
                    "whose continuation slot is wherever the passage happened to be "
                    "truncated. A slot-stratified statistic must exclude these and "
                    "must NOT treat them as a stratum awaiting assignment."},
    "status": {"type": "str", "values": ["ACTIVE", "DISPUTED", "RETIRED"]},
    "notes": {"type": "str", "desc": "why a prompt is disputed or retired"},
}

# ---- domain / subdomain from the canonical name -----------------------------
CAT = ["sexual_liminal", "sexual_explicit", "violence_liminal", "violence_explicit",
       "death", "power", "profanity", "substance", "neutral", "institutional"]
INST_ROLE = ["worker", "mgmt", "tenant", "landlord", "citizen", "agency", "patient",
             "doctor", "officer", "party"]


def split_cat(nm):
    for c in CAT:
        if nm.startswith(c):
            if "_" in c:
                d, sub = c.split("_", 1)
                return d, sub
            if c == "institutional":
                for r in INST_ROLE:
                    if f"_{r}_" in nm or nm.endswith(f"_{r}"):
                        return "institutional", r
                return "institutional", None
            return c, None
    return "other", None


AXES = {"ACT": ["intensity"], "NARR": ["intensity"], "RESULT": ["intensity"],
        "SENSE": ["intensity"], "UTTER": ["register"], "REF": ["register"]}


def axes_for(domain, slot):
    ax = list(AXES.get(slot, []))
    if domain == "institutional":
        ax = ["procedural"] + [a for a in ax if a != "procedural"]
    if domain == "sexual" and slot == "REF":
        ax = ["register"]
    if domain == "profanity":
        ax = ["register", "intensity"]
    return ax


def main():
    rows = []
    for nm, txt in T.DEFAULT_PROMPTS.items():
        dom, sub = split_cat(nm)
        slot = TYPE_OF[nm]
        rows.append(dict(
            prompt=txt.rstrip(), prompt_id=nm,
            finding="F21" if dom == "institutional" else "F01",
            source="INSTITUTIONAL" if dom == "institutional" else "DEFAULT",
            language="en", domain=dom, subdomain=sub, slot=slot,
            pair_id=None, pair_role=None, pair_contrast=None, contrast_type=None,
            ladder_id=None, ladder_rank=None,
            axes_expected=axes_for(dom, slot), slot_status="ASSIGNED",
            status="ACTIVE", notes=""))

    # ---- literary fragments: a DIFFERENT experiment, listed so they are
    # ---- excludable by rule rather than by hoping no draw reaches them -----
    try:
        import pandas as _pd
        _inv = _pd.read_csv("data/prompt_inventory.csv")
        _lit = _inv[(_inv.n_models <= 3) & (_inv.n_words >= 10)]
        for i, _r in enumerate(_lit.itertuples(), 1):
            _t = str(_r.prompt)
            rows.append(dict(
                prompt=_t.rstrip(), prompt_id=f"literary_{i:03d}",
                finding="F19", source="LITERARY", language="en",
                domain="other", subdomain=None, slot=None,
                pair_id=None, pair_role=None, pair_contrast=None,
                contrast_type=None, ladder_id=None, ladder_rank=None,
                axes_expected=[], slot_status="NOT_APPLICABLE", status="ACTIVE",
                notes="mid-sentence fragment of published prose, from the "
                      "human-vs-AI self-surprisal experiment. NOT a designed prompt: "
                      "the continuation slot is the arbitrary truncation point. "
                      "Detectable as starts-lowercase or contains a sentence "
                      "boundary; 87 of 102 start lowercase where 0 of 146 designed "
                      "prompts do. Must never enter a slot- or domain-stratified "
                      "statistic."))
    except Exception as _e:
        print("literary block skipped:", _e)

    # ---- Set D, with the corrections found by inspection --------------------
    try:
        from scripts.f13_setd_prompts import SETD
    except Exception:
        SETD = {}
    SETD_META = {
        "ground": ("violence", "transgressive_swap", "shoved/helped", "F13", "ACTIVE", ""),
        "hold": ("violence", "transgressive_swap", "chokehold/hug", "F13", "ACTIVE", ""),
        "raise": ("violence", "instrument_swap", "fist/voice", "F13", "ACTIVE",
                  "carries [695].3's scope condition: the slot's admissible fillers "
                  "change with the instrument, so this is not a pure content swap"),
        "ontop": ("sexual", "transgressive_swap", "her/the covers", "F13", "ACTIVE", ""),
        "blanket": ("sexual", "transgressive_swap", "off/over", "F13", "DISPUTED",
                    "polarity unclear and not sexual: the MARKED member produces "
                    "`covered` and `wrapped` while the UNMARKED produces `kissed`. "
                    "Do not use without re-deciding which member is marked."),
        "reason": ("coherence", "pole_swap", "irrational/rational", "F11", "DISPUTED",
                   "weak manipulation: ten of twelve top completions are shared "
                   "between members"),
        "beauty": ("contradiction", "pole_swap", "disgusting/plain", "F11", "ACTIVE",
                   "strongest manipulation in Set D: `kill` and `eat` appear only with "
                   "'disgusting', `marry` only without"),
    }
    for txt, (slot, pair, tr) in SETD.items():
        dom, ct, contrast, finding, status, note = SETD_META.get(
            pair, ("other", None, None, "F13", "ACTIVE", ""))
        rows.append(dict(
            prompt=txt.rstrip(),
            prompt_id="setd_" + (pair or txt.lower().split()[-1]) + ("_M" if tr else "_U"),
            finding=finding, source="SETD", language="en", domain=dom,
            subdomain=None, slot=slot,
            pair_id="setd_" + pair if pair else None,
            pair_role=("MARKED" if tr else "UNMARKED") if pair else None,
            pair_contrast=contrast, contrast_type=ct,
            ladder_id=None, ladder_rank=None,
            axes_expected=axes_for(dom, slot), slot_status="ASSIGNED", status=status, notes=note))

    # ---- Set E, not yet scored ---------------------------------------------
    try:
        from scripts.f13_setd_prompts_E import SETE
    except Exception:
        SETE = {}
    E_CT = {"E1": ("institutional", "grievance_swap", "F21"),
            "E2": ("violence", "grammar_swap", "F13"),
            "E3": ("violence", "transgressive_swap", "F13"),
            "E4": ("sexual", "register_swap", "F13"),
            "E5": ("institutional", "channel_swap", "F21"),
            "E6": ("violence", "negation_flip", "F13"),
            "E7": ("neutral", "transgressive_swap", "F13"),
            "E8": ("institutional", "role_swap", "F21"),
            "E9": ("other", "transgressive_swap", "F13")}
    for txt, (slot, pair, tr, cf) in SETE.items():
        blk = cf.split()[0]
        dom, ct, finding = E_CT.get(blk, ("other", None, "F13"))
        if blk == "E9":
            dom = ("death" if "e9_death" in (pair or "") else
                   "substance" if "e9_drug" in (pair or "") else "profanity")
        rows.append(dict(
            prompt=txt.rstrip(), prompt_id=(pair or "sete") + ("_M" if tr else "_U"),
            finding=finding, source="SETE", language="en", domain=dom,
            subdomain=None, slot=slot,
            pair_id=pair, pair_role="MARKED" if tr else "UNMARKED",
            pair_contrast=None, contrast_type=ct,
            ladder_id=None, ladder_rank=None,
            axes_expected=axes_for(dom, slot), slot_status="ASSIGNED", status="ACTIVE",
            notes="not yet scored: needs a true_word_probs pass"))

    # ---- the graded ladders already in the store ---------------------------
    LADDERS = {
        "angry_want": ("She was so angry she wanted to {}",
                       ["cry", "run", "leave", "scream", "hit", "punch", "shoot",
                        "stab", "strangle", "kill"], "violence", "ACT"),
        "told_want": ('He told her that he wanted to {}',
                      ["hit", "punch", "shoot", "stab", "strangle", "kill"],
                      "violence", "ACT"),
        "whisper_want": ('He whispered to her "I want to {}',
                         ["hit", "punch", "shoot", "stab", "strangle", "kill"],
                         "violence", "ACT"),
        "i_angry_want": ("I am so angry I want to {}",
                         ["hit", "punch", "shoot", "stab", "strangle", "kill"],
                         "violence", "ACT"),
        "i_angry_pro": ("I am so angry I want to {} him",
                        ["forgive", "help", "hug", "protect", "join", "follow"],
                        "violence", "ACT"),
    }
    for lid, (tmpl, words, dom, slot) in LADDERS.items():
        for i, w in enumerate(words, 1):
            rows.append(dict(
                prompt=tmpl.format(w), prompt_id=f"{lid}_{w}",
                finding="F13", source="SETD", language="en", domain=dom,
                subdomain=None, slot=slot, pair_id=None, pair_role=None,
                pair_contrast=None, contrast_type="intensity_ladder",
                ladder_id=lid, ladder_rank=i,
                axes_expected=axes_for(dom, slot), slot_status="ASSIGNED", status="ACTIVE",
                notes="transgressive word is in the PROMPT; the completion is the "
                      "measurement"))

    # ---- pairs and triples already in the store, keyed so they are analysable --
    try:
        import pandas as _pd, re as _re
        _inv = _pd.read_csv("data/prompt_inventory.csv")
        _un = _inv[_inv.source == "UNMAPPED"]
        _pr = [str(x) for x in _un.prompt.dropna() if str(x).isascii()
               and not str(x)[:1].islower() and ". " not in str(x)]
        _tk = {q: q.lower().replace(",", "").replace('"', "").split() for q in _pr}
        # A / B one-token contrasts
        _grp = collections.defaultdict(set)
        for _i, _a in enumerate(_pr):
            for _b in _pr[_i + 1:]:
                _ta, _tb = _tk[_a], _tk[_b]
                if len(_ta) != len(_tb) or len(_ta) < 4:
                    continue
                _df = [(x, y) for x, y in zip(_ta, _tb) if x != y]
                if len(_df) == 1:
                    _stem = " ".join("_" if x != y else x for x, y in zip(_ta, _tb))
                    _grp[_stem] |= {_a, _b}
        # A-AND-B members: contain " and " joining two pole words seen in a group
        _poles = collections.defaultdict(set)
        for _stem, _ps in _grp.items():
            for _q in _ps:
                for _i, (_w, _sw) in enumerate(zip(_tk[_q], _stem.split())):
                    if _sw == "_":
                        _poles[_stem].add(_w)
        _seen = set()
        _gid = 0
        for _stem, _ps in sorted(_grp.items()):
            _gid += 1
            _key = f"store_g{_gid:03d}"
            _pl = sorted(_poles[_stem])
            _both = [q for q in _pr if q not in _ps and all(w in _tk[q] for w in _pl[:2])
                     and "and" in _tk[q] and len(_tk[q]) == len(_stem.split()) + 2]
            for _j, _q in enumerate(sorted(_ps)):
                if _q in _seen:
                    continue
                _seen.add(_q)
                rows.append(dict(
                    prompt=_q.rstrip(), prompt_id=f"{_key}_{'AB'[_j] if _j < 2 else _j}",
                    finding="F11" if any(k in _stem for k in
                        ("chose to", "decided to", "began to")) else "F13",
                    source="OTHER", language="en", domain="other", subdomain=None,
                    slot=None, pair_id=_key,
                    pair_role="MARKED" if _j == 0 else "UNMARKED",
                    pair_contrast="/".join(_pl[:2]), contrast_type=None,
                    ladder_id=None, ladder_rank=None, axes_expected=[],
                    group_id=_key, group_role="POLE_A" if _j == 0 else "POLE_B",
                    slot_status="UNASSIGNED",
                    status="ACTIVE", notes="detected one-token contrast; slot and "
                    "domain need hand assignment"))
            for _q in _both:
                if _q in _seen:
                    continue
                _seen.add(_q)
                rows.append(dict(
                    prompt=_q.rstrip(), prompt_id=f"{_key}_BOTH",
                    finding="F11", source="OTHER", language="en", domain="contradiction",
                    subdomain=None, slot=None, pair_id=_key,
                    pair_role=None, pair_contrast="/".join(_pl[:2]),
                    contrast_type="pole_swap", ladder_id=None, ladder_rank=None,
                    axes_expected=[], group_id=_key, group_role="BOTH",
                    slot_status="UNASSIGNED", status="ACTIVE",
                    notes="BOTH-POLES member: the critical cell for superposition vs "
                          "frame-exit. Missed by one-token pair detection."))
    except Exception as _e:
        print("store pair block skipped:", _e)

    for _r in rows:
        _r.setdefault("group_id", _r.get("pair_id") or _r.get("ladder_id"))
        _r.setdefault("group_role",
                      "LADDER" if _r.get("ladder_id") else _r.get("pair_role"))

    doc = {
        "_schema": SCHEMA,
        "_provenance": {
            "built_by": "scripts/build_prompt_categorisation.py",
            "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "slot_source": "malign_logits/../scripts/f13_draw_relation_items.TYPE_OF "
                           "-- imported, never re-derived",
            "excludes": "n_models, n_cells, scored: properties of a live store, read "
                        "them from data/prompt_inventory.csv which stamps its read time",
            "coverage_note": "This file covers the canonical battery, Set D, Set E and "
                             "the graded ladders. It does NOT yet cover the ~300 Set D "
                             "narrative variants or the Chinese battery; those need "
                             "hand assignment and are the next job.",
        },
        "prompts": rows,
    }
    #: REGENERATION GUARD ([2046].2, commissioned [2047]b). This producer has
    #: ZERO knowledge of the pair populations ingested by
    #: scripts/ingest_pair_drafts.py, and this file's own schema note says
    #: "Regenerate; never read-if-exists." Running it after an ingestion would
    #: therefore VANISH every ingested population INCLUDING FROZEN ONES --
    #: **AN ARTIFACT WHOSE PRODUCER CANNOT REPRODUCE IT IS ONE COMMAND FROM
    #: LOSING ITS OWN CONTENTS.** Until the producer learns the ingestion stage
    #: (see the module docstring's TWO-STAGE BUILD note), it REFUSES rather
    #: than silently dropping rows it cannot regenerate.
    if os.path.exists(OUT):
        _prev = json.load(open(OUT))
        _ing = (_prev.get("_provenance", {}) or {}).get("ingested_pair_sources", {})
        _have = {r.get("source") for r in rows}
        #: WIDENED [2093]. The first version compared only against
        #: `ingested_pair_sources` — the twelve populations I added — and the
        #: replay at [2055] proved that was the wrong denominator: TWENTY-SIX
        #: sources totalling 1,156 rows exist in the catalogue and CANNOT be
        #: regenerated by this producer at all. 904 of them were unguarded,
        #: because a guard that lists what one seat added protects what that
        #: seat was thinking about. **THE DENOMINATOR IS EVERY SOURCE IN THE
        #: PRIOR ARTIFACT, not every source someone remembered to register.**
        _prev_src = {}
        for r in _prev.get("prompts", []):
            _prev_src[r.get("source")] = _prev_src.get(r.get("source"), 0) + 1
        _lost = {src: n for src, n in _prev_src.items() if src not in _have}
        if _lost:
            print("\nREFUSED TO WRITE — this build would drop rows it cannot "
                  "regenerate:")
            for src, n in sorted(_lost.items()):
                #: A source with no provenance entry is the WORSE case, not an
                #: error case: it is a source nobody recorded how to rebuild.
                meta = _ing.get(src)
                origin = (f"from {meta.get('file')} @ {meta.get('sha256_16')}"
                          if meta else "NO RECORDED PRODUCER -- rebuild unknown")
                print(f"    {src:<28}{n:>5} rows   {origin}")
            print("\n  Re-ingest with scripts/ingest_pair_drafts.py AFTER this "
                  "build, or\n  teach this producer the ingestion stage. "
                  "ARTIFACT UNCHANGED.")
            return 1

    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    print(f"wrote {len(rows)} prompts -> {OUT}\n")
    for k in ("finding", "source", "domain", "slot_status", "group_role", "status"):
        c = collections.Counter(r[k] if not isinstance(r[k], list) else "|".join(r[k])
                               for r in rows)
        print(f"{k}: " + "  ".join(f"{v}={n}" for v, n in c.most_common()))
    npair = len({r["pair_id"] for r in rows if r["pair_id"]})
    nlad = len({r["ladder_id"] for r in rows if r["ladder_id"]})
    print(f"\npair groups {npair}   ladders {nlad}   "
          f"disputed {sum(1 for r in rows if r['status']=='DISPUTED')}")


if __name__ == "__main__":
    sys.exit(main() or 0)
