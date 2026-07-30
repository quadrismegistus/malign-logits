"""Invariants for data/prompt_categorisation.json.

WHY THIS FILE EXISTS. Every defect found in the categorisation on 2026-07-30 was
RELATIONAL -- invisible in any single row, visible only by cross-checking the file
against something outside it (the hand-written Set D/E declarations, the prompt's own
word order, the build script's source, the census). Four for four. A per-row reader,
human or model, does not encounter them: `group_role: POLE_A` is a valid value on
every row that carries it, and `(POLE_A, POLE_B)` is a valid pair in every group that
has one. The error lives between rows.

That is the argument for assertions over inspection. A manual pass finds these once;
a test keeps finding them, including after the next merge re-introduces one.

Each test below names the defect it exists to prevent and the docket entry that found
it. SEVERAL FAIL AS WRITTEN -- that is deliberate. The file is mid-cleanup and the
failing tests ARE the worklist: run this suite to get the rows that need attention,
rather than reading 823 rows to find them.

    uv run .venv/bin/python -m pytest tests/test_prompt_categorisation.py -v

THE ONE TO KEEP IF YOU KEEP NOTHING ELSE is `test_pole_labels_are_not_a_sort_artifact`.
It caught the worst defect of the day: `group_role` and `pair_role` were both assigned
by `enumerate(sorted(prompts))`, so a field whose values are literally MARKED and
UNMARKED was set by alphabetical accident and read backwards in 5 of 12 checkable
cases (`leaned` labelled MARKED, `slammed` labelled UNMARKED, because `leaned` sorts
first). Nothing raised, because the defect biases results TOWARD THE NULL -- it
produces a weak number rather than an impossible one, and every other check in this
project's apparatus is triggered by impossible numbers. That generalises: a label that
perfectly predicts the alphabetical order of its members carries no information, and
the test for it is cheap.
"""
from __future__ import annotations

import collections
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "prompt_categorisation.json")


@pytest.fixture(scope="module")
def doc():
    with open(CAT) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def rows(doc):
    """ACTIVE rows only. A RETIRED row cannot be consumed by an analysis, so holding it
    to a structural invariant is holding the wrong object to it.

    This was a defect in the first version of these tests, and it surfaced the moment
    the retirements ran: the duplicate test reported all 51 duplicates still present
    when 50 of them had just been retired, and the DISPUTED test reported a bypass by a
    row whose status was RETIRED -- which is the opposite of a bypass. Both were the
    test counting rows the file had already excluded.

    `all_rows` is available where the invariant genuinely covers retired rows too
    (uniqueness of prompt_id, declared vocabularies).
    """
    return [r for r in doc["prompts"] if r.get("status") != "RETIRED"]


@pytest.fixture(scope="module")
def all_rows(doc):
    return doc["prompts"]


@pytest.fixture(scope="module")
def declared_markedness():
    """Markedness as HAND-DECLARED in Set D/E, RESTRICTED TO DECLARED PAIR MEMBERS.

    THE RESTRICTION IS THE POINT AND IT IS A CORRECTION OF MY OWN FIRST VERSION.
    Set E's docstring defines the flag as "the MARKED member of a pair, not a claim
    about harm". For a prompt with `pair_id=None` there is no pair, so the flag is
    UNDEFINED BY THE DECLARATION'S OWN DEFINITION -- and Set D has ten such singletons
    at the end of its dict, three flagged True and seven False, e.g.

        "He was captive and chose to":            ("ACT",  None, False)
        "He leaned her against the wall and":     ("NARR", None, True)
        "He slammed her against the wall and":    ("NARR", None, True)

    `captive` is flagged unmarked and `leaned` is flagged marked, which are both
    defensible for an unpaired prompt and both meaningless as markedness.

    My first version read all of these as ground truth and reported SIX contradictions.
    Four of the six were singletons, so the test was measuring an undefined flag and
    calling the file wrong. Restricting to declared pairs leaves TWO, both real, both
    in `setd_ground`: `helped` labelled MARKED and `shoved` labelled UNMARKED.

    This does not weaken the case against `pair_role`. That case rests on the field
    being `str.sort` at 32/32, which is independent and decisive; the markedness
    comparison was corroboration, and it was the contaminated leg.
    """
    from scripts.f13_setd_prompts import SETD
    from scripts.f13_setd_prompts_E import SETE
    out = {}
    for src in (SETD, SETE):
        for text, spec in src.items():
            if spec[1] is None:          # no pair_id -> markedness undefined
                continue
            out[text.strip()] = bool(spec[2])
    return out


@pytest.fixture(scope="module")
def census_stashes():
    """text -> number of caches the prompt appears in, from the census.

    The census is the record of what was actually SCORED, so it is the external check
    on the file's own `apparatus` field.
    """
    import csv
    path = os.path.join(os.path.dirname(CAT), "prompt_census_all.csv")
    if not os.path.exists(path):
        pytest.skip("prompt_census_all.csv absent")
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            text = (row.get("prompt") or "").rstrip()
            raw = row.get("n_stashes") or row.get("stashes") or ""
            try:
                out[text] = int(raw)
            except ValueError:
                out[text] = len([s for s in raw.split("|") if s.strip()])
    return out


def groups(rows, prefix=None):
    by = collections.defaultdict(list)
    for r in rows:
        g = r.get("group_id")
        if isinstance(g, str) and (prefix is None or g.startswith(prefix)):
            by[g].append(r)
    return by


# --------------------------------------------------------------------------
# 1. IDENTITY AND UNIQUENESS
# --------------------------------------------------------------------------

def test_prompt_id_is_unique(all_rows):
    dup = [k for k, n in collections.Counter(r["prompt_id"] for r in all_rows).items() if n > 1]
    assert not dup, f"{len(dup)} duplicated prompt_id: {dup[:10]}"


def test_no_prompt_string_repeats_within_one_finding(rows):
    """[765]/[766]: 56 strings appear twice. 51 share a `finding` -- those are ONE
    design written by TWO build paths (hand declaration + auto-mapper), carrying two
    group ids and two role vocabularies. Only 5 are genuine dual membership across
    findings, and those are legitimate: `He held her underwater until she stopped` is
    really `violence_explicit_5` in F01 and `e6_water_M` in F13.

    So the invariant is NOT "prompt is unique" -- the prompt is deliberately not a
    key. It is: a prompt may appear twice only under DIFFERENT findings.
    """
    AUTO = lambda r: (str(r.get("prompt_id", "")).startswith("store_g")
                      or r.get("source") == "OTHER")
    seen = collections.defaultdict(list)
    for r in rows:
        seen[(r["prompt"].strip(), r.get("finding"))].append(r)
    bad = {}
    for k, group in seen.items():
        if len(group) < 2:
            continue
        # A prompt may serve TWO GENUINE DESIGNS that happen to share a finding. The
        # case that forced this: `My boss takes credit for my work in every meeting. I
        # should` is BOTH the canonical F21 battery member `institutional_labor_worker_2`
        # AND Set E's E1 pair member `e1_credit_M`. Same finding, two designs, and
        # retiring either destroys a real identity. The signature is that neither row is
        # auto-mapped and they come from different declared sources.
        if not any(AUTO(r) for r in group) and len({r.get("source") for r in group}) > 1:
            continue
        bad[k] = [r["prompt_id"] for r in group]
    assert not bad, (
        f"{len(bad)} prompt strings duplicated WITHIN one finding by two build paths "
        f"(not dual design). First few: "
        + "; ".join(f"{k[1]} {k[0][:40]!r} -> {v}" for k, v in list(bad.items())[:5]))


# --------------------------------------------------------------------------
# 2. THE SORT-ARTIFACT CHECK -- the one that matters most
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field,first_value", [
    ("group_role", "POLE_A"),
    ("pair_role", "MARKED"),
])
def test_pole_labels_are_not_a_sort_artifact(rows, field, first_value):
    """A two-valued label that PERFECTLY predicts alphabetical order carries no
    information about the design -- it is `str.sort` under a semantic name.

    [772]/[774]: both fields were assigned by `enumerate(sorted(prompts))` in
    build_prompt_categorisation.py:364, measured at 32/32. `pair_role` is the worse
    of the two because MARKED/UNMARKED is an explicit claim in plain English, so a
    reader takes it at face value, whereas POLE_A is opaque and needs a specific
    wrong join to do damage.

    This test is deliberately statistical rather than exact: perfect agreement across
    many groups is the signal. A few groups agreeing by chance is expected, since the
    alphabet sometimes matches the design.
    """
    hits = total = 0
    examples = []
    for gid, members in groups(rows).items():
        firsts = [r for r in members if r.get(field) == first_value]
        if len(members) != 2 or len(firsts) != 1:
            continue
        total += 1
        alpha_first = sorted(members, key=lambda r: r["prompt"])[0]
        if alpha_first is firsts[0]:
            hits += 1
            if len(examples) < 4:
                examples.append(f"{gid}: {first_value} = {firsts[0]['prompt'][:44]!r}")
    if total < 8:
        pytest.skip(f"only {total} two-member groups carry {field}; too few to judge")
    assert hits < total, (
        f"{field} == alphabetically-first prompt in {hits}/{total} two-member groups. "
        f"The label is a sort artifact and carries no design information. "
        + " | ".join(examples))


def test_pair_role_matches_declared_markedness(rows, declared_markedness):
    """`pair_role` claims to be markedness. Where Set D/E declares markedness for the
    same prompt, the two must agree. [772]: they disagreed in 5 of 12 checkable cases.

    This one cannot be repaired by a reading convention, because the field IS the
    markedness field -- hence the ruling to delete it on auto-mapped rows rather than
    reinterpret it.
    """
    wrong = []
    for r in rows:
        text = r["prompt"].strip()
        role = r.get("pair_role")
        if text in declared_markedness and role in ("MARKED", "UNMARKED"):
            if (role == "MARKED") != declared_markedness[text]:
                wrong.append(f"{r.get('group_id')}: pair_role={role} but Set D/E "
                             f"declares {'MARKED' if declared_markedness[text] else 'unmarked'}"
                             f" -- {text[:44]!r}")
    assert not wrong, f"{len(wrong)} rows contradict declared markedness: " + " | ".join(wrong[:6])


def test_pole_a_word_precedes_pole_b_word_in_the_both_prompt(rows):
    """[768]/[769]: POLE_A/POLE_B names TWO constructs in this repo -- which pole the
    PROMPT states (group_role, here) and which pole the COMPLETION resolves toward
    (tag_primary in tag_contradictions.py). F11's measurement joins them, so they must
    agree on which pole is "first", and the tagger defines first by PROMPT WORD ORDER.

    The f11_* maps currently satisfy this 15/15, but that is because they were written
    in each pair's natural reading order, not because anything enforced it. A future
    map written the other way inverts the sign for one group silently.
    """
    pairs = {}
    for gid, members in groups(rows, "f11_").items():
        contrast = next((r.get("pair_contrast") for r in members if r.get("pair_contrast")), None)
        if contrast and "/" in contrast:
            pairs[gid] = contrast.split("/", 1)
    bad = []
    for r in rows:
        gid = r.get("group_id")
        if r.get("group_role") != "BOTH" or gid not in pairs:
            continue
        a, b = (w.lower() for w in pairs[gid])
        text = r["prompt"].lower()
        ia, ib = text.find(a), text.find(b)
        if ia < 0 or ib < 0:
            bad.append(f"{gid}: pole word absent from BOTH prompt {r['prompt'][:44]!r}")
        elif ia > ib:
            bad.append(f"{gid}: POLE_A={a!r} appears AFTER POLE_B={b!r} in "
                       f"{r['prompt'][:44]!r} -- sign inverted for this group")
    assert not bad, "; ".join(bad[:6])


# --------------------------------------------------------------------------
# 3. GROUP SHAPE
# --------------------------------------------------------------------------

def test_pole_groups_have_exactly_one_of_each_pole(rows):
    """[776].2: 13 of 45 auto-mapped groups are not pair-shaped -- one is 1 POLE_A
    against 9 POLE_B, and (1,5) recurs seven times, which means the mapper matched one
    prompt against a whole stem family rather than against a partner. The GROUP is
    spurious, not merely its labels.
    """
    bad = []
    for gid, members in sorted(groups(rows).items()):
        n = collections.Counter(r.get("group_role") for r in members)
        if not (n["POLE_A"] or n["POLE_B"]):
            continue
        if n["POLE_A"] != 1 or n["POLE_B"] != 1:
            bad.append(f"{gid}: POLE_A={n['POLE_A']} POLE_B={n['POLE_B']} rows={len(members)}")
    assert not bad, f"{len(bad)} groups are not 1-A-and-1-B: " + ", ".join(bad[:14])


def test_marked_unmarked_groups_have_exactly_one_of_each(rows, declared_markedness):
    """Two-member contrasts need one marked and one unmarked member.

    THE EXEMPTION IS DERIVED FROM THE DECLARATION, and that is the second correction
    to this test. Set E's E2 puts one scene into three slot grammars (every member
    marked; the contrast is GRAMMAR) and E7 is deliberately non-transgressive with an
    intensity gradient (no member marked). Those six groups are CORRECT.

    My first version exempted them by `contrast_type`, which failed because the e7
    groups do not carry an exempting value -- and the tempting fix was to relabel them
    `intensity_ladder` so the test would pass. That would have been mislabelling data
    to satisfy a test, which is worse than the failure. So the exemption now asks the
    DECLARATION whether markedness is the manipulation at all: if Set D/E gives every
    member of the group the same flag, the contrast is something else and markedness
    balance is not a meaningful property of it.

    [765].7 is the original error this guards -- I flagged all six as malformed by
    applying a markedness template to groups declaring a different one, which is the
    error `contrast_type` exists to prevent, committed by whoever added the field.
    """
    bad = []
    for gid, members in sorted(groups(rows).items()):
        n = collections.Counter(r.get("group_role") for r in members)
        if not (n["MARKED"] or n["UNMARKED"]):
            continue
        flags = {declared_markedness[m["prompt"].strip()] for m in members
                 if m["prompt"].strip() in declared_markedness}
        if len(flags) == 1:
            continue                    # declaration says markedness is not the axis
        if n["MARKED"] != 1 or n["UNMARKED"] != 1:
            bad.append(f"{gid}: MARKED={n['MARKED']} UNMARKED={n['UNMARKED']}")
    assert not bad, f"{len(bad)} markedness groups malformed: " + ", ".join(bad[:12])


def test_f11_triples_are_complete_or_declared_incomplete(rows):
    """F11's design is POLE_A + POLE_B + BOTH. `f11_gender` genuinely lacks its single
    pole prompts -- they do not exist in any set -- so it is the one declared gap.

    Two groups carry TWO BOTH cells (`f11_captive`, `f11_gender`); [768].4(a) confirmed
    that is within-group replication of the same contradiction, not over-matching.
    """
    known_incomplete = {"f11_gender"}
    bad = []
    for gid, members in sorted(groups(rows, "f11_").items()):
        n = collections.Counter(r.get("group_role") for r in members)
        if gid in known_incomplete:
            continue
        if not (n["POLE_A"] and n["POLE_B"] and n["BOTH"]):
            bad.append(f"{gid}: {dict(n)}")
    assert not bad, f"{len(bad)} F11 groups incomplete: " + ", ".join(bad)


# --------------------------------------------------------------------------
# 4. CONTROLLED VOCABULARIES
# --------------------------------------------------------------------------

def test_domain_values_are_declared_in_the_schema(all_rows, doc):
    """`domain=sensation` exists in 4 rows and is not a declared value, while the
    declared `class` has none.
    """
    allowed = set(doc["_schema"]["domain"]["values"])
    seen = collections.Counter(r.get("domain") for r in all_rows)
    undeclared = {k: n for k, n in seen.items() if k is not None and k not in allowed}
    assert not undeclared, f"undeclared domain values in use: {undeclared}"


def test_subdomain_uses_one_vocabulary(rows):
    """The build script and the merge script each invented a naming scheme, so the
    file carries both: worker/labor_worker, mgmt/labor_mgmt, tenant/housing_tenant,
    citizen/govt_citizen, agency/govt_agency, officer/police_officer.

    A groupby on subdomain therefore SPLITS every institutional cell in half -- and
    institutional proceduralisation is the finding that runs opposite to violence
    attenuation, so it is the cell we can least afford to halve.
    """
    seen = {r.get("subdomain") for r in rows if r.get("subdomain")}
    collisions = sorted(
        (bare, qual) for bare in seen for qual in seen
        if qual != bare and qual.endswith("_" + bare))
    assert not collisions, (
        f"{len(collisions)} subdomain values exist in both a bare and a qualified "
        f"form: {collisions}")


def test_contrast_type_values_are_declared(rows, doc):
    allowed = set(doc["_schema"]["contrast_type"]["values"])
    undeclared = {r.get("contrast_type") for r in rows
                  if r.get("contrast_type") and r["contrast_type"] not in allowed}
    assert not undeclared, f"undeclared contrast_type values: {sorted(undeclared)}"


# --------------------------------------------------------------------------
# 5. FIELD CO-DEPENDENCIES
# --------------------------------------------------------------------------

def test_slot_status_agrees_with_slot(rows):
    bad_na = [r["prompt_id"] for r in rows
              if r.get("slot_status") == "NOT_APPLICABLE" and r.get("slot") is not None]
    bad_as = [r["prompt_id"] for r in rows
              if r.get("slot_status") == "ASSIGNED" and r.get("slot") is None]
    assert not bad_na, f"{len(bad_na)} rows are NOT_APPLICABLE but carry a slot: {bad_na[:8]}"
    assert not bad_as, f"{len(bad_as)} rows are ASSIGNED but have no slot: {bad_as[:8]}"


def test_ladder_id_and_rank_travel_together(rows):
    bad = [r["prompt_id"] for r in rows
           if (r.get("ladder_id") is None) != (r.get("ladder_rank") is None)]
    assert not bad, f"{len(bad)} rows have one of ladder_id/ladder_rank without the other: {bad[:8]}"


def test_ladder_ranks_are_contiguous_from_one(rows):
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("ladder_id") is not None and r.get("ladder_rank") is not None:
            by[r["ladder_id"]].append(int(r["ladder_rank"]))
    bad = [f"{lid}: {sorted(v)}" for lid, v in by.items()
           if sorted(v) != list(range(1, len(v) + 1))]
    assert not bad, f"{len(bad)} ladders are not 1..n: " + "; ".join(bad[:6])


@pytest.mark.parametrize("field", ["finding", "source", "apparatus", "group_role", "slot"])
def test_enum_fields_only_use_declared_values(all_rows, doc, field):
    """Generalises the domain check to every field the schema gives a value list for.

    The audit found `finding=F19` on 109 rows and `source` in {UNMAPPED, CENSUS} on
    308 rows -- 37% of the file -- none of them declared. Undeclared values are not
    harmless: the grid spec reports its own composition BY `finding`, so a vocabulary
    that the schema does not know about is being used as an axis.
    """
    spec = doc["_schema"].get(field)
    if not isinstance(spec, dict) or "values" not in spec:
        pytest.skip(f"{field} has no declared value list")
    allowed = set(spec["values"])
    seen = collections.Counter(r.get(field) for r in all_rows)
    bad = {k: n for k, n in seen.items() if k is not None and k not in allowed}
    assert not bad, f"{field} uses undeclared values: {bad}"


def test_chinese_prompts_are_labelled_chinese(rows):
    """The audit's most serious finding. 73 Chinese prompts are PRESENT but carry
    `language="en"`, and no row in the file carries `language="zh"` at all -- so no
    consumer can filter them out using the field that exists for exactly that.

    Worse than absent: the file's own coverage_note says the Chinese battery is not yet
    covered, which is false, so a reader has no reason to check.
    """
    zh = [r for r in rows if r.get("source") == "CHINESE"]
    if not zh:
        pytest.skip("no rows marked source=CHINESE")
    wrong = [r["prompt_id"] for r in zh if r.get("language") != "zh"]
    assert not wrong, (f"{len(wrong)} of {len(zh)} source=CHINESE rows are labelled "
                       f"language={zh[0].get('language')!r}: {wrong[:6]}")


def test_chinese_prompts_are_not_uniformly_one_slot(rows):
    """All 73 fell through `slot_of()`, whose regexes are English-only, to the NARR
    default. The correct split (ACT 34 / REF 12 / NARR 11 / UTTER 9 / SENSE 4 /
    RESULT 3) is already sitting in prompt_inventory.csv's `slot` column, which the
    merge script reads for `category` and not for `slot`.

    A slot-stratified statistic silently gains 73 mis-slotted prompts in NARR.
    """
    zh = [r for r in rows if r.get("source") == "CHINESE" and r.get("slot")]
    if len(zh) < 10:
        pytest.skip("too few slotted Chinese rows to judge")
    seen = collections.Counter(r["slot"] for r in zh)
    assert len(seen) > 1, (
        f"all {len(zh)} Chinese prompts carry slot={list(seen)[0]!r} -- the English-only "
        f"slot rule fell through to its default for every one of them")


def test_duplicate_strings_agree_on_slot(rows):
    """Two build paths, one string, two slots. The audit found ~40 rows of the
    desiderative-ladder family assigned NARR by the generic fallback while their
    exact-text twins carry the hand-coded ACT.

    Same text in the same file with two grammars means a slot-stratified statistic
    counts the family in BOTH strata.
    """
    by = collections.defaultdict(set)
    for r in rows:
        if r.get("slot"):
            by[r["prompt"].rstrip()].add(r["slot"])
    bad = {k: sorted(v) for k, v in by.items() if len(v) > 1}
    assert not bad, (f"{len(bad)} prompt strings carry two different slots: "
                     + "; ".join(f"{k[:40]!r} -> {v}" for k, v in list(bad.items())[:6]))


def test_scored_prompts_are_not_marked_unscored(rows, census_stashes):
    """The starvation defect, and it is the same dict-keyed-on-prompt bug that bit the
    grid builder -- in my merge script this time:

        have = {r["prompt"]: r for r in doc["prompts"]}

    When a string appears twice, the census update reaches only the last-inserted row.
    The other defaults to apparatus=UNSCORED, n_stashes=0. In five cases the starved
    row is a CORE CANONICAL prompt whose census entry shows n_stashes=11, so anyone
    filtering `apparatus=="BATTERY"` to get the real battery drops 5 of the 73.
    """
    bad = []
    for r in rows:
        n = census_stashes.get(r["prompt"].rstrip(), 0)
        if n > 0 and r.get("apparatus") == "UNSCORED":
            bad.append(f"{r['prompt_id']} (census n_stashes={n}) {r['prompt'][:36]!r}")
    assert not bad, (f"{len(bad)} rows are marked UNSCORED but the census shows they "
                     f"were scored: " + "; ".join(bad[:8]))


def test_pair_contrast_words_appear_in_the_group_prompts(rows):
    """`pair_contrast` names the tokens the design manipulates, so both should be
    findable in the group's prompts. The audit found `setd_beauty` labelled
    "disgusting/plain" where `plain` occurs in neither member -- the unmarked prompt
    simply omits `disgusting` rather than substituting a synonym. Hardcoded in
    build_prompt_categorisation.SETD_META, so no derivation would catch it.
    """
    bad = []
    for gid, members in sorted(groups(rows).items()):
        contrast = next((r.get("pair_contrast") for r in members if r.get("pair_contrast")), None)
        if not contrast or "/" not in contrast:
            continue
        text = " ".join(r["prompt"].lower() for r in members)
        missing = [w for w in contrast.split("/") if w and w.lower() not in text]
        if missing:
            bad.append(f"{gid}: pair_contrast={contrast!r} but {missing} absent from its prompts")
    assert not bad, f"{len(bad)} groups name a contrast token that is not present: " + "; ".join(bad[:6])


def test_disputed_rows_have_no_active_duplicate(rows):
    """A DISPUTED flag is decorative if the same prompt sits elsewhere as ACTIVE.
    The audit found both DISPUTED pairs bypassed this way -- `setd_blanket` via
    `store_g009`, `setd_reason` via `store_g017`/`f11_reason` -- so any pipeline
    filtering on `status` still consumes the disputed stimulus under its other id.
    """
    status = collections.defaultdict(set)
    ids = collections.defaultdict(list)
    for r in rows:
        status[r["prompt"].rstrip()].add(r.get("status"))
        ids[r["prompt"].rstrip()].append(f"{r['prompt_id']}={r.get('status')}")
    bad = [f"{k[:38]!r} {ids[k]}" for k, v in status.items()
           if "DISPUTED" in v and v - {"DISPUTED"}]
    assert not bad, f"{len(bad)} disputed prompts have a non-disputed duplicate: " + "; ".join(bad[:6])


def test_canonical_slots_match_the_single_source_of_truth(rows):
    """TYPE_OF in f13_draw_relation_items.py is declared the one source of truth for
    canonical slots. Any disagreement means a stratified statistic and the item draw
    disagree about which grammar a prompt has -- and the slot grammars are the largest
    effect in the study.
    """
    from malign_logits import taxonomy as T
    from scripts.f13_draw_relation_items import TYPE_OF
    canonical = {v.rstrip(): TYPE_OF[k] for k, v in T.DEFAULT_PROMPTS.items() if k in TYPE_OF}
    bad = []
    for r in rows:
        want = canonical.get(r["prompt"].rstrip())
        if want and r.get("slot") and r["slot"] != want:
            bad.append(f"{r['prompt_id']}: file={r['slot']} TYPE_OF={want} "
                       f"{r['prompt'][:36]!r}")
    assert not bad, f"{len(bad)} canonical prompts disagree with TYPE_OF: " + "; ".join(bad[:8])
