#!/usr/bin/env python3
"""REGISTRATION P's PRODUCER — the displacement-relation annotation.

Frozen registration: `registration_p_annotation.md`
  @ b17f4c6b255c2761ee71735bf77687144403c50397497525ab0b3a958c51e583
  committed 9cb4644df7d7696d6c9c2362250d838b1509ff26, RH's word
  "Ok -- freeze and run!"  Built at malign's seat, audited at lacan's ([4213]).

**THIS FILE IMPLEMENTS THE FROZEN TEXT AND NOTHING ELSE.** Where the text is
silent it RAISES rather than chooses.

    §P2    the roster DECLARED; identities resolved FROM THE RUN;
           ANSWERED != DECLARED IS A REFUSAL
    §P2.1  the receipt in `usage.no_reasoning_observed()` semantics — and NOT
           in the "field present and equals zero" form, which can never pass
    §P3    the unit is the (prompt, faller) KEY; one REAL riser per key by
           n_edges desc then riser alphabetical; McNemar on discordant pairs;
           n = 148 ACT / 93 REF; pooling across strata FORBIDDEN
    §P3.1  four cells: 3/3 CONFIRMED, 2/3 SPLIT, 1/3 single-coder, 0/3
    §P4.1  agreement reported BEFORE the verdicts it qualifies (clause 5)
    §P5    all 11 EXHIBIT items coded, characterization only, NO RATE
    §P6    the verdict sentences are QUOTED FROM THE FROZEN TEXT, never
           composed here

**THE CACHE DISPOSITION, [4210]:** the stash key carries `max_tokens, model,
prompt, schema, system_prompt, temperature` and **NO thinking field**, so a
pre-patch (thinking-on) entry and a post-patch call produce the SAME KEY. Every
item with a pre-existing stash entry therefore runs FORCED. Measured before the
build: 19 deepseek / 11 gpt-4o-mini / 19 sonnet-5.

    THE CACHE ACCEPTS-AND-IGNORES AT THE KEY WHAT DEEPSEEK ACCEPTS-AND-IGNORES
    AT THE API — the same silence twice, and the second survives the patch.

**NO SEED.** Nothing samples, splits or shuffles except §P5's declared shuffle,
which is a presentation order and touches no quantity.

    python meta/M01_displacement/scripts/p_primary.py [--dry-run]
"""

import collections
import hashlib
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

REGISTRATION = os.path.join(CAMPAIGN, "registrations",
                            "registration_p_annotation.md")
REGISTRATION_SHA = ("b17f4c6b255c2761ee71735bf77687144403c50397497525ab0b3a"
                    "958c51e583")
POPULATION = os.path.join(CAMPAIGN, "populations", "population_p_items.parquet")
POPULATION_SHA16 = "ce506ce9a72a0675"
INSTRUMENT_DIGEST = ("f6a92cc62dcb71efb2d3519ac578d160ab202abac7b7ba58987aa4"
                     "2e998094c3")
OUT = os.path.join(CAMPAIGN, "results", "result_p_primary.json")
ESCROW = os.path.join(CAMPAIGN, "results", "superseded")

#: §P2's DECLARED roster. Not an arity — the names, so the run cannot call
#: something else and record it faithfully.
#: **PIN-DRIFT WARNING, [4245].3.** The patch recognises a specific tuple of
#: deepseek ids. An UNRECOGNISED deepseek id gets NO thinking-disable AND
#: temperature None — strictly worse than pre-patch, and silent. Never drift
#: from `deepseek-v4-pro` without re-checking against the recognised tuple in
#: `providers.deepseek_thinking_default`.
ROSTER = ["deepseek/deepseek-v4-pro", "openai/gpt-4o-mini",
          "anthropic/claude-sonnet-5"]
WORKERS = 12                      #: RH's word, [4203]: 12 per coder arm
#: RH's word, 2026-08-04: the peer seat's final commit is `f726aea`
#: ("providers+llm: closing-review fixes"), now on MAIN. Two predecessors are
#: dead — `b3e0915` was orphaned by a reset-and-rebuild and is on no branch;
#: `0c0a983` and `7afdfc5` were superseded in turn. **THIS PIN HAS MOVED THREE
#: TIMES; the comment moves with it, because a comment calling a value FINAL
#: above a value that keeps moving is a status line that outlived its truth.**
EXPECTED_HEAD = "f726aea"
#: RH's instruction, and [4410]'s form (b): **THE HASH REMAINS THE AUTHORITY
#: AND THE TAG IS A REACHABILITY ASSERTION.** `f726aea` is an ancestor of
#: main today, but this repository ORPHANED A COMMIT ONCE ALREADY — `b3e0915`
#: was reset-and-rebuilt out of history and sits on no branch — so
#: reachability is not a property main can be trusted to keep. The tag
#: guarantees the checkout survives a rewrite.
#:
#: **THE TAG IS NOT THE PIN.** Resolving a name at gate time would make the
#: pin a MUTABLE POINTER the repo supplies, which is exactly the mutability
#: §P2's blob pins were added to close. The tag must AGREE with the hash; it
#: may not replace it, and a moved tag is a loud refusal rather than a silent
#: re-pin.
#:
#: **AND IT IS AN ANNOTATED TAG, SO IT MUST BE DEREFERENCED.** A bare
#: `git rev-parse <tag>` returns the TAG OBJECT (7e34a86f...), which compares
#: unequal to f726aea and reads exactly like a moved pin. `^{commit}` is
#: required. [4410] hit this on its first comparison.
EXPECTED_TAG = "pin-2026-08-04-registrations"
ALPHA = 0.025                     #: §P3, D2's split form

#: §P3's declared tie-break, as a comparator rather than a sentence.
def selection_key(row):
    return (-int(row.n_edges), str(row.b))


# ══════════════════════════════════════════════════════════════════════════
# GATES
# ══════════════════════════════════════════════════════════════════════════
def gate_registration():
    """`require_frozen` first, and the bytes named as well.

    The gate proves a document is frozen; only the hash proves it is THIS one.
    """
    import freeze_gate
    freeze_gate.require_frozen(REGISTRATION)
    got = hashlib.sha256(open(REGISTRATION, "rb").read()).hexdigest()
    if got != REGISTRATION_SHA:
        raise SystemExit("REFUSING: registration hashes %s, this producer was "
                         "written against %s." % (got[:16], REGISTRATION_SHA[:16]))
    print("gate     registration FROZEN and IDENTIFIED: %s" % got[:16], flush=True)


def gate_instrument():
    """§P2's dual pin, each half checked by its own function.

    The instrument digest governs POOLING — it moves on a field-description
    edit even if the file does not. The source hash pins the bytes. Neither
    substitutes for the other and a bare hex value cannot tell a reader which
    it is.
    """
    from malign_logits.tasks.code_displacement_relation import DisplacementRelationTask
    _t = DisplacementRelationTask()
    #: **[4297]: §P2 DEFINES THIS PIN TWICE, AND THE DEFINITIONS DIVERGED.**
    #:     "= DisplacementRelationTask.instrument_sha256()"
    #:     "= sha256 of instrument_text() — THE RENDERED INSTRUMENT
    #:        (system prompt, examples, field descriptions, schema as
    #:        administered)"
    #: `task.py` @ 7b41c94 widened the METHOD to cover instrument_text PLUS an
    #: item-block wrapper, so the method now returns f47a3fd4 while the
    #: QUANTITY still returns f6a92cc6. §P2 says the digest pins "what the
    #: coder was actually shown", and 12,130 stash entries say what that was.
    #:
    #: **THE FROZEN TEXT IS STILL CORRECT. `instrument_sha256()` STOPPED
    #: BEING THE FUNCTION THAT COMPUTES WHAT IT NAMES**, so the producer
    #: computes the named quantity directly and the registration does not move.
    got = hashlib.sha256(_t.instrument_text().encode("utf-8")).hexdigest()
    if got != INSTRUMENT_DIGEST:
        raise SystemExit(
            "REFUSING: sha256(instrument_text()) is %s, §P2 pins %s. The "
            "RENDERED INSTRUMENT moved — judgments from a different "
            "instrument DO NOT POOL. (If the method `instrument_sha256()` "
            "differs while this matches, the framework's DEFINITION moved and "
            "the instrument did not; that is not this refusal.)"
            % (got[:16], INSTRUMENT_DIGEST[:16]))
    #: reported, never gating: the method's value, so a reader can see the
    #: divergence rather than infer it.
    _method = _t.instrument_sha256()
    if _method != got:
        print("gate     [note] instrument_sha256() = %s — the METHOD is "
              "wrapper-inclusive AT the pinned checkout f726aea; it widened "
              "between 0c0a983 and f726aea, which 7b41c94 CARRIES in the "
              "rebuilt history and does NOT date. It is NOT the quantity §P2 "
              "pins; not a gate, printed so the divergence is visible"
              % _method[:16], flush=True)
    src = os.path.join(ROOT, "malign_logits", "tasks",
                       "code_displacement_relation.py")
    fsha = hashlib.sha256(open(src, "rb").read()).hexdigest()
    print("gate     §P2 instrument digest  %s  (sha256 of instrument_text())"
          % got[:16], flush=True)
    print("gate     §P2 source sha256      %s  (sha256 of file bytes)" % fsha[:16],
          flush=True)
    return got, fsha


def gate_population():
    """§P1's deposit, pinned by the hash the document names."""
    import pandas as pd
    sha = hashlib.sha256(open(POPULATION, "rb").read()).hexdigest()
    if sha[:16] != POPULATION_SHA16:
        raise SystemExit("REFUSING: population sha256 is %s, §P1 pins %s"
                         % (sha[:16], POPULATION_SHA16))
    d = pd.read_parquet(POPULATION)
    print("gate     population %s  %d rows" % (sha[:16], len(d)), flush=True)
    return d, sha


def known_answers(d):
    """The document's own figures, fired BEFORE any coding quantity exists."""
    R = d[d.item_class == "REAL"]
    checks = [
        ("REAL", int((d.item_class == "REAL").sum()), 2722),
        ("NEAR-MISS", int((d.item_class == "NEAR-MISS").sum()), 1710),
        ("EXHIBIT", int((d.item_class == "EXHIBIT").sum()), 11),
        ("prompts", int(R.prompt.nunique()), 685),
        ("ACT REAL", int((R.slot == "ACT").sum()), 214),
        ("REF REAL", int((R.slot == "REF").sum()), 115),
        ("ACT keys", int(R[R.slot == "ACT"].groupby(["prompt", "a"]).ngroups), 148),
        ("REF keys", int(R[R.slot == "REF"].groupby(["prompt", "a"]).ngroups), 93),
    ]
    NM = set(zip(d[d.item_class == "NEAR-MISS"].prompt,
                 d[d.item_class == "NEAR-MISS"].a))
    for s in ("ACT", "REF"):
        keys = set(R[R.slot == s].groupby(["prompt", "a"]).groups)
        checks.append(("%s unmatched keys" % s, len(keys - NM), 0))
    print("\n=== KNOWN ANSWERS (the document's own figures)", flush=True)
    for name, got, want in checks:
        ok = got == want
        print("  %-18s %6d  published %6d  %s"
              % (name, got, want, "OK" if ok else "** MISMATCH **"), flush=True)
        if not ok:
            raise SystemExit(
                "REFUSING: %s is %d and §P1 publishes %d. The producer and the "
                "frozen text disagree about the POPULATION; no coding quantity "
                "may be read past this line." % (name, got, want))
    print("  ALL KNOWN ANSWERS MATCH.", flush=True)


# ══════════════════════════════════════════════════════════════════════════
# §P3 — the paired selection
# ══════════════════════════════════════════════════════════════════════════
def select_pairs(d):
    """One REAL riser per (prompt, faller) key, plus that key's single decoy.

    §P3's tie-break is DECLARED and load-bearing: it resolves a real tie on 16
    of 241 keys (11 ACT, 5 REF), so on 7% of the ACT primary the item is
    decided by alphabetical order of the riser word. Legitimate only because
    it was declared before any coding existed.
    """
    R = d[d.item_class == "REAL"]
    NM = {(r.prompt, r.a): r for r in d[d.item_class == "NEAR-MISS"].itertuples()}
    out = {}
    for s in ("ACT", "REF"):
        rows = []
        for key, g in R[R.slot == s].groupby(["prompt", "a"]):
            best = sorted(g.itertuples(), key=selection_key)[0]
            decoy = NM.get(key)
            if decoy is None:
                raise SystemExit("REFUSING: key %r has no decoy; §P3 declares "
                                 "unmatched keys 0 in both strata." % (key,))
            rows.append((best, decoy))
        out[s] = rows
        print("§P3      %s  %d matched keys selected (one REAL + one decoy each)"
              % (s, len(rows)), flush=True)
    return out


#: [4413]'s STANDING RULE, per shape and never per stash. The disposition
#: lives HERE, in the actor, because this is the function that decides WHICH
#: items to force; the gate stays the guard and now refuses on the PREMISE
#: (an unrecognised key shape) rather than on the symptom ([4418](a)).
DECLARED_SHAPES = {
    #: the pre-patch shape. [4210]'s premise HOLDS: the key cannot express
    #: thinking state, so a thinking-on entry and a thinking-off call collide
    #: and forcing is the only disposition needing no argument about dating.
    ("max_tokens", "model", "prompt", "schema", "system_prompt",
     "temperature"): "FORCE",
    #: the post-f726aea shape. [4210]'s premise DOES NOT REACH IT: a pre-patch
    #: entry cannot occupy a post-patch key, so a stale answer is unreachable
    #: BY CONSTRUCTION and forcing buys nothing.
    ("max_tokens", "model", "prompt", "schema", "system_prompt",
     "temperature", "thinking"): "WARM",
}


def forced_set(items, model):
    """§P4/[4210] as AMENDED by [4413]: force PER KEY SHAPE, never per stash.

    [4210] forced every pre-existing entry because the key could not express
    thinking state — a thinking-on entry and a thinking-off call produced the
    SAME KEY and were indistinguishable. **That premise expired in July.**
    The stash now holds two shapes ([4351]: 16,561 without `thinking`, 8,885
    with), and where the key carries thinking a pre-patch entry CANNOT occupy
    a post-patch key. Forcing there buys nothing.

    **THE GATE DID NOT BECOME WRONG — ITS PREMISE DID, and nothing in the
    gate could notice** ([4414]). It refused nothing on either run of
    2026-08-04 only because `stash_key_fields()` sampled one key of two
    shapes: a defect and an expired premise cancelling each other out.

    **RUN 3's NON-THINKING SHAPE IS UNNECESSARY-NOT-EXEMPTED.** [4355]
    measured it: gpt-4o-mini's entries were overwritten IN PLACE, 11 keys
    carry two versions, and the newest of every straddling key is run 1's —
    verified at two seats. So its warm service rests on MEASUREMENT first and
    [4348] second, and [4348] is **SUPERSEDED BY THE STANDING RULE, NOT SPENT
    BY THIS RUN** ([4418](b)): a rule grants by construction and is
    checkable; a spent-marker is bookkeeping a future run could inherit
    silently. Recorded in the provenance block, asserted at the START.
    """
    from malign_logits.tasks.code_displacement_relation import DisplacementRelationTask
    t = DisplacementRelationTask()
    t.model = model
    have, seen, parsed, warm_shape = set(), 0, 0, 0
    for k in t.stash.keys():
        if not (isinstance(k, dict) and k.get("model") == model):
            continue
        #: the per-shape rule. A shape reaching here that DECLARED_SHAPES does
        #: not name is impossible — gate_stash_shapes() refuses on it before
        #: any call is made — so this is an assertion, not a branch.
        if DECLARED_SHAPES.get(tuple(sorted(k.keys()))) == "WARM":
            warm_shape += 1
            continue
        seen += 1
        pr = re.search(r'PROMPT: "(.*?)"\s*\n', k["prompt"], re.S)
        ab = re.search(r"A = (\S+)\s+B = (\S+)", k["prompt"])
        if pr and ab:
            parsed += 1
            have.add((re.sub(r"\s*_+\s*$", "", pr.group(1)).strip(),
                      ab.group(1), ab.group(2)))
    #: A4, [4232]: [4210] ordered a refusal if any CACHE-ELIGIBLE item is not
    #: in the forced set. An entry this regex fails to parse is exactly such
    #: an item — it would be served warm from an undatable entry — and the
    #: first version dropped it silently. The guard is on the DENOMINATOR:
    #: entries seen against entries parsed.
    if seen != parsed:
        raise SystemExit(
            "REFUSING: %d of %d stash entries for %s did not parse, so %d "
            "cache-eligible items cannot be placed in the forced set and "
            "would be served warm from entries nobody can date. [4210]'s "
            "completeness guard." % (seen - parsed, seen, model, seen - parsed))
    if warm_shape:
        print("  per-shape [4413]  %-28s %d entries on the WARM shape "
              "(thinking in key) excluded from forcing; %d on the FORCE shape"
              % (model, warm_shape, seen), flush=True)
    return {it for it in items if it in have}, seen


def stash_key_fields():
    """[4210]: the key's fields, so its blindness is on the record.

    **[4410]: THIS RETURNED THE FIRST KEY IT MET AND STOPPED — a sample of
    size one over a stash that demonstrably holds MORE THAN ONE KEY SHAPE.**
    [4351] measured two: 16,561 keys without a `thinking` field and 8,885
    with it, 35% of the stash. The gate below asks whether `thinking` is in
    the key; under the old sampler the answer depended on which key
    `.keys()` happened to yield first, **so whether this producer refused was
    a property of iteration order and not of the data.** It passed in the
    safe direction by accident.

    Returns the UNION over every dict key, so a field present anywhere is
    reported, and the per-shape census beside it so a reader sees the
    heterogeneity rather than a single tidy list.
    """
    from malign_logits.tasks.code_displacement_relation import DisplacementRelationTask
    shapes = collections.Counter()
    union = set()
    for k in DisplacementRelationTask().stash.keys():
        if isinstance(k, dict):
            shapes[tuple(sorted(k.keys()))] += 1
            union |= set(k.keys())
    return sorted(union), shapes


# ══════════════════════════════════════════════════════════════════════════
# §P3 — the test
# ══════════════════════════════════════════════════════════════════════════
def mcnemar_one_sided(b, c):
    """Exact binomial on the discordant pairs, one-sided (REAL > decoy).

    `b` = keys where REAL carries the field and its decoy does not;
    `c` = the reverse. Concordant pairs carry no information and are excluded
    by construction, which is what makes this the paired test.
    """
    n = b + c
    if n == 0:
        return None, n
    p = sum(math.comb(n, i) for i in range(b, n + 1)) / (2 ** n)
    return min(1.0, p), n


def cell(n_confirm):
    """§P3.1's four cells. Named before any coding existed; not a choice here."""
    return {3: "CONFIRMED", 2: "SPLIT", 1: "SINGLE-CODER", 0: "NONE"}[n_confirm]


def gate_pinned_blobs():
    """[4283]: the COMMIT is not the CODE, and `hasattr` is not the bytes.

    Free — two git calls, no provider — so it runs on every invocation
    INCLUDING `--dry-run`, which is where a gate that cannot be exercised
    stops being a gate.
    """
    import largeliterarymodels, inspect, subprocess
    path = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(largeliterarymodels))))
    #: **[4283]: `hasattr` CATCHES REMOVAL AND NOT EDITING.** A
    #: `deepseek_thinking_default` altered to leave thinking ON keeps the
    #: attribute, keeps `response_model` arriving, and keeps HEAD at the pin —
    #: three passes on the exact disposition P exists to control. HEAD says a
    #: tree DIFFERS from a commit and cannot say WHERE, so the commit check
    #: passes identically whether the dirt is in `tasks/` or in `providers.py`.
    #: **THE BLOB IS THE ANSWER, and a check performed by hand is not a check
    #: installed.** `llm.py` is included because `UsageTracker` and
    #: `no_reasoning_observed()` live there and an edit fails the same silent
    #: way.
    for _f in ("largeliterarymodels/providers.py", "largeliterarymodels/llm.py"):
        here = subprocess.run(["git", "-C", path, "hash-object", _f],
                              capture_output=True, text=True).stdout.strip()
        pinned = subprocess.run(
            ["git", "-C", path, "rev-parse", "%s:%s" % (EXPECTED_HEAD, _f)],
            capture_output=True, text=True).stdout.strip()
        if not here or not pinned:
            raise SystemExit("REFUSING: could not resolve %s at %s in %s"
                             % (_f, EXPECTED_HEAD, path))
        if here != pinned:
            #: **EACH FILE GETS ITS OWN FAILURE MODE.** One template handed a
            #: reader tripping `llm.py` the reason belonging to `providers.py`
            #: — and the missing sentence was the argument for including
            #: `llm.py` at all ([4289]).
            why = ("an edited `deepseek_thinking_default` keeps its name, "
                   "keeps the field arriving, and leaves thinking ON"
                   if _f.endswith("providers.py") else
                   "an edited `no_reasoning_observed()` keeps returning True "
                   "while measuring nothing")
            raise SystemExit(
                "REFUSING: %s is blob %s on disk and %s at %s. The commit "
                "matches and THE CODE DOES NOT — %s."
                % (_f, here[:12], pinned[:12], EXPECTED_HEAD, why))
        print("gate     blob pinned  %-38s %s" % (_f, here[:12]), flush=True)
    return path


def gate_response_model():
    """[4228]: OBSERVE the field, never assume the checkout.

    A patch on a branch is landed only for the process that imports that
    branch. One live probe — the cheapest possible call — and the producer
    refuses by NAMING THE CHECKOUT it imported rather than the branch it
    hoped for.
    """
    import largeliterarymodels, inspect, subprocess
    path = os.path.dirname(inspect.getfile(largeliterarymodels))
    head = subprocess.run(["git", "-C", path, "log", "-1", "--format=%h %s"],
                          capture_output=True, text=True).stdout.strip()
    from malign_logits.tasks.code_displacement_relation import (
        DisplacementRelationTask, prepare)
    t = DisplacementRelationTask()
    t.model = ROSTER[0]
    piu = {}
    t.map([prepare("The probe asked only whether the field arrives", "a", "b")],
          num_workers=1, force=True, per_item_usage=piu, errors={})
    got = piu.get(0, {})
    #: [4240], the fix [4239] promised: the gate takes the EXPECTED head as a
    #: PIN. Observation answers "is the capability here"; only a comparison
    #: answers "is this the checkout I was promised", and `0c0a983` against
    #: `b3e0915` is exactly the gap between them. A hash that arrives in a
    #: message is the kind of value that needs comparing, not trusting.
    if not head.startswith(EXPECTED_HEAD):
        raise SystemExit(
            "REFUSING: the imported checkout is at\n    %s\nand this producer "
            "pins %s. `b3e0915` was ORPHANED by a reset-and-rebuild and exists "
            "on no branch; a run pinning it would name a commit that ceases to "
            "exist." % (head or "(no git head)", EXPECTED_HEAD))
    #: THE TAG AGREES WITH THE HASH, OR THIS REFUSES. `^{commit}` dereferences
    #: the annotated tag; without it the tag OBJECT is returned and every run
    #: would refuse against a pin that never moved.
    _tagged = subprocess.run(
        ["git", "-C", os.path.dirname(os.path.dirname(_pr.__file__)),
         "rev-parse", "%s^{commit}" % EXPECTED_TAG],
        capture_output=True, text=True)
    _tc = (_tagged.stdout or "").strip()
    if _tagged.returncode != 0 or not _tc:
        raise SystemExit(
            "REFUSING: tag %s does not resolve. It is the only guarantee that "
            "%s survives a history rewrite, and this repository has orphaned a "
            "commit before." % (EXPECTED_TAG, EXPECTED_HEAD))
    if not _tc.startswith(EXPECTED_HEAD):
        raise SystemExit(
            "REFUSING: tag %s now points at\n    %s\nand this producer pins "
            "%s. **THE TAG MOVED.** The hash is the authority and it has not "
            "changed, so this is a loud failure rather than a silent re-pin — "
            "which is the whole reason the tag is an ASSERTION and not the "
            "pin." % (EXPECTED_TAG, _tc, EXPECTED_HEAD))
    print("gate     tag %s -> %s  (reachability assured; hash remains the "
          "authority)" % (EXPECTED_TAG, _tc[:12]), flush=True)
    #: HAZARD 1, [4240]: the patch is NOT ON MAIN. An interpreter resolving
    #: largeliterarymodels from main, a wheel or a non-editable install gets
    #: thinking ON and temperature accepted-and-discarded — the exact
    #: uncontrolled sampling P exists to avoid, with no error to catch it.
    import largeliterarymodels.providers as _pr
    if not hasattr(_pr, "deepseek_thinking_default"):
        raise SystemExit(
            "REFUSING: %s is UNPATCHED. Thinking would be ON and `temperature` "
            "accepted-and-silently-discarded. Nothing in the artifact would "
            "contradict it." % _pr.__file__)
    print("gate     providers PATCHED: %s" % _pr.__file__, flush=True)
    if "response_model" not in got:
        raise SystemExit(
            "REFUSING: per-item usage carries no `response_model`. This "
            "process imported largeliterarymodels from\n    %s\n    at %s\n"
            "and [4228]'s patch is not in it. A branch name is not a "
            "capability." % (path, head or "(no git head)"))
    print("gate     response_model OBSERVED: %r  (checkout %s)"
          % (got["response_model"], head), flush=True)
    return got["response_model"], path, head


def main():
    gate_registration()
    gate_pinned_blobs()
    inst, fsha = gate_instrument()
    d, psha = gate_population()
    known_answers(d)

    fields, shapes = stash_key_fields()
    print("\ngate     stash key fields (UNION over every key): %s"
          % ", ".join(fields), flush=True)
    #: [4410]: the census, not a sample. The heterogeneity IS the finding.
    for shape, n in shapes.most_common():
        print("         %6d keys : %s" % (n, ", ".join(shape)), flush=True)
    if len(shapes) > 1:
        print("         **%d DISTINCT KEY SHAPES.** The old sampler returned the "
              "first key it met, so this gate's verdict was a property of "
              "iteration order." % len(shapes), flush=True)
    #: **[4418](a): THE GATE GUARDS THE PREMISE, NOT THE SYMPTOM.** An
    #: earlier cut refused whenever `thinking` appeared in the key union —
    #: which is now the NORMAL state and which [4413]'s per-shape rule
    #: handles in `forced_set()`. Demoting this to a printer was not
    #: available either: "a gate that reports is not a gate that guards".
    #:
    #: So it refuses on the thing that would silently INVALIDATE the rule —
    #: **a key shape the disposition does not name.** A new field, a new
    #: provider, a library widening: any of those makes the two-shape
    #: analysis stale, and none of them would announce itself.
    undeclared = [sh for sh in shapes if sh not in DECLARED_SHAPES]
    if undeclared:
        raise SystemExit(
            "REFUSING: the stash holds %d key shape(s) the registered "
            "disposition does not name:\n%s\n"
            "[4413]'s per-shape rule was derived for exactly two shapes — the "
            "6-field pre-patch key (FORCE, [4210]'s premise holds) and the "
            "6+thinking post-patch key (WARM, a pre-patch entry cannot occupy "
            "it). A THIRD SHAPE MAKES THAT ANALYSIS STALE and nothing in the "
            "rule could notice. Re-derive the disposition before running."
            % (len(undeclared),
               "\n".join("    %6d keys : %s" % (shapes[sh], ", ".join(sh))
                         for sh in undeclared)))
    for sh, n in shapes.most_common():
        print("         %-5s %6d keys : %s"
              % (DECLARED_SHAPES[sh], n, ", ".join(sh)), flush=True)
    print("         all shapes declared; [4413]'s per-shape rule applies in "
          "forced_set()", flush=True)

    sel = select_pairs(d)

    #: **THE BATTERY IS THE WHOLE POPULATION, NOT THE SELECTION.** §P3's
    #: selection decides which pairs carry the PRIMARIES; it does not decide
    #: what gets coded. §P4's descriptive layer is the bulk product and names
    #: NARR — 3,713 items, 85% of the set — as its main exhibit, and §P4's own
    #: clause requires the 88 discarded risers to APPEAR. A run coding only the
    #: 581 selected-plus-decoy-plus-discarded items would cut the descriptive
    #: layer to 13% of what the frozen text describes.
    #:
    #: Caught by this producer's own dry run: 581 is an impossible number
    #: against the frozen text's "4,443 items x 3 coders".
    coded = [(r.slot if isinstance(r.slot, str) else "UNASSIGNED",
              r.item_class, r) for r in d.itertuples()]
    picked = {(real.prompt, real.a, real.b) for s in sel for real, _dc in sel[s]}
    roles = {}
    for _s, cls, r in coded:
        k = (r.prompt, r.a, r.b)
        roles[k] = ("PRIMARY-REAL" if k in picked else
                    "DECOY" if cls == "NEAR-MISS" else
                    "EXHIBIT" if cls == "EXHIBIT" else
                    "DISCARDED" if r.slot in ("ACT", "REF") else "DESCRIPTIVE")
    print("§P4      %d items to code — the WHOLE population, %s"
          % (len(coded), dict(collections.Counter(roles.values()))), flush=True)
    if len(coded) != len(d):
        raise SystemExit("REFUSING: the battery is the population; %d != %d"
                         % (len(coded), len(d)))

    keys = [(re.sub(r"\s*_+\s*$", "", str(x.prompt)).strip(), str(x.a), str(x.b))
            for _s, _role, x in coded]
    if "--dry-run" in sys.argv:
        print("\n--dry-run: gates, known answers, selection. NO CALL MADE.")
        for m in ROSTER:
            f, _seen = forced_set(set(keys), m)
            print("  forced (cache-cold) for %-28s %3d items" % (m, len(f)),
                  flush=True)
        return 0

    gate_response_model()

    # ── THE CALL LOOP ──────────────────────────────────────────────────────
    from malign_logits.tasks.code_displacement_relation import (
        DisplacementRelationTask, prepare)
    prompts = [prepare(str(x.prompt), str(x.a), str(x.b)) for _s, _r, x in coded]
    coders, per_coder_usage = {}, {}
    for model in ROSTER:
        forced, n_seen = forced_set(set(keys), model)
        idx_forced = [i for i, k in enumerate(keys) if k in forced]
        idx_warm = [i for i, k in enumerate(keys) if k not in forced]
        print("\n=== CODER %s" % model, flush=True)
        print("  forced (cache-cold) %d items — every item with a pre-existing "
              "stash entry, [4210]" % len(idx_forced), flush=True)
        for i in idx_forced:
            print("      %s | %s -> %s" % (keys[i][0][:52], keys[i][1], keys[i][2]),
                  flush=True)
        t = DisplacementRelationTask()
        t.model = model
        #: **THE SECOND CALL SITE, AND THE ONE [4299] MISSED.** `gate_instrument`
        #: was corrected to compare the quantity §P2 names; THIS re-check was
        #: left calling the method, so the top gate passed and the loop
        #: refused. **The fix for a diverged definition is a SWEEP OF THE
        #: METHOD, not a repair of the line the traceback named** — a grep for
        #: `instrument_sha256()` finds both; a traceback finds one.
        _here = hashlib.sha256(t.instrument_text().encode("utf-8")).hexdigest()
        if _here != INSTRUMENT_DIGEST:
            raise SystemExit(
                "REFUSING: sha256(instrument_text()) is %s under coder %s, "
                "§P2 pins %s. The RENDERED INSTRUMENT moved mid-run."
                % (_here[:16], model, INSTRUMENT_DIGEST[:16]))
        anns = [None] * len(prompts)
        piu, errs = {}, {}
        #: [4240], the peer seat's belt-and-braces, verbatim: assert
        #: `no_reasoning_observed()` AND empty `dropped_params` on the FIRST
        #: item, before spending the other 13,328. PER FAMILY — sonnet's
        #: dropped `temperature` is EXPECTED for its arm (it REJECTS, a 400,
        #: reported); deepseek's arm is where the drop must be empty, because
        #: deepseek ACCEPTS-AND-IGNORES and a drop there is invisible.
        t.usage.reset()
        first = t.map([prompts[0]], num_workers=1, force=True,
                      per_item_usage={}, errors={})
        r0 = t.usage.report()
        drops = dict(r0.get("dropped_params") or {})
        print("  first-item assert  calls=%d  reasoning_tokens=%d  dropped=%s"
              % (r0["calls"], r0["reasoning_tokens"], drops or "{}"), flush=True)
        #: [4245].2 — THE NO-REASONING GATE IS DEEPSEEK-ONLY. On Anthropic
        #: `no_reasoning_observed()` returns True UNCONDITIONALLY because no
        #: reasoning split is reported — a reviewer reproduced a false clean
        #: bill on 1,124 mostly-thinking tokens. Applying it there would be a
        #: gate that cannot fail, which is [4143]'s shape in a receipt.
        if model.startswith("deepseek/"):
            if not t.usage.no_reasoning_observed():
                raise SystemExit(
                    "REFUSING after ONE item: %s reported reasoning tokens "
                    "(%d). Thinking is on; the other 13,328 calls would be "
                    "uncontrolled sampling with nothing in the artifact to "
                    "contradict them." % (model, r0["reasoning_tokens"]))
        else:
            print("  [scoped out] no_reasoning gate NOT applied to %s — it "
                  "returns True unconditionally on this provider and would be "
                  "a gate that cannot fail ([4245].2). This arm's receipt is "
                  "its rejection report + the fingerprint." % model, flush=True)
        #: [4245].1 — THE DEEPSEEK dropped_params ASSERT IS HOLLOW AND IS
        #: MARKED NON-ATTESTING RATHER THAN KEPT AS A GATE. DeepSeek's
        #: temperature drop is never RECORDED (it is accepted and ignored at
        #: the API, which is the whole finding), so an empty `dropped_params`
        #: on that arm is guaranteed by construction and attests nothing. It
        #: prints as a diagnostic and refuses on nothing.
        if model.startswith("deepseek/"):
            print("  [non-attesting] deepseek dropped_params=%s — EMPTY BY "
                  "CONSTRUCTION, not evidence: the drop is never recorded "
                  "because it is never reported ([4245].1)" % (drops or "{}"),
                  flush=True)
        t.usage.reset()
        for label, idxs, force in (("forced", idx_forced, True),
                                   ("battery", idx_warm, False)):
            if not idxs:
                continue
            sub = [prompts[i] for i in idxs]
            #: [4250].1/.3 — `warm_cache=False` plus OUR OWN serial warm, and
            #: `fail_fast=False` doubly ordered: breaker signatures truncate at
            #: 120 chars, so a wide schema collapses distinct failures into one
            #: signature and makes a spurious abort likelier for us than for
            #: anyone. One serial call warms the prefix; the rest run parallel.
            kw = dict(num_workers=WORKERS, force=force, verbose=True,
                      errors=errs, per_item_usage=piu, fail_fast=False)
            try:
                got = t.map(sub, warm_cache=False, **kw)
            except TypeError:
                #: an older framework without the kwarg — named, not swallowed
                print("  [note] warm_cache kwarg absent in this checkout; "
                      "framework default in effect", flush=True)
                got = t.map(sub, **kw)
            for j, i in enumerate(idxs):
                anns[i] = got[j]
                if j in piu:
                    per_coder_usage.setdefault(model, {})[i] = piu.pop(j)
            #: [4250].2 — TAIL ACCOUNTING COUNTS NONES IN RESULTS, NEVER
            #: ENTRIES IN `errors`. A silent drop leaves a None and no error
            #: entry; counting the error dict reports zero failures on a run
            #: that lost items.
            n_none = sum(1 for x in got if x is None)
            print("  %-8s %d items  none=%d  %s" % (
                label, len(idxs), n_none, t.usage.summary_line()), flush=True)
        #: **[4255].1 — RETRY BEFORE ANY REFUSAL.** A failed call leaves a
        #: None and NO stash entry, so retrying Nones is fresh-call cheap and
        #: forced-set neutral. Without it a transient transport tail converts
        #: a complete-able run into an INCOMPLETE artifact for nothing.
        #: ONE pass, and its outcome is reported whether or not it helps.
        missing = [i for i, x in enumerate(anns) if x is None]
        if missing:
            print("  retry pass  %d None(s) — one pass, fresh calls, no stash "
                  "entries to force" % len(missing), flush=True)
            r_errs, r_piu = {}, {}
            kw = dict(num_workers=WORKERS, force=True, verbose=True,
                      errors=r_errs, per_item_usage=r_piu, fail_fast=False)
            try:
                again = t.map([prompts[i] for i in missing],
                              warm_cache=False, **kw)
            except TypeError:
                again = t.map([prompts[i] for i in missing], **kw)
            recovered = 0
            for j, i in enumerate(missing):
                if again[j] is not None:
                    anns[i] = again[j]
                    recovered += 1
                if j in r_piu:
                    per_coder_usage.setdefault(model, {})[i] = r_piu[j]
            print("  retry pass  recovered %d of %d; %d still missing"
                  % (recovered, len(missing), len(missing) - recovered),
                  flush=True)
        coders[model] = anns

        # ── §P2's identity, and §P2.1's receipt ────────────────────────────
        rep = t.usage.report()
        served = rep.get("response_models") or {}
        coders.setdefault("_meta", {})[model] = {
            "coder_asked": model,
            #: [4228], the peer seat's own phrasing: THE ID THE SERVER
            #: REPORTED HAVING SERVED. A self-report — strictly better than
            #: the asked name, still the provider characterising itself, and
            #: it cannot detect one checkpoint served under another's name.
            "coder_server_reported": dict(served),
            "_identity_observability": (
                "MEASURED — per-item response_model observed" if served else
                "NOT OBSERVED — no response_model reached this run; the id "
                "below is what was ASKED and nothing recorded what answered"),
            "dropped_params": rep.get("dropped_params", {}),
            #: §P2.1's receipt in the framework's own semantics. NOT
            #: "field present and equals zero" — with thinking off the
            #: reasoning fields are ABSENT, so that form can never pass.
            "no_reasoning_observed": t.usage.no_reasoning_observed(),
            "unreported_calls": t.usage.unreported_calls,
            "coverage": {"calls": rep["calls"], "items": len(prompts),
                         "cache_hit_rate": rep["cache_hit_rate"],
                         "forced": len(idx_forced),
                         #: [4250].2 — Nones in results, NEVER len(errors).
                         "missing_annotations": sum(1 for x in anns if x is None),
                         "_errors_note": ("`errors` is not a failure count — a "
                                          "silent drop leaves a None and no "
                                          "entry. See missing_annotations.")},
            "usage": rep,
            #: [4250].4 — `call_anthropic` has ZERO tests: the cache wiring,
            #: the dropped-temperature record and the whole usage mapping are
            #: untested upstream. The number is REPORTED and never cited as
            #: verified, and this arm's receipt stays what [4243] scoped —
            #: the rejection report plus the fingerprint.
            "_cache_hit_rate_status": (
                "REPORTED, NOT VERIFIED — call_anthropic's usage mapping is "
                "untested upstream; do not cite this rate as a measurement"
                if model.startswith("anthropic/") else "reported"),
        }
        m = coders["_meta"][model]
        print("  receipt  no_reasoning_observed=%s  calls=%d  items=%d  "
              "cache_hit=%.1f%%  forced=%d  errors=%d  served=%s" % (
                  m["no_reasoning_observed"], rep["calls"], len(prompts),
                  100 * rep["cache_hit_rate"], len(idx_forced), len(errs),
                  dict(served) or "NONE"),
              flush=True)

    return emit(d, sel, coded, keys, coders, roles,
                {"registration": REGISTRATION_SHA, "instrument": inst,
                 "source": fsha, "population": psha})


if __name__ == "__main__":
    sys.exit(main())


# ══════════════════════════════════════════════════════════════════════════
# §P4.1 — the agreement statistic, five clauses
# ══════════════════════════════════════════════════════════════════════════
def krippendorff(units, ordinal=False, levels=None):
    """Krippendorff's alpha. `units` = [[label, ...], ...], one list per item.

    Implemented here because the package is not installed and a chance-
    corrected statistic is §P4.1 clause 3's requirement, not an option. Nominal
    delta by default; ordinal delta over `levels` when asked.
    """
    vals = [[v for v in u if v is not None] for u in units]
    vals = [u for u in vals if len(u) >= 2]
    if not vals:
        return None
    cats = sorted({v for u in vals for v in u})
    if ordinal:
        if levels is None:
            return None
        rank = {c: i for i, c in enumerate(levels)}
        if any(c not in rank for u in vals for c in u):
            return None
    n_total = sum(len(u) for u in vals)

    #: A3, [4232]: the ORDINAL delta is marginal-based, not rank-based.
    #: (rank[a]-rank[b])**2 is the INTERVAL metric — a different statistic
    #: that happens to land within 0.007 of this one on these marginals.
    #: Closeness is not identity and the frozen text names ordinal.
    marg = collections.Counter(v for u in vals for v in u)

    def delta(a, b):
        if not ordinal:
            return 0.0 if a == b else 1.0
        lo, hi = sorted((rank[a], rank[b]))
        run = sum(marg.get(levels[g], 0) for g in range(lo, hi + 1))
        return (run - (marg.get(a, 0) + marg.get(b, 0)) / 2.0) ** 2

    do = 0.0
    for u in vals:
        m = len(u)
        for i in range(m):
            for j in range(m):
                if i != j:
                    do += delta(u[i], u[j]) / (m - 1)
    do /= n_total
    flat = [v for u in vals for v in u]
    de = 0.0
    for a in cats:
        for b in cats:
            if a != b:
                de += flat.count(a) * flat.count(b) * delta(a, b)
    de /= (n_total * (n_total - 1)) if n_total > 1 else 1
    return 1.0 - do / de if de else None


def percent_agreement(units):
    """Raw agreement. Reported ONLY beside alpha — clause 3 forbids it alone."""
    hits = tot = 0
    for u in units:
        u = [v for v in u if v is not None]
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                tot += 1
                hits += (u[i] == u[j])
    return (hits / tot) if tot else None


ORDINAL_LEVELS = ["B_MILDER", "SAME_PITCH", "B_STRONGER"]


def agreement_block(rows, ordinal=False):
    """One field's agreement: alpha + percent + marginals + PAIRWISE.

    Clause 4: a three-way alpha averages away the asymmetry between a pinned
    pair and a pair involving an unpinned coder. The pairwise table is where
    that becomes visible, so it is not optional.
    """
    models = list(rows[0].keys()) if rows else []
    units = [[r.get(m) for m in models] for r in rows]
    if ordinal:
        units = [[v for v in u if v in ORDINAL_LEVELS] for u in units]
    out = {"n_items": len(units),
           "alpha": krippendorff(units, ordinal, ORDINAL_LEVELS if ordinal else None),
           "percent_agreement": percent_agreement(units),
           "marginals": {m: dict(collections.Counter(
               r.get(m) for r in rows if r.get(m) is not None)) for m in models},
           "pairwise": {}}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            pu = [[r.get(a), r.get(b)] for r in rows]
            if ordinal:
                pu = [[v for v in u if v in ORDINAL_LEVELS] for u in pu]
            out["pairwise"]["%s|%s" % (a, b)] = {
                "alpha": krippendorff(pu, ordinal,
                                      ORDINAL_LEVELS if ordinal else None),
                "percent_agreement": percent_agreement(pu)}
    return out


# ══════════════════════════════════════════════════════════════════════════
# §P6 — the verdict sentences, QUOTED from the frozen text
# ══════════════════════════════════════════════════════════════════════════
def verdict_sentence(stratum, n_confirm, dissenter=None):
    """§P6.1/§P6.2's four sentences. Composed nowhere — selected here.

    A verdict that needs a caveat appended from elsewhere will be quoted
    without it, so these are whole sentences and the producer picks one.
    """
    where = "In ACT slots" if stratum == "ACT" else "In REFERENT slots"
    read = ("read the risen word as an exclamation" if stratum == "ACT"
            else "read the relation as metonymy")
    if n_confirm == 3:
        return ("%s, all three coder families independently %s more often than "
                "the stationary control drawn from the same faller. CONFIRMED "
                "under LLM coding; this is agreement among three model families "
                "and not human validation." % (where, read))
    if n_confirm == 2:
        return ("%s, two of three coder families %s more often than its control "
                "and one did not. NOT SUPPORTED, reported as a SPLIT; the "
                "dissenting family is %s. A two-of-three split is not a "
                "confirmation and is not reported as one."
                % (where, read, dissenter or "<NAME>"))
    if n_confirm == 1:
        return ("%s, one of three coder families showed the effect. NOT "
                "SUPPORTED, single-coder." % where)
    tail = ("" if stratum == "ACT" else
            " METONYMY is untaught in the few-shot (§P7), so it had no example "
            "to prime it and its absence is correspondingly weaker evidence.")
    return ("%s, no coder family %s more often than its control. NOT SUPPORTED. "
            "This is not evidence that the relation is absent — absence of a "
            "rate difference under LLM coding is not absence of the relation.%s"
            % (where, read, tail))


def atomic_dump(payload, path):
    """[4357].2 / [4360].4 — the canonical path never holds a partial file.

    `open(path, "w")` TRUNCATES before `json.dump` serialises, so a dump that
    raises leaves a truncated artifact where readers look — and where this
    producer's OWN escrow branch looks: `if os.path.exists(OUT)` would copy
    the corpse to a `PREFIX-<hash>` sibling and chmod it 444, **canonizing an
    invalid file as a legitimate prior version.** Run 2 left 11,342,223 bytes
    of unparseable JSON exactly this way.

    Dump to a temp path in the SAME directory (so `os.replace` stays on one
    filesystem and is atomic), fsync, then rename. The canonical path holds
    either the previous artifact or a complete new one, and never a partial.
    """
    d = os.path.dirname(path) or "."
    tmp = os.path.join(d, ".%s.partial-%d" % (os.path.basename(path), os.getpid()))
    try:
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        #: a failed serialise leaves the temp file, never the destination
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def emit(d, sel, coded, keys, coders, roles, pins):
    """§P4's agreement FIRST, then §P3's primaries, then §P6's sentences.

    Clause 5's ordering is load-bearing and is enforced by the order of this
    function: if coders disagree on the field a primary tests, that primary is
    measuring noise and the reader must see it BEFORE the outcome.
    """
    meta = coders.pop("_meta", {})
    idx = {k: i for i, k in enumerate(keys)}

    def field(model, key, name):
        a = coders[model][idx[key]] if key in idx else None
        if a is None:
            return None
        v = a.model_dump() if hasattr(a, "model_dump") else a
        if name == "relations":
            r = v.get("relations") or []
            return list(r)
        return v.get(name)

    def k3(x):
        return (re.sub(r"\s*_+\s*$", "", str(x.prompt)).strip(), str(x.a), str(x.b))

    # ── §P4.1 clause 5: the primaries' own fields, FIRST ───────────────────
    print("\n=== §P4.1 AGREEMENT — REPORTED BEFORE THE VERDICTS IT QUALIFIES",
          flush=True)
    agree = {}
    for stratum, fname, ordinal in (("ACT", "speech_act", False),
                                    ("REF", "relations", False)):
        rows = []
        for real, decoy in sel[stratum]:
            for x in (real, decoy):
                r = {m: field(m, k3(x), fname) for m in ROSTER}
                if fname == "relations":
                    r = {m: (v[0] if v else None) for m, v in r.items()}
                rows.append(r)
        agree[stratum + "/" + fname] = agreement_block(rows, ordinal)
        b = agree[stratum + "/" + fname]
        print("  %-14s %-12s alpha %s   percent %s   n %d" % (
            stratum, fname,
            "%.4f" % b["alpha"] if b["alpha"] is not None else "n/a",
            "%.4f" % b["percent_agreement"] if b["percent_agreement"] is not None else "n/a",
            b["n_items"]), flush=True)
        for pair, pv in b["pairwise"].items():
            print("      %-56s alpha %s" % (
                pair, "%.4f" % pv["alpha"] if pv["alpha"] is not None else "n/a"),
                flush=True)

    # ── §P3 the primaries ──────────────────────────────────────────────────
    print("\n=== §P3 PRIMARIES — McNemar on discordant pairs, per coder", flush=True)
    prim = {}
    for stratum, test in (("ACT", "EXCLAMATION"), ("REF", "METONYMY")):
        prim[stratum] = {}
        for m in ROSTER:
            b = c = 0
            for real, decoy in sel[stratum]:
                if stratum == "ACT":
                    rv = field(m, k3(real), "speech_act") == test
                    dv = field(m, k3(decoy), "speech_act") == test
                else:
                    rv = test in (field(m, k3(real), "relations") or [])
                    dv = test in (field(m, k3(decoy), "relations") or [])
                b += (rv and not dv)
                c += (dv and not rv)
            p, n = mcnemar_one_sided(b, c)
            prim[stratum][m] = {"b_real_only": b, "c_decoy_only": c,
                                "n_discordant": n, "p_one_sided": p,
                                "confirms": bool(p is not None and p < ALPHA)}
            print("  %-6s %-28s b=%3d c=%3d  n_disc=%3d  p=%s  %s" % (
                stratum, m, b, c, n,
                "%.5f" % p if p is not None else "n/a",
                "CONFIRMS" if prim[stratum][m]["confirms"] else "no"), flush=True)

    # ── §P3.1's four cells, §P6's sentences ────────────────────────────────
    print("\n=== §P3.1 / §P6 READING", flush=True)
    reading = {}
    for stratum in ("ACT", "REF"):
        n_ok = sum(1 for m in ROSTER if prim[stratum][m]["confirms"])
        diss = [m for m in ROSTER if not prim[stratum][m]["confirms"]]
        reading[stratum] = {
            "n_confirm": n_ok, "cell": cell(n_ok),
            "dissenting": diss if n_ok == 2 else None,
            "sentence": verdict_sentence(stratum, n_ok,
                                         diss[0] if (n_ok == 2 and diss) else None)}
        print("\n  %s  %d/3  -> %s" % (stratum, n_ok, cell(n_ok)), flush=True)
        print("  %s" % reading[stratum]["sentence"], flush=True)

    # ── §P4 THE DESCRIPTIVE LAYER — the registration's "bulk product" ──────
    #: A1, [4232]: the first build made 13,329 paid judgments and wrote an
    #: artifact from which §P4 could not be reconstructed. The judgments are
    #: the bulk of the work and the artifact is the record; storing only the
    #: tests kept the cheap half.
    print("\n=== §P4 DESCRIPTIVE LAYER", flush=True)
    cells = []
    for n, (_s, cls, x) in enumerate(coded):
        key = k3(x)
        row = {"prompt": str(x.prompt), "a": str(x.a), "b": str(x.b),
               "item_class": cls, "slot": (x.slot if isinstance(x.slot, str)
                                           else None),
               "domain": (x.domain if isinstance(x.domain, str) else None),
               "role": roles.get(key), "n_edges": int(x.n_edges),
               "coders": {}}
        for m in ROSTER:
            a = coders[m][n]
            if a is None:
                row["coders"][m] = None
                continue
            v = a.model_dump() if hasattr(a, "model_dump") else dict(a)
            row["coders"][m] = {
                "relations": list(v.get("relations") or []),
                "intensity": v.get("intensity"),
                "speech_act": v.get("speech_act"),
                "a_is_content_word": v.get("a_is_content_word"),
                "b_is_content_word": v.get("b_is_content_word"),
                "reason": v.get("reason"), "slot_note": v.get("slot_note")}
        cells.append(row)

    def modal(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        c = collections.Counter(vals).most_common()
        top = [k for k, n in c if n == c[0][1]]
        #: A tie is REPORTED as a tie. A modal label invented by a tie-break
        #: is a synthetic label, which §P3.1 refuses for the primaries and
        #: §P4 has no reason to allow.
        return top[0] if len(top) == 1 else sorted(top)

    for r in cells:
        rel = [(r["coders"][m] or {}).get("relations") or [] for m in ROSTER]
        r["modal_relation"] = modal([x[0] if x else None for x in rel])
        r["modal_intensity"] = modal([(r["coders"][m] or {}).get("intensity")
                                      for m in ROSTER])
        r["modal_speech_act"] = modal([(r["coders"][m] or {}).get("speech_act")
                                       for m in ROSTER])
        r["unanimous_relation"] = (isinstance(r["modal_relation"], str) and
                                   len({x[0] if x else None for x in rel}) == 1)

    REAL = [r for r in cells if r["item_class"] == "REAL"]

    #: **[4360]: A TIE IS NOT A CATEGORY AND NEVER BECOMES A KEY.**
    #: `modal()` returns a LIST when the three coders tie, and the previous
    #: `mix()` tupled it into the same dict as the labels — so a dict mixing
    #: `str` and `tuple` keys reached `json.dump(sort_keys=True)` and raised
    #: AFTER every quantity was computed. **The serialization crash was the
    #: cheap half of the defect.** The expensive half: `b_milder_share`'s
    #: denominator was `sum(... if isinstance(k, str))`, so the tuples were
    #: silently dropped from it — an UNDECLARED population rule wearing a
    #: type check, excluding 630 of 4,443 items (14.2%) on intensity and 954
    #: (21.5%) on relation. **Stringifying the keys would have re-populated
    #: that denominator and moved two published rates ~14% relative, as a
    #: side effect of a serialization repair ([4359]).**
    #:
    #: [4360] rules: the rate keeps the denominator it already used — ITEMS
    #: BEARING A MODAL — **declared here rather than produced by accident**,
    #: and the ties are published as their own quantity.
    def is_tie(v):
        return isinstance(v, list)

    def mix(rows, f):
        """Counts over MODAL-BEARING rows ONLY. Never returns a tuple key."""
        return dict(collections.Counter(
            r[f] for r in rows if isinstance(r[f], str)))

    def tie_census(rows, f):
        """The complement `mix()` drops, [4360].2 — reported, never inferred.

        A tied item HAS no modal, so it cannot be an item whose modal is
        B_MILDER and does not belong in that rate's denominator. It is also
        not nothing: a three-way split across coder families is the sharpest
        disagreement this instrument can record. Tied sets are rendered with
        a ` | ` join so no reader can mistake one for a lexicon label.
        """
        tied = [r[f] for r in rows if is_tie(r[f])]
        return {
            "n_total": len(rows),
            "n_modal_bearing": sum(1 for r in rows if isinstance(r[f], str)),
            "n_tied": len(tied),
            "n_absent": sum(1 for r in rows if r[f] is None),
            "tied_sets": dict(collections.Counter(
                " | ".join(str(x) for x in t) for t in tied)),
        }

    desc = {
        "_unit": ("one row per item; a rate over these rows is an ITEM rate. "
                  "§P4: a descriptive rate comparing REAL to decoy STATES ITS "
                  "UNIT or does not compare — at item level one decoy serves "
                  "up to 11 REALs (1.59x average reuse)."),
        "relation_mix_by_slot": {
            s: mix([r for r in REAL if r["slot"] == s], "modal_relation")
            for s in sorted({r["slot"] for r in REAL if r["slot"]})},
        "intensity_mix_overall": mix(REAL, "modal_intensity"),
        "intensity_mix_by_domain": {
            dm: mix([r for r in REAL if r["domain"] == dm], "modal_intensity")
            for dm in sorted({r["domain"] for r in REAL if r["domain"]})},
        "speech_act_mix_by_slot": {
            s: mix([r for r in REAL if r["slot"] == s], "modal_speech_act")
            for s in sorted({r["slot"] for r in REAL if r["slot"]})},
        "narr_relation_taxonomy": mix([r for r in REAL if r["slot"] == "NARR"],
                                      "modal_relation"),
        "discarded_risers": [
            {"prompt": r["prompt"], "a": r["a"], "b": r["b"], "slot": r["slot"],
             "modal_relation": r["modal_relation"],
             "modal_intensity": r["modal_intensity"],
             "modal_speech_act": r["modal_speech_act"]}
            for r in cells if r["role"] == "DISCARDED"],
    }
    #: §P4's B_MILDER share, named because the registration names it.
    #: [4360].2 — the ties, as their own quantity, per field and per stratum.
    NARR_ROWS = [r for r in REAL if r["slot"] == "NARR"]
    desc["_tie_census"] = {
        "_what": ("A TIE IS THREE CODER FAMILIES SPLIT THREE WAYS, not a "
                  "compound label. `modal()` returns no modal for these "
                  "items, so they are EXCLUDED from every modal-based rate "
                  "below and counted here instead. Tied sets are joined with "
                  "' | ' so none can be read as a lexicon category."),
        "intensity_overall": tie_census(REAL, "modal_intensity"),
        "relation_overall": tie_census(REAL, "modal_relation"),
        "speech_act_overall": tie_census(REAL, "modal_speech_act"),
        "relation_narr": tie_census(NARR_ROWS, "modal_relation"),
        "relation_by_slot": {
            s: tie_census([r for r in REAL if r["slot"] == s], "modal_relation")
            for s in sorted({r["slot"] for r in REAL if r["slot"]})},
    }

    im = desc["intensity_mix_overall"]
    #: **THE DENOMINATOR IS DECLARED, NOT INHERITED FROM A TYPE CHECK.**
    #: `mix()` now yields modal-bearing rows only, so this sum IS the
    #: modal-bearing count — the same value the old `isinstance(k, str)`
    #: produced, arrived at by a stated rule instead of by an accident.
    tot = sum(im.values())
    desc["intensity_denominator"] = {
        "n_modal_bearing": tot,
        "n_total_REAL": len(REAL),
        "_rule": ("[4360].1: modal-based rates are over ITEMS BEARING A "
                  "MODAL. Ties and absent modals are not in this "
                  "denominator; see _tie_census for both counts."),
    }
    desc["b_milder_share"] = (im.get("B_MILDER", 0) / tot) if tot else None
    desc["not_comparable_rate"] = ((im.get("NOT_COMPARABLE", 0) / tot)
                                   if tot else None)
    print("  NARR taxonomy (main exhibit, %d items): %s" % (
        len(NARR_ROWS),
        dict(sorted(desc["narr_relation_taxonomy"].items(),
                    key=lambda kv: -kv[1])[:6])), flush=True)
    _tn = desc["_tie_census"]["relation_narr"]
    print("  NARR ties (3-way coder splits, NOT a category): %d of %d "
          "(%.1f%%); top tied sets: %s" % (
              _tn["n_tied"], _tn["n_total"],
              100 * _tn["n_tied"] / _tn["n_total"] if _tn["n_total"] else 0.0,
              dict(sorted(_tn["tied_sets"].items(),
                          key=lambda kv: -kv[1])[:3])), flush=True)
    #: [4250].2's form: a rate is quoted WITH the denominator it achieved.
    print("  B_MILDER share %s   NOT_COMPARABLE rate %s   — both OF %d "
          "MODAL-BEARING ITEMS of %d REAL (%d tied, %d absent). (§P4.1 clause "
          "2: NOT_COMPARABLE is off-scale and excluded from the ordinal "
          "metric)"
          % ("%.4f" % desc["b_milder_share"] if desc["b_milder_share"] is not None else "n/a",
             "%.4f" % desc["not_comparable_rate"] if desc["not_comparable_rate"] is not None else "n/a",
             tot, len(REAL), desc["_tie_census"]["intensity_overall"]["n_tied"],
             desc["_tie_census"]["intensity_overall"]["n_absent"]),
          flush=True)
    print("  discarded risers carried into §P4: %d (66 ACT + 22 REF)"
          % len(desc["discarded_risers"]), flush=True)

    #: A2, [4232]: intensity's own agreement, on the ORDINAL metric declared
    #: by §P4.1 clause 2, NOT_COMPARABLE excluded and its rate reported above.
    for stratum in ("ACT", "REF", "NARR"):
        rows = [{m: (r["coders"][m] or {}).get("intensity") for m in ROSTER}
                for r in REAL if r["slot"] == stratum]
        if rows:
            agree[stratum + "/intensity"] = agreement_block(rows, ordinal=True)
            b = agree[stratum + "/intensity"]
            print("  %-6s intensity   alpha %s (ordinal)   percent %s   n %d" % (
                stratum,
                "%.4f" % b["alpha"] if b["alpha"] is not None else "n/a",
                "%.4f" % b["percent_agreement"] if b["percent_agreement"] is not None else "n/a",
                b["n_items"]), flush=True)

    #: §P5 — characterization only, NO RATE. The exhibits' own reasons are
    #: the point: the instrument's account of the relation, in its words.
    exhibits = [r for r in cells if r["item_class"] == "EXHIBIT"]
    for r in exhibits:
        r["_may_not_enter_any_rate"] = True

    #: [4250]'s OWN guard, cheap and ours: 4,443 rows PER CODER in `cells`
    #: before anything is written. It catches a silent drop or duplicate no
    #: matter whose code path produced it — upstream's, the framework's or
    #: mine. THE COUNT CATCHES WHAT THE DISPLAY HIDES.
    if len(cells) != 4443:
        raise SystemExit("REFUSING to write: %d cells, §P1 publishes 4,443."
                         % len(cells))
    #: **[4252]: THE FIRST VERSION OF THIS GUARD COULD NOT FIRE.** It computed
    #: `n_missing` per coder, PRINTED it, and then refused on `len(cells)` — a
    #: global property already checked two lines above and invariant across
    #: coders by construction. lacan built the exact state it exists to catch
    #: (sonnet missing 443) and watched it print the defect and pass.
    #:
    #: **A GUARD THAT COMPUTES THE RIGHT QUANTITY AND CONDITIONS ON ANOTHER
    #: IS A GUARD THAT REPORTS RATHER THAN GUARDS**, and it reads as a check
    #: in every diff.
    #: **[4255].2 — THE VERDICT-WITHHOLDING CONDITION IS PINNED-ROWS-ONLY.**
    #: The first version withheld ALL verdicts if ANY row was missing, so one
    #: lost NARR annotation would kill §P3's complete, frozen-n primaries.
    #: That over-protects: §P3's denominator is threatened only by the rows
    #: §P3 USES. Two tiers, and the tier is decided by WHICH rows are short.
    PINNED_ROLES = {"PRIMARY-REAL", "DECOY", "EXHIBIT"}
    #: the paired decoys are the ones §P3 reads — a decoy on a key whose REAL
    #: was not selected is descriptive, not pinned.
    pinned_keys = set()
    for _s in ("ACT", "REF"):
        for _real, _dc in sel[_s]:
            pinned_keys.add(k3(_real))
            pinned_keys.add(k3(_dc))
    for r in cells:
        kk = (re.sub(r"\s*_+\s*$", "", str(r["prompt"])).strip(),
              str(r["a"]), str(r["b"]))
        r["_pinned"] = (kk in pinned_keys) or (r["item_class"] == "EXHIBIT")

    incomplete, gaps = {}, {}
    for _m in ROSTER:
        short = [r for r in cells if r["coders"].get(_m) is None]
        n_missing = len(short)
        pinned_short = [r for r in short if r["_pinned"]]
        print("  reconciliation  %-28s %d/4443 coded  (%d missing, %d of them "
              "PINNED)" % (_m, 4443 - n_missing, n_missing, len(pinned_short)),
              flush=True)
        if pinned_short:
            incomplete[_m] = len(pinned_short)
        if n_missing:
            gaps[_m] = {
                "missing_total": n_missing,
                "missing_pinned": len(pinned_short),
                "by_stratum": dict(collections.Counter(
                    (r["slot"] or "UNASSIGNED") for r in short)),
                "by_role": dict(collections.Counter(
                    r["role"] for r in short))}
    if gaps and not incomplete:
        print("  gaps are confined to DESCRIPTIVE / non-primary DECOY / "
              "DISCARDED rows — §P3's denominator is untouched, so the FULL "
              "artifact writes WITH verdicts and every §P4 rate carries its "
              "achieved denominator ([4255].2)", flush=True)
    #: **THE DISPOSITION IS DECLARED HERE BECAUSE THE FROZEN TEXT DOES NOT
    #: MAKE IT ([4252]).** A hard abort would destroy 13,329 paid judgments
    #: over one API failure; a silent pass is what the guard exists to stop.
    #: So: the EVIDENCE is written and the VERDICTS are withheld. An
    #: incomplete run yields a citable descriptive artifact and no primary.
    if incomplete:
        payload_incomplete = {
            "_what": "Registration P — INCOMPLETE RUN, evidence only.",
            "_status": "INCOMPLETE — NO VERDICT MAY BE READ FROM THIS FILE",
            "_incomplete_pinned": incomplete,
            "_gaps": gaps,
            "_disposition": (
                "Rows §P3 USES — the selected REALs, their paired decoys and "
                "the exhibits — are short for at least one coder, AFTER the "
                "retry pass. The cells are WRITTEN (13,329 paid judgments are "
                "not discarded over a transport failure) and the PRIMARIES "
                "ARE NOT COMPUTED, because §P3 is a paired count whose "
                "denominator a short PINNED row changes silently. A gap "
                "confined to descriptive rows does NOT reach here — it writes "
                "the full artifact with achieved denominators ([4255].2)."),
            "_pins": pins, "cells": cells,
        }
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        out_i = OUT.replace(".json", ".INCOMPLETE.json")
        #: **[4256]: THIS PATH LOCKED WITHOUT UNLOCKING.** It chmod 444'd its
        #: own output and had no escrow-and-unlock, so a SECOND incomplete run
        #: made every call, reached the save, and died on a permission bit —
        #: losing exactly the evidence this path exists to preserve, at the
        #: only moment it matters. The main path's three lines, which this
        #: path skipped:
        if os.path.exists(out_i):
            os.makedirs(ESCROW, exist_ok=True)
            prior = open(out_i, "rb").read()
            h = hashlib.sha256(prior).hexdigest()[:16]
            dst = os.path.join(ESCROW,
                               "result_p_primary.INCOMPLETE.PREFIX-%s.json" % h)
            if not os.path.exists(dst):
                open(dst, "wb").write(prior)
                os.chmod(dst, 0o444)
            os.chmod(out_i, 0o644)
            print("  escrowed the prior INCOMPLETE artifact @ %s" % h, flush=True)
        atomic_dump(payload_incomplete, out_i)
        os.chmod(out_i, 0o444)
        raise SystemExit(
            "REFUSING to emit verdicts: PINNED rows short after retry, %r. "
            "The evidence is written to %s and locked; the primaries are NOT "
            "computed. A gap confined to descriptive rows would not have "
            "reached this branch." % (incomplete, os.path.basename(out_i)))

    payload = {
        "_what": "Registration P — the displacement-relation annotation.",
        "cells": cells,
        "descriptives": desc,
        "exhibits": exhibits,
        "_exhibit_order": ("Exhibits ran in POPULATION ORDER, not shuffled. "
                           "§P5 says 'shuffled among the battery'; a shuffle "
                           "needs an RNG and this registration declares NO "
                           "SEED, so the absence is DECLARED rather than "
                           "silently supplied. Eleven contiguous items is a "
                           "mild form of the position effect §P2 forbids "
                           "batching for, and a reader should not assume a "
                           "randomisation that did not happen."),
        "_pins": pins,
        #: [4413]/[4418](b) — a RECORD, not a guard. The load-bearing check
        #: is `gate_stash_shapes()` at the START, because this producer has
        #: died on its last line TWICE (emit unreachable, then json.dump on
        #: mixed keys) and an exit-path assertion is exactly the code that
        #: does not run when it matters.
        "_forcing_disposition": (
            "[4413]'s STANDING PER-SHAPE RULE governs: key WITH `thinking` -> "
            "serve warm (a pre-patch entry cannot occupy a post-patch key); "
            "key WITHOUT -> [4210] stands, force. **[4348]'s one-time "
            "exemption is SUPERSEDED BY THIS RULE, NOT SPENT BY THIS RUN** — "
            "a rule grants by construction and is checkable, where a "
            "spent-marker is bookkeeping a later run could inherit silently. "
            "This run's non-thinking shape served warm on MEASUREMENT first "
            "([4355]: entries overwritten in place, newest-wins verified at "
            "two seats) and [4348] second."),
        "_population_duplicate": (
            "One (prompt, faller, riser) triple carries BOTH a REAL and a "
            "NEAR-MISS row — 'pay -> implement', institutional, slot "
            "unassigned. Both are coded and the coder sees identical inputs. "
            "Neither is ACT or REF so no primary is touched; any "
            "decoy-relative descriptive rate must exclude it or state it is "
            "in. 4,443 rows over 4,442 distinct triples."),
        "_coverage_gaps": gaps or None,
        "_partition": {"rows_coded": len(coded),
                       "distinct_triples": len(set(keys)),
                       "roles": dict(collections.Counter(roles.values()))},
        "coders": meta,
        "agreement": agree,
        "primaries": prim,
        "reading": reading,
    }
    if os.path.exists(OUT):
        os.makedirs(ESCROW, exist_ok=True)
        prior = open(OUT, "rb").read()
        h = hashlib.sha256(prior).hexdigest()[:16]
        dst = os.path.join(ESCROW, "result_p_primary.PREFIX-%s.json" % h)
        if not os.path.exists(dst):
            open(dst, "wb").write(prior)
            os.chmod(dst, 0o444)
        os.chmod(OUT, 0o644)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    atomic_dump(payload, OUT)
    os.chmod(OUT, 0o444)
    print("\nwrote %s @ %s  RE-LOCKED a-w" % (
        os.path.basename(OUT),
        hashlib.sha256(open(OUT, "rb").read()).hexdigest()[:16]), flush=True)
    return 0
