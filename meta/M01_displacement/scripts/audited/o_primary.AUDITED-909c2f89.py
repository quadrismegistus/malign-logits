#!/usr/bin/env python3
"""REGISTRATION O's PRODUCER — H1/H2/H3, BOTH ARMS. The campaign's first crosslingual run.

Frozen registration: `registration_o_crosslingual.md`
  @ cb8518528077f7d0e370db4b980fe420404eb4264ae1a62a429f97ab502a0e17
  committed aa03cc82c3fd9232b0c7800f4abdab47e58cc41f, RH's word "Freeze O".
  Run authorized by RH, "Run O now that it's frozen."

**THIS FILE IMPLEMENTS THE FROZEN TEXT AND NOTHING ELSE.** Where the text is
silent it RAISES rather than chooses. Sections are cited by number; a producer
citing a docket message cites something a reader of this repository cannot
resolve (Amendment D3b-B §B7).

    §O0    the predicates, pinned to the COMMIT and not the constants alone
    §O1    301 ratified pairs, content pin 7ad8a39d1ac85d48
    §O1.1  two filters: capacity (cjk_tier) then COMPETENCE (>= 0.30)
    §O1.2  seven clauses for "yields an A", including the join and denominator
    §O3    H1 tail_excess < 0; H2 A_|valence| > 0; H3 A_arousal > 0, BOTH ARMS
    §O4    the four-row reading rule, OPPOSED reported as REVERSAL
    §O5    per-arm bias columns per cell

**THE FIVE SILENCES, RULED BEFORE ANY QUANTITY EXISTED.** Each is a declared
constant below; each RAISES if it is ever set otherwise. The parse is the
producer's, the rulings are the pen's, and both predate the first number.

**NO SEED.** Nothing here samples, splits or shuffles. O registers no
permutation null, so `m01_norms.A_and_null` is deliberately NOT used.

**NO ARM IS PRIMARY (§O4).** en and zh carry identical machinery and weight.

    python meta/M01_displacement/scripts/o_primary.py [--dry-run]
"""

import collections
import hashlib
import json
import math
import os
import re
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

REGISTRATION = os.path.join(CAMPAIGN, "registrations",
                            "registration_o_crosslingual.md")
REGISTRATION_SHA = ("cb8518528077f7d0e370db4b980fe420404eb4264ae1a62a429f97"
                    "ab502a0e17")
POPULATION = os.path.join(CAMPAIGN, "populations", "population_o_pairs.json")
PAIR_SET_SHA16 = "7ad8a39d1ac85d48"
OUT = os.path.join(CAMPAIGN, "results", "result_o_primary.json")
ESCROW = os.path.join(CAMPAIGN, "results", "superseded")

#: §O0. The pin carries the COMMIT, not the constants alone: both sides of the
#: 2026-08-04 repair satisfy (0.003, 0.5) and select different faller sets in
#: 11% of cells.
MOVEMENT_PIN = "e7864dab"
#: §O0's floor, the `true_word_probs` retention floor, restated at use.
THETA = 0.001
#: §O1.2 clause 6.
QUALIFYING_MIN = 3

# ══════════════════════════════════════════════════════════════════════════
# THE FIVE SILENCES — parsed at [4122], ruled at [4123], before any number.
# A constant set to anything else RAISES; none of them is a runtime option.
# ══════════════════════════════════════════════════════════════════════════
#: 1. H1 runs on ALL ANALYSED cells (post zero-faller), not on A-cells. H1 is
#:    distributional and needs no norms; restricting it to A-cells would make it
#:    depend on LEXICON COVERAGE, a dependence nowhere registered, and would
#:    import §O1.2's clauses into a hypothesis they were scoped away from.
H1_CELL_SET = "analysed"
#: 2. The competence partition is re-asserted on ALL 301 zh prompts, and it is
#:    the PARTITION that is checked, never the values: §O1.1's void runs 0.0674
#:    to 0.5365, so no sample choice inside it can move the cut.
COMPETENCE_BASIS = "all-301-zh"
COMPETENCE_MIN = 0.30
#: 3. Three agreement rates, named separately. NO aggregate: a bare singular
#:    would be quoted and three numbers cannot share one name.
AGREEMENT_PER_STATISTIC = True
#: 4. The push column beside H1 where it is defined; NOTHING for H2/H3. An
#:    absent column is a stated gap; a fabricated one is a claim.
BIAS_COLUMN_FOR = ("H1",)
#: 5. The ties clause imports with the form (§O3 "identical to N's"): a cell
#:    whose statistic is exactly 0.0 is EXCLUDED, not split, and the
#:    denominator is cells with a NON-ZERO statistic.
TIES = "exclude"
#: 6. A cell whose §N6 reconstruction cross-check refuses KEEPS its raw
#:    `tail_excess` and carries push=None, COUNTED. H1 needs no reconstruction;
#:    dropping the cell would shrink a REGISTERED population for a DIAGNOSTIC's
#:    sake. N dropped such cells because N's primary WAS the corrected
#:    statistic -- the difference is the designs', not the seats'. Named at
#:    [4124] before the audit, ruled at [4125]. A non-zero count posts before
#:    any hypothesis quantity is read.
CROSS_CHECK_REFUSAL = "keep-cell-push-none"

_RULED = {"H1_CELL_SET": "analysed", "COMPETENCE_BASIS": "all-301-zh",
          "AGREEMENT_PER_STATISTIC": True, "BIAS_COLUMN_FOR": ("H1",),
          "TIES": "exclude", "CROSS_CHECK_REFUSAL": "keep-cell-push-none"}

#: §O1.1's named case, asserted as the excluded one rather than trusted as it.
COMPETENCE_EXCLUDED_NAMES = "bloomz"

CJK_ONLY = re.compile(r"^[一-鿿]+$")


# ══════════════════════════════════════════════════════════════════════════
# GATES — called, not remembered.
# ══════════════════════════════════════════════════════════════════════════
def gate_registration():
    """`require_frozen` before ANY read, and the bytes named as well.

    The gate proves the registration is frozen; it does not prove WHICH frozen
    registration. A producer that passes the gate against a different frozen
    document has satisfied the ceremony and answered another question.
    """
    import freeze_gate
    freeze_gate.require_frozen(REGISTRATION)
    got = hashlib.sha256(open(REGISTRATION, "rb").read()).hexdigest()
    if got != REGISTRATION_SHA:
        raise SystemExit(
            "REFUSING: the registration hashes %s, this producer was written "
            "against %s." % (got[:16], REGISTRATION_SHA[:16]))
    print("gate     registration FROZEN and IDENTIFIED: %s" % got[:16], flush=True)


def gate_rulings():
    """The five silences are what [4123] ruled, or this does not run.

    A constant that drifts from its ruling is the defect this whole parse
    exists to prevent, and it would drift silently.
    """
    live = {"H1_CELL_SET": H1_CELL_SET, "COMPETENCE_BASIS": COMPETENCE_BASIS,
            "AGREEMENT_PER_STATISTIC": AGREEMENT_PER_STATISTIC,
            "BIAS_COLUMN_FOR": BIAS_COLUMN_FOR, "TIES": TIES,
            "CROSS_CHECK_REFUSAL": CROSS_CHECK_REFUSAL}
    if live != _RULED:
        bad = {k: (live[k], _RULED[k]) for k in live if live[k] != _RULED[k]}
        raise SystemExit("REFUSING: a ruled constant was changed: %r" % bad)
    print("gate     %d ruled constants match the ruling ([4123] x5, [4125] x1)"
          % len(_RULED), flush=True)


def gate_movement_pin():
    """§O0 pins the COMMIT. Constants alone do not identify the instrument.

    Both sides of the residual-as-faller repair satisfy `min_prob 0.003,
    fall_ratio 0.5` and disagree on 11% of faller sets. Asserting the pair
    would certify the wrong instrument; asserting the BLOB certifies this one.
    """
    from malign_logits.movement import CANONICAL
    if (CANONICAL.min_prob, CANONICAL.fall_ratio) != (0.003, 0.5):
        raise SystemExit("REFUSING: CANONICAL is (%r, %r), §O0 declares "
                         "(0.003, 0.5)." % (CANONICAL.min_prob,
                                            CANONICAL.fall_ratio))
    path = "malign_logits/movement.py"
    here = subprocess.run(["git", "hash-object", os.path.join(ROOT, path)],
                          capture_output=True, text=True, cwd=ROOT).stdout.strip()
    pinned = subprocess.run(["git", "rev-parse", "%s:%s" % (MOVEMENT_PIN, path)],
                            capture_output=True, text=True, cwd=ROOT).stdout.strip()
    if not here or not pinned:
        raise SystemExit("REFUSING: could not resolve %s at %s"
                         % (path, MOVEMENT_PIN))
    if here != pinned:
        raise SystemExit(
            "REFUSING: %s is blob %s; §O0 pins the instrument at commit %s "
            "whose blob is %s. The constants match and the CODE DOES NOT."
            % (path, here[:12], MOVEMENT_PIN, pinned[:12]))
    print("gate     §O0 instrument pinned: movement.py == %s:%s (blob %s)"
          % (MOVEMENT_PIN, path, here[:12]), flush=True)


# ══════════════════════════════════════════════════════════════════════════
# §O1 — POPULATION, read from the ratified enumeration and never re-derived.
# ══════════════════════════════════════════════════════════════════════════
def population():
    """The 301 ratified pairs. The CONTENT pin is checked, not the file hash.

    A whole-file hash moves whenever `_status` moves -- it did, at ratification
    -- so a producer pinning the file pins the hash guaranteed to change for
    reasons that are not the population.
    """
    d = json.load(open(POPULATION))
    pairs = d["pairs"]
    s = "\n".join(sorted(p["english"] + "\t" + p["chinese"] for p in pairs))
    pin = hashlib.sha256(s.encode("utf-8")).hexdigest()
    if pin[:16] != PAIR_SET_SHA16:
        raise SystemExit("REFUSING: pair-set pin is %s, §O1 declares %s"
                         % (pin[:16], PAIR_SET_SHA16))
    if len(pairs) != 301:
        raise SystemExit("REFUSING: §O1 declares 301 pairs, found %d" % len(pairs))
    if not str(d.get("_status", "")).startswith("RATIFIED"):
        raise SystemExit("REFUSING: the enumeration is not RATIFIED.")
    print("§O1      population %d pairs   content pin %s   RATIFIED"
          % (len(pairs), pin[:16]), flush=True)
    return pairs, pin


def fluent_edges():
    """§O1.1's FIRST filter: `cjk_tier` FLUENT on BOTH sides. CAPACITY."""
    import m01_concentration as CC
    reg = json.load(open(os.path.join(ROOT, "data", "model_registry.json")))
    models = reg["models"] if isinstance(reg, dict) and "models" in reg else reg
    rows = models.values() if isinstance(models, dict) else models
    tier = {}
    for m in rows:
        if isinstance(m, dict):
            mid = m.get("id") or m.get("model_id") or m.get("hf_id")
            if mid:
                tier[mid] = m.get("cjk_tier") or ""
    _p, mods, _h, _d = CC.frozen_population()
    edges, _dropped = CC.operation_edges(mods)

    def mid(o):
        return getattr(o, "id", None) or getattr(o, "model_id", None) or str(o)

    out = []
    for fam, _pos, step in edges:
        a, b = mid(step.pre), mid(step.post)
        if tier.get(a) == "FLUENT" and tier.get(b) == "FLUENT":
            out.append((fam, a, b, step))
    return out


def competence(model, zh_texts):
    """§O1.1's SECOND filter: retained mass on CJK-ONLY forms. COMPETENCE.

    `cjk_tier` says the tokenizer CAN represent Chinese. This says what the
    model DOES with it. A mixed Latin/CJK form is a tokenizer artefact of
    exactly the kind being measured and counts toward neither side.
    """
    from malign_logits.movement import word_probs
    cjk, n = 0.0, 0
    for t in zh_texts:
        wp = word_probs(model, t)
        if wp is None:
            continue
        n += 1
        for w, p in wp.probs.items():
            if CJK_ONLY.match(w):
                cjk += p
    return (cjk / n) if n else None


def qualifying_edges(pairs):
    """10 -> 9 by COMPETENCE, asserting the PARTITION and never the values."""
    edges = fluent_edges()
    zh = [p["chinese"] for p in pairs if p.get("chinese")]
    shares, missing = {}, []
    for _fam, a, b, _s in edges:
        for m in (a, b):
            if m not in shares:
                c = competence(m, zh)
                if c is None:
                    missing.append(m)
                else:
                    shares[m] = c
    if missing:
        raise SystemExit("REFUSING: no scored cell for %r" % missing)
    below = sorted(k for k, v in shares.items() if v < COMPETENCE_MIN)
    above = sorted(k for k, v in shares.items() if v >= COMPETENCE_MIN)
    print("§O1.1    capacity-fluent edges %d over %d models; competence "
          ">= %.2f keeps %d, drops %d"
          % (len(edges), len(shares), COMPETENCE_MIN, len(above), len(below)),
          flush=True)
    #: THE PARTITION IS THE KNOWN ANSWER. §O1.1 publishes 10 -> 9 with exactly
    #: one model excluded and names it. Any other partition stops the run.
    if len(edges) != 10:
        raise SystemExit("REFUSING: §O1.1 publishes 10 capacity-fluent edges, "
                         "found %d" % len(edges))
    if len(below) != 1 or len(above) != 19:
        raise SystemExit(
            "REFUSING: §O1.1's partition is 19 above / 1 below. Found %d / %d: "
            "below=%r" % (len(above), len(below), below))
    if COMPETENCE_EXCLUDED_NAMES not in below[0]:
        raise SystemExit("REFUSING: §O1.1 names %r as the excluded case; the "
                         "derivation excluded %r" % (COMPETENCE_EXCLUDED_NAMES,
                                                     below[0]))
    drop = set(below)
    kept = [(f, a, b, s) for f, a, b, s in edges if a not in drop and b not in drop]
    if len(kept) != 9:
        raise SystemExit("REFUSING: §O1.1 publishes 9 qualifying edges, kept %d"
                         % len(kept))
    if len({a for _f, a, _b, _s in kept}) != 9:
        raise SystemExit("REFUSING: §O1 declares 9 DISTINCT base checkpoints.")
    print("§O1.1    qualifying edges 9 over 9 distinct bases; excluded %s "
          "(share %.4f, §O1.1's named case)" % (below[0], shares[below[0]]),
          flush=True)
    return kept, shares


# ══════════════════════════════════════════════════════════════════════════
# §N6 (imported with the form) — the reconstruction behind §O5's bias column.
# ══════════════════════════════════════════════════════════════════════════
def reconstruct(P, Q, m):
    """(R, S, n_unscored) or None. S IS RECOMPUTED, NEVER DIVIDED OUT.

    `S = R/inflation` is arithmetically true only because the residual was
    excluded from faller candidacy on 2026-08-04; a producer dividing by
    `inflation` is correct while one line of `_movement` stays as it is and has
    no way to notice if it moves. A cross-check over `m.null` is FORBIDDEN --
    `movement()` pops RESIDUAL_KEY from null/excess/delta (load-bearing, since
    `top_riser()` is an argmax the bucket would win), so `sum(m.null.values())`
    is short by the residual's pre-mass.
    """
    keys = set(P) | set(Q)
    fall = set(m.fallers)
    R = 1.0 - sum(Q.get(w, 0.0) for w in fall)
    S = sum(P.get(k, 0.0) for k in keys if k not in fall)
    n_unscored = sum(1 for w in fall if Q.get(w, 0.0) == 0.0)
    if m.inflation != m.inflation:            # NaN: the null was not computed
        return None
    if S <= 0 or abs(R / S - m.inflation) > 1e-9:
        return None
    return R, S, n_unscored


# ══════════════════════════════════════════════════════════════════════════
# §O3 — the statistics
# ══════════════════════════════════════════════════════════════════════════
def _ppf(q):
    """Inverse standard-normal CDF by bisection. No scipy dependency."""
    if q <= 0:
        return -40.0
    if q >= 1:
        return 40.0
    lo, hi = -40.0, 40.0
    for _ in range(300):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sign_test(values, predict):
    """Exact one-sided sign test in the REGISTERED direction.

    `predict` is -1 (H1: tail_excess < 0) or +1 (H2/H3: A > 0). z is POSITIVE
    when the registered direction dominates, so Stouffer accumulates evidence
    FOR the hypothesis as positive and a REVERSAL shows as a negative Z.

    TIES (§O3's imported form): a cell whose statistic is exactly 0.0 is
    EXCLUDED, not split; the denominator is cells with a NON-ZERO statistic.
    """
    if predict not in (-1, 1):
        raise ValueError("predict must be -1 or +1, not %r" % predict)
    if TIES != "exclude":
        raise SystemExit("REFUSING: the ties clause is %r" % TIES)
    hit = sum(1 for v in values if (v < 0 if predict < 0 else v > 0))
    miss = sum(1 for v in values if (v > 0 if predict < 0 else v < 0))
    tie = sum(1 for v in values if v == 0)
    n = hit + miss
    if n == 0:
        return hit, miss, tie, None, None
    p = sum(math.comb(n, i) for i in range(hit, n + 1)) / (2 ** n)
    p = min(1.0, max(p, 1e-300))
    if hit == miss:
        return hit, miss, tie, p, 0.0
    z = abs(_ppf(p)) * (1 if hit > miss else -1)
    return hit, miss, tie, p, z


def stouffer(zs):
    """§O3: equal weight per CLUSTER, over O's NINE base checkpoints."""
    zs = [z for z in zs if z is not None]
    if not zs:
        return None, None, 0
    Z = sum(zs) / math.sqrt(len(zs))
    p = 1.0 - 0.5 * (1 + math.erf(Z / math.sqrt(2)))          #: one-sided
    return Z, p, len(zs)


def run_arm(rows, field, fam_base, predict):
    """One (arm, hypothesis). Cells -> family z -> cluster z -> Stouffer."""
    by_fam = collections.defaultdict(list)
    for r in rows:
        if r.get(field) is not None:
            by_fam[r["family"]].append(r[field])
    per_fam = {}
    for fam, vals in sorted(by_fam.items()):
        hit, miss, tie, p, z = sign_test(vals, predict)
        per_fam[fam] = {"n": len(vals), "n_hit": hit, "n_miss": miss,
                        "n_tie": tie, "p": p, "z": z}
    by_base = collections.defaultdict(list)
    for fam, rec in per_fam.items():
        if rec["z"] is not None:
            by_base[fam_base[fam]].append(rec["z"])
    clusters = {b: sum(v) / len(v) for b, v in by_base.items()}
    Z, p, k = stouffer(list(clusters.values()))
    hit = sum(v["n_hit"] for v in per_fam.values())
    miss = sum(v["n_miss"] for v in per_fam.values())
    tie = sum(v["n_tie"] for v in per_fam.values())
    return {"field": field, "predict": predict, "families": per_fam,
            "clusters": clusters, "n_clusters": k, "stouffer_Z": Z,
            "stouffer_p": p, "n_cells": sum(v["n"] for v in per_fam.values()),
            "n_hit": hit, "n_miss": miss, "n_tie": tie,
            "sign_split_registered": (hit / (hit + miss)) if (hit + miss) else None}


# ══════════════════════════════════════════════════════════════════════════
# §O4 — THE READING RULE, fixed before any number.
# ══════════════════════════════════════════════════════════════════════════
ALPHA = 0.05


def arm_outcome(a):
    """confirms | null | OPPOSED. A one-sided test has THREE outcomes, not two.

    OPPOSED is significance at the SAME alpha in the direction opposite to the
    registered one. Collapsing it into "does not confirm" is the defect §O4
    exists to prevent: `en confirms / zh null` is "we could not detect it in
    Chinese"; `en confirms / zh OPPOSED` is "the mechanism RUNS BACKWARDS".
    """
    Z, p = a["stouffer_Z"], a["stouffer_p"]
    if Z is None or p is None:
        return "null"
    if Z > 0 and p < ALPHA:
        return "confirms"
    #: the opposite tail at the same alpha; p is the one-sided upper tail, so
    #: the lower tail is 1 - p.
    if Z < 0 and (1.0 - p) < ALPHA:
        return "OPPOSED"
    return "null"


def reading(en, zh):
    """§O4's four rows, implemented as written. Never a choice at read time."""
    a, b = arm_outcome(en), arm_outcome(zh)
    if a == "confirms" and b == "confirms":
        return "SUPPORTED IN BOTH ARMS", (a, b)
    if "OPPOSED" in (a, b) and "confirms" in (a, b):
        return ("NOT SUPPORTED, AND REPORTED AS A REVERSAL "
                "(§O4: never as an asymmetry)"), (a, b)
    if "confirms" in (a, b):
        arm = "en" if a == "confirms" else "zh"
        return ("NOT SUPPORTED -- ASYMMETRY, confirming arm %s (§O4: a "
                "hypothesis stated as 'in both arms' is not confirmed by one)"
                % arm), (a, b)
    if a == "OPPOSED" or b == "OPPOSED":
        return "NOT SUPPORTED -- OPPOSED without a confirming arm", (a, b)
    return "NOT SUPPORTED", (a, b)


# ══════════════════════════════════════════════════════════════════════════
def collect(pairs, edges):
    """One pass. Per (edge, pair, arm): H1's cell, and H2/H3's where they exist."""
    from malign_logits.movement import (movement, decompose, CANONICAL,
                                        RESIDUAL_KEY)
    import m01_norms as N

    norms, _f, _r = N.load_norms(verify=True)
    tabs = {arm: {d: norms[(arm, d, "primary")] for d in ("arousal", "valence")}
            for arm in ("en", "zh")}

    rows = {"en": [], "zh": []}
    diag = {"en": collections.Counter(), "zh": collections.Counter()}
    for ei, (fam, pre, post, step) in enumerate(edges, 1):
        for pair in pairs:
            for arm, key in (("en", "english"), ("zh", "chinese")):
                text = pair.get(key)
                if not text:
                    diag[arm]["pair side absent"] += 1
                    continue
                c = step.cell(text)
                if not c.is_present:
                    diag[arm]["cell absent"] += 1
                    continue
                m = c.movement(CANONICAL)
                #: zero-faller cells are EXCLUDED because the CLAIM does not
                #: apply: no mass departed, so nothing can have landed.
                #: Counted, and §O1.2 clause 7's denominator is what remains.
                if m is None or not m.fallers:
                    diag[arm]["zero-faller"] += 1
                    continue
                A, B = c.pre, c.post
                d = decompose(A.probs, B.probs,
                              residual_pre=A.residual, residual_post=B.residual)
                P = {**A.probs, RESIDUAL_KEY: A.residual}
                Q = {**B.probs, RESIDUAL_KEY: B.residual}
                #: §O5's bias column, N's form. A cell whose reconstruction
                #: cross-check refuses keeps its `tail_excess` -- H1 needs no
                #: reconstruction -- and carries push=None, counted.
                rec = reconstruct(P, Q, m)
                if rec is None:
                    diag[arm]["push undefined (§N6 cross-check)"] += 1
                    push = None
                else:
                    _R, S, n_unscored = rec
                    dR = min(n_unscored * THETA, B.residual)
                    push = A.residual * dR / S
                #: §O1.2's seven clauses. The join is `N.lookup`'s LEMMA
                #: CANDIDATES (clause 4) in the ARM'S OWN tables (clause 3),
                #: both dimensions required (clause 2), function words out
                #: (clause 5). Clause 4's stated limit: no second derivation
                #: of these can be independent of `N.lookup`.
                fal, ris = [], []
                for w, wt, role in N.cell_roles(c, "CANONICAL"):
                    k = N.norm_key(w, arm, fold=False)
                    if N.is_function_word(k, arm):
                        continue
                    zs = {}
                    for dim in ("valence", "arousal"):
                        v, _how = N.lookup(tabs[arm][dim], k.casefold(), arm)
                        if v is None:
                            break
                        zs[dim] = v
                    if len(zs) != 2:
                        continue
                    (fal if role == "faller" else ris).append((wt, zs))
                a_val = a_aro = None
                #: clause 6: >= 3 fallers AND >= 3 risers. Both roles.
                if len(fal) >= QUALIFYING_MIN and len(ris) >= QUALIFYING_MIN:
                    a_val = (N.weighted_mean([(w, abs(z["valence"])) for w, z in fal])
                             - N.weighted_mean([(w, abs(z["valence"])) for w, z in ris]))
                    a_aro = (N.weighted_mean([(w, z["arousal"]) for w, z in fal])
                             - N.weighted_mean([(w, z["arousal"]) for w, z in ris]))
                    diag[arm]["A-cell"] += 1
                rows[arm].append({
                    "family": fam, "base": pre, "aligned": post,
                    "pair_en": pair["english"], "prompt": text,
                    "tail_excess": d["tail_excess"],
                    "A_absvalence": a_val, "A_arousal": a_aro,
                    "push": push,
                    "n_fallers": d["n_fallers"], "n_risers": d["n_risers"],
                    "n_scored_fallers": len(fal), "n_scored_risers": len(ris),
                })
        print("  [%d/%d] %-22s en %5d  zh %5d" % (
            ei, len(edges), fam, len(rows["en"]), len(rows["zh"])), flush=True)
    return rows, diag


def agreement(rows, field):
    """§O3's REPORTED-NEVER-TESTED within-pair rate, per statistic.

    Three statistics cannot share one name, so there is no aggregate and this
    is never called without a field.
    """
    idx = {}
    for arm in ("en", "zh"):
        for r in rows[arm]:
            v = r.get(field)
            if v is None or v == 0:
                continue
            idx.setdefault((r["base"], r["aligned"], r["pair_en"]), {})[arm] = v
    both = [d for d in idx.values() if len(d) == 2]
    same = sum(1 for d in both if (d["en"] > 0) == (d["zh"] > 0))
    return {"n_pairs_both_arms": len(both), "n_same_sign": same,
            "rate": (same / len(both)) if both else None}


def main():
    gate_registration()
    gate_rulings()
    gate_movement_pin()

    pairs, pin = population()
    edges, shares = qualifying_edges(pairs)
    fam_base = {f: a for f, a, _b, _s in edges}
    reach = len(pairs) * len(edges)
    print("§O1      reachable %d per arm  (301 x 9)" % reach, flush=True)
    if reach != 2709:
        raise SystemExit("REFUSING: §O1 declares 2,709 cells per arm.")
    if "--dry-run" in sys.argv:
        print("\n--dry-run: gates, population, edges. No cell opened.")
        return 0

    rows, diag = collect(pairs, edges)

    # ── KNOWN ANSWERS, fired BEFORE any hypothesis quantity is read ────────
    print("\n=== KNOWN ANSWERS (§O1.3), re-derived and asserted", flush=True)
    known = {"en": {"zero_faller": 67, "A": 1844, "analysed": 2642},
             "zh": {"zero_faller": 105, "A": 1298, "analysed": 2604}}
    for arm in ("en", "zh"):
        zf, na = diag[arm]["zero-faller"], diag[arm]["A-cell"]
        an = len(rows[arm])
        print("  %s  cells %d  zero-faller %d (pub %d)  analysed %d (pub %d)  "
              "A-cells %d (pub %d)" % (arm, zf + an, zf, known[arm]["zero_faller"],
                                       an, known[arm]["analysed"],
                                       na, known[arm]["A"]), flush=True)
        if zf + an != 2709:
            raise SystemExit("REFUSING: %s cells %d, §O1 declares 2,709."
                             % (arm, zf + an))
        for name, got, want in (("zero-faller", zf, known[arm]["zero_faller"]),
                                ("analysed", an, known[arm]["analysed"]),
                                ("A-cells", na, known[arm]["A"])):
            if got != want:
                raise SystemExit(
                    "REFUSING: %s %s is %d, §O1.3 publishes %d. The producer "
                    "and the frozen text disagree about the POPULATION; no "
                    "hypothesis quantity may be read past this line."
                    % (arm, name, got, want))
    print("  ALL KNOWN ANSWERS MATCH.", flush=True)

    # ── §O3 ────────────────────────────────────────────────────────────────
    H = [("H1", "tail_excess", -1, "substitution: tail_excess < 0"),
         ("H2", "A_absvalence", +1, "|valence| decreases: A > 0"),
         ("H3", "A_arousal", +1, "arousal decreases: A > 0")]
    results, readings = {}, {}
    for name, field, predict, gloss in H:
        en = run_arm(rows["en"], field, fam_base, predict)
        zh = run_arm(rows["zh"], field, fam_base, predict)
        verdict, arms = reading(en, zh)
        results[name] = {"gloss": gloss, "field": field, "predict": predict,
                         "en": en, "zh": zh, "arm_outcomes": {"en": arms[0],
                                                              "zh": arms[1]},
                         "reading": verdict}
        readings[name] = verdict
        print("\n%s  %s" % (name, gloss), flush=True)
        for arm, a, out in (("en", en, arms[0]), ("zh", zh, arms[1])):
            print("  %s  cells %5d  clusters %d  Z %+.4f  p %.6g  split %.4f"
                  "  -> %s" % (arm, a["n_cells"], a["n_clusters"],
                               a["stouffer_Z"] if a["stouffer_Z"] is not None else float("nan"),
                               a["stouffer_p"] if a["stouffer_p"] is not None else float("nan"),
                               a["sign_split_registered"] if a["sign_split_registered"] is not None else float("nan"),
                               out), flush=True)
        print("  §O4 READING: %s" % verdict, flush=True)

    # ── §O3's reported-never-tested description ────────────────────────────
    agree = {name: agreement(rows, field) for name, field, _p, _g in H}
    print("\n=== WITHIN-PAIR AGREEMENT (REPORTED, NEVER TESTED)", flush=True)
    for name, a in agree.items():
        print("  %s  %s of %s cells share a sign  %s" % (
            name, a["n_same_sign"], a["n_pairs_both_arms"],
            ("%.4f" % a["rate"]) if a["rate"] is not None else "n/a"), flush=True)

    # ── §O5's per-arm bias columns ─────────────────────────────────────────
    bias = {}
    for arm in ("en", "zh"):
        ps = [r["push"] for r in rows[arm] if r["push"] is not None]
        bias[arm] = {"n": len(ps), "n_undefined": sum(1 for r in rows[arm]
                                                      if r["push"] is None),
                     "median": st.median(ps) if ps else None,
                     "max": max(ps) if ps else None,
                     "all_non_negative": all(p >= 0 for p in ps)}
    print("\n=== §O5 BIAS COLUMN (H1 only; §O5 defines none for H2/H3)", flush=True)
    for arm in ("en", "zh"):
        b = bias[arm]
        print("  %s  push median %.3e  max %.3e  (undefined %d)"
              % (arm, b["median"], b["max"], b["n_undefined"]), flush=True)

    payload = {
        "_what": "Registration O — H1/H2/H3, both arms. First crosslingual run.",
        "_registration": REGISTRATION_SHA,
        "_registration_commit": "aa03cc82c3fd9232b0c7800f4abdab47e58cc41f",
        "_population": {"pairs": len(pairs), "pair_set_sha256_16": pin[:16],
                        "edges": len(edges), "clusters": len(set(fam_base.values())),
                        "per_arm_cells": reach,
                        "diagnostics": {a: dict(diag[a]) for a in diag}},
        "_ruled_constants": {k: list(v) if isinstance(v, tuple) else v
                             for k, v in _RULED.items()},
        "_competence_shares": shares,
        "_readings": readings,
        "hypotheses": results,
        "within_pair_agreement": agree,
        "bias_columns": bias,
        "cells": rows,
    }
    if os.path.exists(OUT):
        os.makedirs(ESCROW, exist_ok=True)
        prior = open(OUT, "rb").read()
        h = hashlib.sha256(prior).hexdigest()[:16]
        dst = os.path.join(ESCROW, "result_o_primary.PREFIX-%s.json" % h)
        if not os.path.exists(dst):
            with open(dst, "wb") as fh:
                fh.write(prior)
            os.chmod(dst, 0o444)
        print("\n  escrowed prior artifact @ %s" % h, flush=True)
        os.chmod(OUT, 0o644)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    os.chmod(OUT, 0o444)
    print("  wrote %s @ %s  RE-LOCKED a-w"
          % (os.path.basename(OUT),
             hashlib.sha256(open(OUT, "rb").read()).hexdigest()[:16]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
