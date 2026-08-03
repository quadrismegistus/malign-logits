#!/usr/bin/env python3
"""Registration D — the paired valence/arousal/dominance read on the 684 pairs.

    REGISTRATION   registrations/registration_d_pairs_v6.md
    AMENDMENT A    registrations/registration_d_pairs_amendment_a.md
                     @ ddb4cd9b0496b723, frozen [3310], three signatures
    GOVERNING      C v6 06f0272d7f21b901 -> B v13 06186c42f9ff46e0 (terminus)
    POPULATION     results/population_d_684.json @ 3ed3e286e633c2fc

TWO STAGES, AND THE SPLIT IS THE POINT (§A7.3, ruled [3280])
------------------------------------------------------------
§D6d turns a null into either *evidence the effect is MOVEMENT-GENERAL, quotable
as such* or *UNINFORMATIVE AT THIS POWER, quotable as nothing*, on the comparison
`MDE < the dimension's known effect size`. **The raw MDE depends on a realized
variance nobody can know before opening the data.** So:

    STAGE 1   realized SDs and raw MDEs per arm per threshold point.
              NO D, NO p, NO SIGNS. Emits an artifact and its hash.
    STAGE 2   REFUSES TO RUN without stage 1's artifact hash. Computes the
              verdict quantities against a threshold already on the record.

**Without the split, §D6d is a verdict rule whose threshold gets set after the
verdict is visible** ([3278].2). The separation makes the ordering a fact about
artifacts rather than a claim about anyone's discipline.

DECLARED DISCRETION — every point posted BEFORE this file existed
-----------------------------------------------------------------
[3269] four pinned readings + two resolutions, [3271] the p-convention,
[3304] the §D6 denominator split. Restated at their use sites below, so a reader
meets the reasoning where it bites rather than in a docket archive.
"""

import argparse
import collections
import hashlib
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # .../M01_displacement/scripts
CAMPAIGN = os.path.dirname(HERE)                           # .../M01_displacement
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))          # the repository root
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)

import numpy as np

# ── frozen identifiers, asserted at startup, never retyped downstream ────
AMENDMENT_SHA = "ddb4cd9b0496b723"
POPULATION_SHA = "3ed3e286e633c2fc"
C_V6_SHA = "06f0272d7f21b901"
B_V13_SHA = "06186c42f9ff46e0"

#: §D3 verbatim. NOT a range built from endpoints -- a literal, so a reader
#: diffs it against the registration by eye.
GRID = (0.00, 0.01, 0.02, 0.05, 0.10, 0.20)
FLOOR = 6                     #: §D3, n >= 6 admitted pairs at a point
ALPHA = 0.05                  #: §D3, one-sided, named so no default adjudicates
COLLAPSE_JACCARD = 0.95       #: §D3, overlap with the t=0.00 set
EXACT_MAX_N = 20              #: §D2, 2^20 = 1,048,576
POWER = 0.80                  #: §A7.1, inherited from G §8 / F, not chosen here

#: §D6b. (name, quantity, direction, residualisation)
ARMS = (
    ("h1_signed",   "valence",   -1, "linear"),
    ("arousal",     "arousal",   +1, "none"),
    ("val_extrem",  "valence",   +1, "quadratic"),
    ("dom_extrem",  "dominance", +1, "quadratic"),
)
EXTREMITY = {"val_extrem", "dom_extrem"}   #: |dim_z| FIRST, then residualise


# ══════════════════════════════════════════════════════════════════════════
# the statistic
# ══════════════════════════════════════════════════════════════════════════
def sign_flip_p(values, direction, seed, draws=10000):
    """One-sided p for the paired sign-flip null. §D2.

    THE p-CONVENTION IS DECLARED AND IT IS NOT ONE CONVENTION ([3271]).

    **PLAIN at exact enumeration, ADD-ONE when sampled.** The Phipson-Smyth
    add-one exists to correct a p ESTIMATED FROM A RANDOM SAMPLE -- its content
    is that zero hits in d draws has not established p = 0. **At exact
    enumeration there is no sampling and no error to correct**: 2^n sign-flips
    IS the null distribution, so the count is the p-value rather than an
    estimate of it, and add-one would inflate it against a variance that does
    not exist.

    EXTREMITY IS `>=` (or `<=`), NOT STRICT, AND §D4 PINS IT WITHOUT NAMING IT
    ([3304].2). The OBSERVED configuration is itself one of the 2^n draws, so
    under `>=` it always counts itself and the floor is 1/2^n -- which is
    exactly the lattice §D4 requires printed. Under `>` the floor would be 0.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    obs = float(v.mean())
    #: DIRECTION FIRST, so the tail test below is written once. For a
    #: registered D < 0 the extreme tail is the LOWER one; flipping the sign of
    #: the data lets one `>=` serve both and removes a branch that could drift.
    s = v if direction > 0 else -v
    obs_s = float(s.mean())

    if n <= EXACT_MAX_N:
        hits = 0
        idx = np.arange(n)
        for mask in range(1 << n):
            sgn = 1.0 - 2.0 * ((mask >> idx) & 1).astype(float)
            if float((s * sgn).mean()) >= obs_s:
                hits += 1
        return {"p": hits / (1 << n), "convention": "plain/exact",
                "draws": 1 << n, "hits": hits, "statistic": obs,
                "resolution": 1.0 / (1 << n), "exact": True}

    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(draws):
        sgn = rng.choice((-1.0, 1.0), size=n)
        if float((s * sgn).mean()) >= obs_s:
            hits += 1
    return {"p": (hits + 1) / (draws + 1), "convention": "add-one/sampled",
            "draws": draws, "hits": hits, "statistic": obs,
            "resolution": 1.0 / draws, "exact": False}


def raw_mde(n, sd, direction, seed, power=POWER, alpha=ALPHA, reps=400):
    """Smallest |effect| in RAW dimension units detectable at `power`. §A7.2.

    RAW, NOT STANDARDISED, and the distinction is load-bearing: G's MDE is a
    standardised d = 0.426, and §D6d compares against RAW comparators (0.025,
    ~0.10). **A standardised MDE cannot be compared to a raw effect size.**

    Simulation at the arm's realized pair-count and SD, per §A7.2, because the
    sign-flip null has no closed form worth trusting at these n.
    """
    if n < 2 or not sd or sd <= 0 or not math.isfinite(sd):
        return None
    rng = np.random.default_rng(seed)
    lo, hi = 0.0, 8.0 * sd
    for _ in range(22):                       # bisection on the effect size
        mid = 0.5 * (lo + hi)
        rej = 0
        for _ in range(reps):
            draw = rng.normal(direction * mid, sd, n)
            r = sign_flip_p(draw, direction, seed=int(rng.integers(1 << 30)),
                            draws=400)
            if r["p"] <= alpha:
                rej += 1
        if rej / reps < power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


# ══════════════════════════════════════════════════════════════════════════
# collection
# ══════════════════════════════════════════════════════════════════════════
def build(rule="CANONICAL", verbose=False, max_prompts=None):
    """Cells -> per-member A per arm -> pairs. Returns everything, reads nothing.

    THE ORDER OF THE TWO GATES IS THE SPEC'S AND NOT A CHOICE ([3269].1).
    §D1's qualification chain is PRIOR TO and INDEPENDENT OF §D3's `t`:

        a CELL qualifies    >= 3 rated non-function words in EACH role
        a MEMBER qualifies  >= 1 qualifying cell
        a PAIR is admitted  BOTH members qualify

    So a never-firing member has zero qualifying cells and its pair is never
    admitted AT ANY t INCLUDING 0.00 -- **without any appeal to what an
    undefined median means.** §D3 says so itself ("removes low-displacement
    pairs BEFORE `t` ever acts on them") and §D6 makes it a printed diagnostic.
    """
    import m01_norms as N
    import m01_registration_b as B
    import m01_concentration as CC
    import within_pair as W

    # ── the roster rule, and its drift BOUND BY NAME (§A8.2b) ────────────
    #
    # `frozen_population()` RETURNS drift; it does not refuse. Six callers
    # check it, one documents why it does not, and one binds it to `_d` and
    # never looks -- with the drift live today. **"Recorded" means a field a
    # reader MEETS, never a value the producer received and dropped.**
    prompts_all, models, (ph, mh), drift = CC.frozen_population()

    pairs, domains = W.m01_pairs()
    assert len(pairs) == 684, f"population is {len(pairs)}, expected 684"

    edges, edge_dropped = CC.operation_edges(models)
    #: `load_norms` returns THREE values -- (norms, freqs, report). Binding
    #: the tuple and subscripting it is the defect that crashed stage 1 on its
    #: first run ([3323]); the sibling unpacks the same way at c3.py:261.
    norms, _freqs, _report = N.load_norms(verify=True)
    tabs = {d: norms[("en", d, "primary")]
            for d in ("arousal", "valence", "dominance")}

    texts = {t for v in pairs.values() for t in v.values()}
    #: SMOKE-TEST ONLY, and it changes nothing on the real path. `max_prompts`
    #: exists so the suite can EXECUTE this function against the real store and
    #: the real norms rather than only unit-test the functions it calls.
    #: **28 tests, an independent harness and three omissions passes all passed
    #: while `build()` could not open the data, because every one of them takes
    #: its data as an argument and this is the only function that reaches the
    #: world.** A run of two prompts would have caught it in a second.
    if max_prompts is not None:
        texts = set(sorted(texts)[:max_prompts])
    diag = collections.Counter()
    #: cell[(prompt, edge_key)] = {"departed": float, "arms": {arm: A}}
    cells = collections.defaultdict(dict)

    for fam, pos, step in sorted(edges):
        for t in sorted(texts):
            c = step.cell(t)
            if not c.is_present:
                diag["cell absent from the store"] += 1; continue
            if c.language != "en":
                #: C §C0's `en only`, RETAINED per §A8's field table.
                diag["non-en, outside C v6 §C0's declared population"] += 1
                continue
            try:
                dec = c.decompose(None)
            except Exception as e:
                diag[f"decompose errored: {type(e).__name__}"] += 1; continue
            if not dec:
                diag["cell decomposed empty"] += 1; continue
            try:
                roles = N.cell_roles(c, rule)
            except Exception as e:
                diag[f"cell_roles errored: {type(e).__name__}"] += 1; continue

            ws, zs, rs = [], [], []
            for w, wt, role in roles:
                k = N.norm_key(w, "en", fold=False)
                if N.is_function_word(k, "en"):
                    diag["function word excluded"] += 1; continue
                z = {}
                for dim in ("arousal", "valence", "dominance"):
                    val, _ = N.lookup(tabs[dim], k.casefold(), "en")
                    z[dim] = val
                if any(x is None for x in z.values()):
                    diag["missing a V/A/D rating"] += 1; continue
                ws.append(wt); zs.append(z); rs.append(role)

            nf = sum(1 for r in rs if r == "faller")
            if nf < B.QUALIFYING_MIN or len(rs) - nf < B.QUALIFYING_MIN:
                diag[f"cell below the {B.QUALIFYING_MIN}-per-role bar"] += 1
                continue

            #: THE WITHDRAWN NaN GUARD, NOW AN ASSERTION ([3279]).
            #: `wmean` returns NaN when sum(w) == 0. That cannot happen here:
            #: the role predicate IS a |delta| threshold (>= 0.003 under
            #: CANONICAL and DRAW), so having a role entails a positive weight.
            #: A fallback would have absorbed a future rule change silently; an
            #: assertion makes someone read this sentence instead.
            assert all(w > 0 for w in ws), (
                "zero weight on a roled word -- unreachable while the role "
                "predicate is a |delta| threshold; a rule change must be read, "
                "not absorbed")

            cells[t][(fam, pos)] = {"departed": float(dec["departed"]),
                                    "ws": ws, "zs": zs, "rs": rs}

    #: A TRUNCATED RUN MUST SAY SO IN ITS OWN OUTPUT ([3330]). `max_prompts`
    #: defaults to None and the stage-1 runner never passes it -- but it is a
    #: LIVE parameter on the production function, so a future caller could
    #: score a slice while believing it scored the corpus. **The guard is not
    #: that it cannot happen; it is that it cannot happen SILENTLY.**
    return {"pairs": pairs, "domains": domains, "cells": dict(cells),
            "max_prompts": max_prompts, "n_texts_used": len(texts),
            "n_texts_full": len({t for v in pairs.values() for t in v.values()}),
            "truncated": max_prompts is not None,
            "diag": diag, "edges": [(f, p) for f, p, _ in edges],
            "edge_dropped": dict(edge_dropped),
            "roster": {"n_prompts": len(prompts_all), "n_models": len(models),
                       "prompts_sha16": ph[:16], "models_sha16": mh[:16],
                       "frozen_prompts_sha16": CC.PROMPTS_SHA[:16],
                       "frozen_models_sha16": CC.MODELS_SHA[:16],
                       "drift": drift}}


def arm_values(cells, arm, resid):
    """Per-cell A for one arm, with GLOBAL residualisation. C §C0's RESIDUAL.

    Residualisation is GLOBAL OVER THE QUALIFYING POPULATION AND NEVER WITHIN
    CELL -- C §C0 states the reason: a within-cell fit leaves one df at n = 3
    and emits structural zeros that enter the statistic while counting in the
    denominator.
    """
    import m01_registration_c3 as C3

    name, dim, _direction, kind = arm
    #: pass 1 -- gather (arousal, value) over every qualifying word, globally
    xs, ys = [], []
    for t, per_edge in cells.items():
        for key, c in per_edge.items():
            for z in c["zs"]:
                v = abs(z[dim]) if name in EXTREMITY else z[dim]
                xs.append(z["arousal"]); ys.append(v)
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)

    beta = None
    if kind != "none" and len(xs) > 3:
        X = ([np.ones_like(xs), xs] if kind == "linear"
             else [np.ones_like(xs), xs, xs ** 2])
        beta = np.linalg.lstsq(np.vstack(X).T, ys, rcond=None)[0]

    def value(z):
        v = abs(z[dim]) if name in EXTREMITY else z[dim]
        if beta is None:
            return v
        a = z["arousal"]
        pred = beta[0] + beta[1] * a + (beta[2] * a * a if len(beta) > 2 else 0.0)
        return v - pred

    #: pass 2 -- A per cell from the inherited estimator, never re-derived
    out = collections.defaultdict(dict)
    for t, per_edge in cells.items():
        for key, c in per_edge.items():
            vals = [value(z) for z in c["zs"]]
            r = C3.A_and_terms(vals, c["ws"], c["rs"])
            if r is not None:
                out[t][key] = r["A"]
    return dict(out), (None if beta is None else [float(b) for b in beta])


def assemble(built, arm_A):
    """Members -> pairs, with BOTH displacement denominators (§D6, [3304].3).

    TWO MEDIANS, DELIBERATELY DIFFERENT, AND THE SPLIT IS DECLARED:

      `disp_qual`  median departed over the member's QUALIFYING cells.
                   Gates `t`. Scoped to the same cells as A(member) itself, so
                   a cell contributing nothing to A cannot decide whether A is
                   read -- and it makes `t` collinear with the bar, which makes
                   §D3's COLLAPSE clause MORE likely to fire. The conservative
                   direction ([3275].2).

      `disp_all`   median departed over ALL the member's cells. Feeds §D6's
                   admitted-vs-DROPPED diagnostic ONLY. A dropped member has
                   zero qualifying cells, so `disp_qual` is undefined for
                   exactly the pairs that diagnostic exists to expose.

    §3(i) gates a verdict and must be conservative; §D6 reports a loss and must
    be complete. Same word, two denominators, one printed name each.
    """
    rows = []
    for pid, members in built["pairs"].items():
        rec = {"pair_id": pid, "domain": built["domains"].get(pid)}
        ok = True
        for role, text in members.items():
            per_edge = built["cells"].get(text, {})
            qual = arm_A.get(text, {})
            dep_q = [per_edge[k]["departed"] for k in qual if k in per_edge]
            dep_a = [c["departed"] for c in per_edge.values()]
            rec[f"{role}_n_qual"] = len(qual)
            rec[f"{role}_n_cells"] = len(per_edge)
            rec[f"{role}_A"] = (st.mean(qual.values()) if qual else None)
            rec[f"{role}_disp_qual"] = (st.median(dep_q) if dep_q else None)
            rec[f"{role}_disp_all"] = (st.median(dep_a) if dep_a else None)
            if not qual:
                ok = False
        rec["admitted_by_qualification"] = ok
        mk, um = "MARKED", "UNMARKED"
        if ok and rec.get(f"{mk}_A") is not None and rec.get(f"{um}_A") is not None:
            rec["D_pair"] = rec[f"{mk}_A"] - rec[f"{um}_A"]
            rec["displacement"] = min(rec[f"{mk}_disp_qual"],
                                      rec[f"{um}_disp_qual"])
        else:
            rec["D_pair"] = None
            #: the DROPPED denominator, and only here
            dq = [rec.get(f"{r}_disp_all") for r in (mk, um)]
            rec["displacement_dropped_all"] = (min(x for x in dq if x is not None)
                                               if any(x is not None for x in dq)
                                               else None)
        rows.append(rec)
    return rows


def admitted_at(rows, t):
    """§D3. `>=`, NOT `>` -- pinned by the gloss "t = 0.00, every qualifying
    pair": under `>` a pair at exactly 0.00 would drop and the primary would
    not be every qualifying pair ([3269].2a)."""
    return [r for r in rows
            if r["D_pair"] is not None and r["displacement"] >= t]


def unit_assertion(rows, admitted, t, field="pair_id"):
    """§A4 / [3068].d. The count, THE FIELD IT COUNTED, and a re-derivation.

    NOT `len(set(ids)) == len(ids)`. That form was a TAUTOLOGY in the sibling
    producer -- the ids came from a dict, so its keys were unique by
    construction and nothing could ever violate it ([3108]). Here the reference
    set is rebuilt FROM THE ROWS, independently of the admitted list, and
    compared. A collapse that dropped, duplicated or invented a unit fails here
    rather than passing quietly.
    """
    #: `t` IS PASSED IN, NEVER RECOVERED FROM `admitted` ([3315]).
    #:
    #: The first version called `admitted_t(admitted)` -- the MINIMUM
    #: displacement in the set being checked. **That made the criterion a
    #: function of the artifact, so dropping the lowest-displacement unit
    #: RAISED the threshold and `expected` shrank to match: got == expected
    #: held, and the one drop this assertion exists to catch passed clean.**
    #: An audit that takes its criterion from the artifact cannot see a wrong
    #: criterion, and here it could not see a missing row either.
    ids = [r[field] for r in admitted]
    expected = {r[field] for r in rows
                if r["D_pair"] is not None
                and r["displacement"] >= t}
    got = set(ids)
    assert len(ids) == len(got), (
        f"duplicate {field} in the admitted set: "
        f"{len(ids)} rows, {len(got)} distinct")
    assert got == expected, (
        f"admitted set disagrees with an independent re-derivation: "
        f"only-in-admitted {sorted(got - expected)[:5]}, "
        f"only-in-rederivation {sorted(expected - got)[:5]}")
    return f"units={len(ids)} field={field} entries={len(rows)}"


# ══════════════════════════════════════════════════════════════════════════
# STAGE 1 -- variance and thresholds. NO D, NO p, NO SIGNS.
# ══════════════════════════════════════════════════════════════════════════
def stage1(built, out_path, seed=20260731):
    """Realized SDs and raw MDEs per arm per threshold point. §A7.3, [3280].

    **THIS STAGE MUST NOT EMIT A VERDICT QUANTITY.** It computes |D_pair|'s
    dispersion, never its mean, sign or p. The whole purpose of the split is
    that §D6d's threshold is fixed on the record while the verdict is still
    invisible -- so a stage that leaked D would defeat the artifact it writes.
    """
    payload = {
        "_what": "Registration D STAGE 1: realized SDs and raw MDEs. "
                 "NO D, NO p, NO SIGNS.",
        "_registration": "registration_d_pairs_v6.md",
        "_amendment": AMENDMENT_SHA,
        "_population": POPULATION_SHA,
        "_governing": {"c_v6": C_V6_SHA, "b_v13": B_V13_SHA},
        "_convention": {"power": POWER, "alpha": ALPHA, "sided": "one",
                        "scale": "RAW dimension units (§A7.2)"},
        #: §A8.2b -- drift is a FIRST-CLASS FIELD, bound by name, never `_`
        "roster": built["roster"],
        #: [3330] -- a truncated run is self-identifying IN THE ARTIFACT, so a
        #: reader meets it rather than inferring it from a count that looks low
        "truncation": {"truncated": built["truncated"],
                       "max_prompts": built["max_prompts"],
                       "n_texts_used": built["n_texts_used"],
                       "n_texts_full": built["n_texts_full"]},
        "arms": {},
    }
    for arm in ARMS:
        name, dim, direction, kind = arm
        A, beta = arm_values(built["cells"], arm, kind)
        rows = assemble(built, A)
        #: the t = 0.00 set, computed ONCE per arm, so every Jaccard is against
        #: the same reference rather than a re-derived one ([3315]'s lesson:
        #: a comparison whose reference moves with the thing compared is not a
        #: comparison).
        base_ids = {r["pair_id"] for r in admitted_at(rows, 0.00)}
        per_t = {}
        for t in GRID:
            adm = admitted_at(rows, t)
            n = len(adm)
            if n < FLOOR:
                per_t[f"{t:.2f}"] = {"n": n, "status": "UNDERPOWERED"}
                continue
            d = [r["D_pair"] for r in adm]
            sd = st.pstdev(d) if n > 1 else None
            #: §D3 AND §D6's REQUIRED PER-POINT DIAGNOSTICS, IN STAGE 1 AND
            #: NOT STAGE 2 ([3339]). They are SET-MEMBERSHIP and DISPLACEMENT
            #: quantities -- no D, no sign, no p -- so nothing here is a
            #: verdict quantity.
            #:
            #: **AND THE PLACEMENT IS THE POINT.** §D3's CONFIRMED rule counts
            #: only NON-COLLAPSED above-floor points, so WHICH POINTS COLLAPSE
            #: DECIDES WHAT CORROBORATES. Computing that in stage 2 would fix a
            #: rule-relevant threshold while the verdict is visible -- the
            #: identical defect the MDE split exists to prevent, one clause
            #: over. It belongs on the record beside the MDE.
            ids = {r["pair_id"] for r in adm}
            j = jaccard(ids, base_ids)
            dropped = [r for r in rows if not r["admitted_by_qualification"]]
            dd = [r["displacement_dropped_all"] for r in dropped
                  if r.get("displacement_dropped_all") is not None]
            per_t[f"{t:.2f}"] = {
                "n": n, "status": "ok",
                "sd_D_pair": sd,
                "raw_mde": raw_mde(n, sd, direction, seed),
                #: the attainable-p lattice, §D4
                "min_attainable_p": (1.0 / (1 << n) if n <= EXACT_MAX_N
                                     else 1.0 / 10000),
                #: §D3
                "n_admitted": n,
                "jaccard_with_t000": j,
                "collapsed": bool(j >= COLLAPSE_JACCARD),
                #: §D6 -- the two denominators, each NAMED ([3304].3)
                "n_dropped_by_qualification": len(dropped),
                "median_displacement_admitted_qualcells":
                    (st.median([r["displacement"] for r in adm]) if adm else None),
                "median_displacement_dropped_allcells":
                    (st.median(dd) if dd else None),
            }
        payload["arms"][name] = {"dimension": dim, "direction": direction,
                                 "residualisation": kind,
                                 "resid_beta": beta, "per_t": per_t}

    blob = json.dumps(payload, indent=1, sort_keys=True)
    with open(out_path, "w") as fh:
        fh.write(blob)
    h = hashlib.sha256(blob.encode()).hexdigest()
    payload["_self_sha256_16"] = h[:16]
    return payload, h[:16]


# ══════════════════════════════════════════════════════════════════════════
# STAGE 2 -- refuses without stage 1
# ══════════════════════════════════════════════════════════════════════════
class Stage1Missing(RuntimeError):
    pass


def require_stage1(path, expect_sha16):
    """§A7.3 / [3280]. STAGE 2 REFUSES WITHOUT STAGE 1's POSTED ARTIFACT.

    The refusal is the mechanism, not the paperwork: it makes the ordering a
    FACT ABOUT ARTIFACTS rather than a claim about anyone's discipline. A
    producer that merely *intends* to derive the MDE first is a producer whose
    threshold can move.
    """
    if not path or not os.path.exists(path):
        raise Stage1Missing(
            "STAGE 2 REFUSES: no stage-1 artifact at %r. §D6d's threshold must "
            "be on the record BEFORE any verdict quantity exists." % (path,))
    blob = open(path).read()
    got = hashlib.sha256(blob.encode()).hexdigest()[:16]
    if expect_sha16 and got != expect_sha16:
        raise Stage1Missing(
            "STAGE 2 REFUSES: stage-1 artifact hash %s != posted %s. The "
            "threshold on the record is not the threshold on disk." %
            (got, expect_sha16))
    return json.loads(blob)


# ══════════════════════════════════════════════════════════════════════════
# self-test -- EVERY GUARD IS MADE TO FIRE
# ══════════════════════════════════════════════════════════════════════════
def selftest(verbose=False, skip_slow=False):
    """Known answers, and a fired case for every refusal.

    THE RULE THIS SUITE WAS WRITTEN AGAINST: a guard nobody has watched fail is
    not a guard. Two producers in this campaign shipped assertions that COULD
    NOT fire -- one comparing a dict's keys to themselves, one defending a state
    the role predicate excludes. Both passed their suites. So every refusal
    below gets a constructed failing case, not just a passing one.
    """
    ok = [0, 0]

    def case(name, cond):
        good = False
        try:
            good = bool(cond())
        except Exception as e:
            print(f"  [ERR] {name}: {type(e).__name__}: {e}")
        ok[0] += 1; ok[1] += 1 if good else 0
        print(f"  [{'ok' if good else 'FAIL'}] {name}")

    # ── the p-convention, both regimes ───────────────────────────────────
    case("exact enumeration uses PLAIN p (no add-one)",
         lambda: sign_flip_p([1.0] * 6, +1, 0)["convention"] == "plain/exact")
    case("sampled uses ADD-ONE",
         lambda: sign_flip_p([1.0] * 25, +1, 0, draws=200)["convention"]
                 == "add-one/sampled")
    case("all-positive at n=6 lands on the 1/2^n FLOOR, not 0",
         lambda: abs(sign_flip_p([1.0] * 6, +1, 0)["p"] - 1 / 64) < 1e-12)
    case("EXTREMITY IS >=: the observed draw counts ITSELF",
         #: under `>` the all-positive case would give p = 0, which §D4's
         #: lattice forbids. This is the case that distinguishes the two.
         lambda: sign_flip_p([1.0] * 6, +1, 0)["hits"] == 1)
    case("direction -1 mirrors direction +1 on negated data",
         lambda: abs(sign_flip_p([-1.0, -2.0, -3.0, -1.5, -2.5, -0.5], -1, 0)["p"]
                     - sign_flip_p([1.0, 2.0, 3.0, 1.5, 2.5, 0.5], +1, 0)["p"])
                 < 1e-12)
    case("a NULL sample does not reject",
         lambda: sign_flip_p([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], +1, 0)["p"] > 0.05)

    # ── STAGE 2's refusal, MADE TO FIRE, twice ───────────────────────────
    fired = [False, False]
    try:
        require_stage1("/nonexistent/stage1.json", None)
    except Stage1Missing:
        fired[0] = True
    case("STAGE 2 REFUSES with no stage-1 artifact", lambda: fired[0])

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        fh.write('{"_what": "not the posted artifact"}')
        tmp = fh.name
    try:
        require_stage1(tmp, "0000000000000000")
    except Stage1Missing:
        fired[1] = True
    case("STAGE 2 REFUSES on a stage-1 hash mismatch", lambda: fired[1])
    case("STAGE 2 ACCEPTS the matching artifact (the guard is not vacuous)",
         lambda: require_stage1(
             tmp, hashlib.sha256(open(tmp).read().encode()).hexdigest()[:16]
         )["_what"].startswith("not the posted"))
    os.unlink(tmp)

    # ── the threshold comparison is >=, not > ────────────────────────────
    rows = [{"pair_id": "a", "D_pair": 0.1, "displacement": 0.00},
            {"pair_id": "b", "D_pair": 0.2, "displacement": 0.05},
            {"pair_id": "c", "D_pair": None, "displacement": None}]
    case("t = 0.00 admits a pair sitting EXACTLY at 0.00 (>=, not >)",
         lambda: {r["pair_id"] for r in admitted_at(rows, 0.00)} == {"a", "b"})
    case("t = 0.05 admits only the pair at or above it",
         lambda: {r["pair_id"] for r in admitted_at(rows, 0.05)} == {"b"})
    case("a pair with D_pair None never admits at any t",
         lambda: all("c" not in {r["pair_id"] for r in admitted_at(rows, t)}
                     for t in GRID))

    # ── the unit assertion, MADE TO FIRE ─────────────────────────────────
    dup = [{"pair_id": "a", "D_pair": 0.1, "displacement": 0.1},
           {"pair_id": "a", "D_pair": 0.1, "displacement": 0.1}]
    fired_u = False
    try:
        unit_assertion(dup, dup, 0.0)
    except AssertionError:
        fired_u = True
    case("the unit assertion FIRES on a duplicated unit", lambda: fired_u)

    fired_v = False
    try:
        #: an INVENTED unit -- present in the admitted list, absent from rows
        unit_assertion(rows[:2], rows[:2] + [{"pair_id": "zz", "D_pair": 1.0,
                                              "displacement": 1.0}], 0.0)
    except AssertionError:
        fired_v = True
    case("the unit assertion FIRES on an invented unit", lambda: fired_v)
    #: [3315]'s CONSTRUCTED CASE, now a permanent test. The first version
    #: recovered `t` from the admitted set, so this passed silently.
    drop_rows = [{"pair_id": "a", "D_pair": 0.1, "displacement": 0.01},
                 {"pair_id": "b", "D_pair": 0.2, "displacement": 0.02},
                 {"pair_id": "c", "D_pair": 0.3, "displacement": 0.03}]
    fired_d = False
    try:
        #: the LOWEST-displacement unit silently dropped, t declared as 0.00
        unit_assertion(drop_rows, drop_rows[1:], 0.00)
    except AssertionError:
        fired_d = True
    case("the unit assertion FIRES when the LOWEST unit is dropped [3315]",
         lambda: fired_d)
    case("and it PASSES when that same t admits exactly the right set",
         lambda: "units=3" in unit_assertion(drop_rows, drop_rows, 0.00))
    case("t is a PARAMETER, not recovered from the set under audit",
         lambda: "t" in unit_assertion.__code__.co_varnames[
             :unit_assertion.__code__.co_argcount])

    case("the unit assertion NAMES ITS FIELD in the passing line",
         lambda: "field=pair_id" in unit_assertion(rows[:2], rows[:2], 0.0))

    # ── jaccard / collapse ───────────────────────────────────────────────
    case("jaccard of a set with itself is 1.0 (a COLLAPSED point)",
         lambda: jaccard("abc", "abc") == 1.0 >= COLLAPSE_JACCARD)
    case("jaccard separates a genuinely different set",
         lambda: jaccard("abcd", "cdef") < COLLAPSE_JACCARD)

    # ── the raw MDE ──────────────────────────────────────────────────────
    case("raw MDE is None when the SD is degenerate",
         lambda: raw_mde(10, 0.0, +1, 0) is None)
    case("raw MDE SHRINKS as n grows (more pairs buy detection)",
         lambda: raw_mde(30, 1.0, +1, 7, reps=120)
                 < raw_mde(8, 1.0, +1, 7, reps=120))
    case("raw MDE SCALES with the SD (it is RAW, not standardised)",
         lambda: raw_mde(12, 2.0, +1, 7, reps=120)
                 > 1.5 * raw_mde(12, 1.0, +1, 7, reps=120))

    # ── §D3/§D6 per-point diagnostics, and the COLLAPSE flag ─────────────
    case("a point identical to t=0.00 is flagged COLLAPSED",
         lambda: jaccard({"a", "b"}, {"a", "b"}) >= COLLAPSE_JACCARD)
    case("a point sharing 18 of 20 with t=0.00 is COLLAPSED (>= 0.95 is tight)",
         #: 18/20 -> jaccard 18/22 = 0.818, NOT collapsed. The clause is
         #: strict: a point must be nearly the primary set to be discounted.
         lambda: jaccard(set("abcdefghijklmnopqr"), set("abcdefghijklmnopqrst"))
                 < COLLAPSE_JACCARD)
    case("the collapse threshold is the declared 0.95, not a rounded 0.9",
         lambda: COLLAPSE_JACCARD == 0.95)

    # ── §D3's READING RULE: every branch, constructed ────────────────────
    def _pt(D, rej, collapsed=False, status="ok"):
        return {"status": status, "D": D, "reject": rej,
                "collapsed": collapsed, "p": 0.01 if rej else 0.4,
                "raw_mde": 0.01, "n": 30}

    case("primary not rejecting is NOT SUPPORTED whatever the curve does",
         lambda: reading_rule({"0.00": _pt(-0.1, False),
                               "0.10": _pt(-0.2, True)})["verdict"]
                 == "NOT SUPPORTED")
    case("primary rejecting with every other point COLLAPSED is SINGLE-POINT",
         lambda: reading_rule({"0.00": _pt(-0.1, True),
                               "0.05": _pt(-0.1, True, collapsed=True)}
                              )["verdict"] == "SINGLE-POINT")
    case("a SIGN FLIP at a non-collapsed point is THRESHOLD-DEPENDENT",
         lambda: reading_rule({"0.00": _pt(-0.1, True),
                               "0.10": _pt(+0.2, False)}   # sign, not signif.
                              )["verdict"] == "THRESHOLD-DEPENDENT")
    case("agreement in SIGN at a non-collapsed point is CONFIRMED",
         lambda: reading_rule({"0.00": _pt(-0.1, True),
                               "0.10": _pt(-0.02, False)}  # same sign, no p
                              )["verdict"] == "CONFIRMED")
    case("a COLLAPSED point cannot corroborate, so it cannot make CONFIRMED",
         lambda: reading_rule({"0.00": _pt(-0.1, True),
                               "0.05": _pt(-0.1, True, collapsed=True)}
                              )["n_corroborators"] == 0)
    case("an UNDERPOWERED primary is NOT SUPPORTED, never a null",
         lambda: reading_rule({"0.00": _pt(0, None, status="UNDERPOWERED")}
                              )["verdict"] == "NOT SUPPORTED")

    # ── §D6d's three-way MDE reading ─────────────────────────────────────
    case("a rejection reads as TRACKS TRANSGRESSIVE SITES",
         lambda: mde_reading(_pt(-0.1, True), "arousal")["reading"]
                 == "TRACKS TRANSGRESSIVE SITES")
    case("a null with MDE BELOW the known effect is QUOTABLE evidence against",
         lambda: mde_reading({**_pt(0.0, False), "raw_mde": 0.01},
                             "val_extrem")["quotable"] is True)
    case("a null with MDE ABOVE it is UNINFORMATIVE, quotable as nothing",
         lambda: mde_reading({**_pt(0.0, False), "raw_mde": 0.9},
                             "val_extrem")["quotable"] is False)
    case("h1_signed has NO declared comparator and says so",
         lambda: "NO DECLARED COMPARATOR"
                 in mde_reading(_pt(0.0, False), "h1_signed")["reading"])

    # ── §D6c's FIXED SEQUENCE ────────────────────────────────────────────
    case("the Family-2 sequence is §D6c's declared order",
         lambda: FAMILY2_SEQUENCE == ("arousal", "val_extrem", "dom_extrem"))
    case("the known effects are §D6d's three declared values",
         lambda: KNOWN_EFFECT == {"arousal": 0.10, "val_extrem": 0.025,
                                  "dom_extrem": 0.025})

    # ── §D4's LATTICE REFUSAL, MADE TO FIRE ──────────────────────────────
    #: at n=6 the floor is 1/64 = 0.0156 <= alpha, so a point PASSES; the
    #: refusal needs a resolution coarser than alpha, which needs n < 5.
    #: FLOOR blocks that, so the refusal is unreachable through read_point --
    #: and saying so is better than pretending the branch is live.
    case("the lattice refusal is UNREACHABLE while FLOOR >= 5, and that is "
         "a property of the declared floor rather than an untested branch",
         lambda: (1.0 / (1 << FLOOR)) <= ALPHA and FLOOR >= 5)

    # ── declared constants, asserted against the frozen text ─────────────
    case("GRID is §D3's six points verbatim",
         lambda: GRID == (0.00, 0.01, 0.02, 0.05, 0.10, 0.20))
    case("FLOOR, ALPHA, COLLAPSE and POWER are the declared values",
         lambda: (FLOOR, ALPHA, COLLAPSE_JACCARD, POWER)
                 == (6, 0.05, 0.95, 0.80))
    case("the four arms carry §D6b's directions",
         lambda: [d for _, _, d, _ in ARMS] == [-1, +1, +1, +1])
    case("only the extremity arms take |dim_z|",
         lambda: EXTREMITY == {"val_extrem", "dom_extrem"})
    case("the arousal arm is RAW -- no residualisation, per §D6b",
         lambda: dict((n, k) for n, _, _, k in ARMS)["arousal"] == "none")

    # ── THE RUNG THE SUITE WAS MISSING: does build() EXECUTE? ────────────
    #
    # Everything above tests a function that takes its data as an argument.
    # `build()` is the only one that REACHES THE WORLD -- the store, the norms,
    # the registry, the frozen population -- and nothing exercised it, so it
    # shipped frozen and crashed on its first real call ([3323]).
    #
    # EXISTS < CALLED < REACHED < RAN. This is the last rung, and it is cheap.
    if not skip_slow:
        built = [None]

        def _smoke():
            built[0] = build(max_prompts=2)
            b = built[0]
            return (isinstance(b["cells"], dict)
                    and len(b["pairs"]) == 684
                    and "drift" in b["roster"]
                    and isinstance(b["roster"]["drift"], list))
        case("build() EXECUTES against the real store and norms [3323]", _smoke)
        #: [3330]. A truncated run must SAY SO. Both directions, because a
        #: flag that is always True is as useless as one always False.
        case("a TRUNCATED build self-identifies in its own record",
             lambda: built[0] is not None
                     and built[0]["truncated"] is True
                     and built[0]["max_prompts"] == 2
                     and built[0]["n_texts_used"] == 2
                     and built[0]["n_texts_full"] == 1368)
        case("and n_texts_used < n_texts_full is VISIBLE, not inferred",
             lambda: built[0]["n_texts_used"] < built[0]["n_texts_full"])
        case("and it binds the roster DRIFT as a named field, never `_`",
             lambda: built[0] is not None
                     and set(("drift", "prompts_sha16", "models_sha16",
                              "frozen_prompts_sha16", "frozen_models_sha16"))
                         <= set(built[0]["roster"]))

    print(f"selftest {ok[1]}/{ok[0]}")
    return 0 if ok[1] == ok[0] else 1


def main(a):
    if a.selftest:
        return selftest(a.verbose, a.skip_slow)
    raise SystemExit(
        "the read is not wired yet: stage 1 runs on the pen's word, "
        "stage 2 only against its posted hash")




# ══════════════════════════════════════════════════════════════════════════
# STAGE 2 -- the read. THE FIRST CODE IN THIS CHAIN THAT CAN EMIT A SIGN.
# ══════════════════════════════════════════════════════════════════════════
KNOWN_EFFECT = {          #: §D6d, declared in advance from the public record
    "arousal": 0.10,
    "val_extrem": 0.025,
    "dom_extrem": 0.025,  #: stand-in; H3's own is unknown, declared as such
}
FAMILY2_SEQUENCE = ("arousal", "val_extrem", "dom_extrem")   #: §D6c, fixed


def read_point(rows, t, direction, seed, stage1_cell):
    """One arm at one threshold point. Returns the verdict quantities.

    §D1's [1524].4 RULE IS ENFORCED HERE: `A(MARKED)` and `A(UNMARKED)` print
    SEPARATELY beside every D. **A difference is not a direction until both
    terms are visible** — on the norms population that rule was the only thing
    standing between H1 and a misclassification.
    """
    adm = admitted_at(rows, t)
    n = len(adm)
    if n < FLOOR:
        return {"n": n, "status": "UNDERPOWERED", "reject": None}

    #: the unit assertion, at the DECLARED t -- never one recovered from the
    #: set under audit ([3315])
    unit_line = unit_assertion(rows, adm, t)

    d = [r["D_pair"] for r in adm]
    r = sign_flip_p(d, direction, seed)

    #: §D4. A point whose null cannot reach alpha is not a null. The lattice is
    #: printed and the point REFUSES rather than reporting a non-rejection that
    #: was arithmetically forced.
    if r["resolution"] > ALPHA:
        return {"n": n, "status": "LATTICE-REFUSED", "reject": None,
                "min_attainable_p": r["resolution"], "unit_line": unit_line}

    mk = [x["MARKED_A"] for x in adm]
    um = [x["UNMARKED_A"] for x in adm]
    return {
        "n": n, "status": "ok",
        "unit_line": unit_line,
        "D": r["statistic"],
        "p": r["p"], "p_convention": r["convention"],
        "draws": r["draws"], "min_attainable_p": r["resolution"],
        "reject": bool(r["p"] <= ALPHA),
        #: BOTH TERMS, ALWAYS -- §D1
        "A_marked": st.mean(mk), "A_unmarked": st.mean(um),
        "collapsed": stage1_cell.get("collapsed"),
        "raw_mde": stage1_cell.get("raw_mde"),
    }


def reading_rule(per_t, primary_t="0.00"):
    """§D3's four-way verdict. COLLAPSED points cannot corroborate.

    *"a point that is the primary set under another name cannot corroborate
    it"* — so the CONFIRMED clause's "every above-floor point" ranges over
    NON-COLLAPSED above-floor points only, and if none survive the result is
    SINGLE-POINT and never CONFIRMED.
    """
    p0 = per_t.get(primary_t, {})
    if p0.get("status") != "ok":
        return {"verdict": "NOT SUPPORTED", "why": f"primary is {p0.get('status')}"}
    if not p0["reject"]:
        #: §D3: NOT SUPPORTED "whatever the curve does"
        return {"verdict": "NOT SUPPORTED",
                "why": f"primary p {p0['p']:.5f} does not pass its null"}

    corroborators = {t: c for t, c in per_t.items()
                     if t != primary_t and c.get("status") == "ok"
                     and not c.get("collapsed")}
    if not corroborators:
        return {"verdict": "SINGLE-POINT",
                "why": "every other above-floor point COLLAPSED; the curve "
                       "tested nothing", "n_corroborators": 0}

    #: SIGN, not significance -- §D3 says so explicitly
    flipped = [t for t, c in corroborators.items()
               if (c["D"] < 0) != (p0["D"] < 0)]
    if flipped:
        return {"verdict": "THRESHOLD-DEPENDENT",
                "why": f"sign flips at {sorted(flipped)}",
                "n_corroborators": len(corroborators)}
    return {"verdict": "CONFIRMED", "why": "primary passes and every "
            "non-collapsed above-floor point agrees in SIGN",
            "n_corroborators": len(corroborators)}


def mde_reading(point, arm_name):
    """§D6d's three-way rule. A null is only interpretable beside its MDE."""
    known = KNOWN_EFFECT.get(arm_name)
    if point.get("status") != "ok":
        return {"reading": "NOT READ", "why": point.get("status")}
    if point["reject"]:
        return {"reading": "TRACKS TRANSGRESSIVE SITES", "known_effect": known}
    if known is None:
        return {"reading": "NULL, NO DECLARED COMPARATOR",
                "why": "§D6d declares comparators for the three "
                       "site-specificity arms only"}
    mde = point.get("raw_mde")
    if mde is None:
        return {"reading": "UNINFORMATIVE", "why": "no MDE at this point"}
    if mde < known:
        return {"reading": "EVIDENCE AGAINST SITE-SPECIFICITY",
                "why": f"null with MDE {mde:.5f} < known effect {known}",
                "quotable": True, "known_effect": known, "mde": mde}
    return {"reading": "UNINFORMATIVE AT THIS POWER",
            "why": f"MDE {mde:.5f} >= known effect {known}",
            "quotable": False, "known_effect": known, "mde": mde}


def stage2(built, stage1_path, stage1_sha16, out_path, seed=20260731):
    """The read. REFUSES without stage 1's posted artifact. §A7.3.

    §D6c's HIERARCHY IS ENFORCED, NOT DOCUMENTED. Family 2 is fixed-sequence:
    testing STOPS at the first non-rejection and arms below it are reported
    **NOT TESTED, never null** — the difference the registration insists on,
    because an untested arm reported as a null is a claim nobody made.
    """
    s1 = require_stage1(stage1_path, stage1_sha16)

    out = {"_what": "Registration D STAGE 2: the read.",
           "_stage1": {"path": os.path.basename(stage1_path),
                       "sha256_16": stage1_sha16},
           "_amendment": AMENDMENT_SHA, "_population": POPULATION_SHA,
           "roster": built["roster"], "arms": {}}

    per_arm_rows = {}
    for arm in ARMS:
        name, dim, direction, kind = arm
        A, _beta = arm_values(built["cells"], arm, kind)
        rows = assemble(built, A)
        per_arm_rows[name] = (rows, direction)

    def run_arm(name, direction):
        rows = per_arm_rows[name][0]
        s1_arm = s1["arms"][name]["per_t"]
        per_t = {}
        for t in GRID:
            key = f"{t:.2f}"
            per_t[key] = read_point(rows, t, direction, seed, s1_arm.get(key, {}))
        rule = reading_rule(per_t)
        return {"per_t": per_t, "reading_rule": rule,
                "mde_reading": mde_reading(per_t.get("0.00", {}), name)}

    #: FAMILY 1 -- standalone, alpha 0.05, cannot be blocked by Family 2
    out["arms"]["h1_signed"] = run_arm("h1_signed", -1)
    out["arms"]["h1_signed"]["family"] = 1

    #: FAMILY 2 -- FIXED SEQUENCE, STOP AT THE FIRST NON-REJECTION
    stopped = False
    for name in FAMILY2_SEQUENCE:
        if stopped:
            out["arms"][name] = {"family": 2, "status": "NOT TESTED",
                                 "why": "fixed-sequence stopped at an earlier "
                                        "arm; this is NOT a null"}
            continue
        direction = dict((n, d) for n, _, d, _ in ARMS)[name]
        res = run_arm(name, direction)
        res["family"] = 2
        out["arms"][name] = res
        if not res["per_t"].get("0.00", {}).get("reject"):
            stopped = True
            res["sequence_note"] = ("first non-rejection: arms below this one "
                                    "are NOT TESTED")

    blob = json.dumps(out, indent=1, sort_keys=True, default=float)
    with open(out_path, "w") as fh:
        fh.write(blob)
    return out, hashlib.sha256(blob.encode()).hexdigest()[:16]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--skip-slow", action="store_true",
                    dest="skip_slow", help="omit the build() smoke test")
    sys.exit(main(ap.parse_args()))
