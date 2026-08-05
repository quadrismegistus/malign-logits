"""q_dispersion_pass.py — Q's THREE PAIRED DISPERSIONS, ONE FILE. Pen-authored
([4443].1's form: one pass, three (sd, k) pairs, provably identical code).

EGRESS CONTRACT ([4323]'s criterion, [4333]'s dedupe discipline, [4443]'s
one-pass form): exactly THREE (sd, k) pairs leave — H1 tail_excess, H5
departed, H6 A_|valence| — plus a counts-verified line naming no number not
already public, plus H1's line, which prints ONLY the published constants it
was checked against. Every mean of paired differences is necessarily formed
inside its sd and NOTHING derived from any of them egresses: no sum, sign
count, median, quantile, extremum, t, p, CI, per-stem or per-key value, no
direction-keyed output, no intermediate file, no flag or branch that could
compute an undeclared rule.

THE FREE KNOWN ANSWER ([4445]): H1's pair was PUBLISHED by q_h1_sd_pass —
sd 0.014765, k 684, computed from N's artifact. This pass recomputes it
through the canonical machinery (Cell.decompose, the §Q3 pairing rule). If
its H1 output differs from the published pair it REFUSES — a blind
instrument self-verified through the same code path that produces the two
unpublished numbers.

THE DECLARED RULE ONLY (§Q3, frozen; [4321]/[4333]): the key is
(stem, base, aligned); FAMILY LABELS COLLAPSE (tulu / tulu-no-safety are one
edge — steps deduped by transition); strict both-sides pairing; d_i per stem
is the mean over its surviving keys of (MARKED − UNMARKED); sd over the d_i,
ddof=1. H1/H5 run on both-sides ANALYSED keys (clause 7: analysed = present,
decomposable, >= 1 faller). H6 runs on both-sides A-CELL keys (clause 6:
>= 3 qualifying words per role, both dimensions rated, function words out —
the qualifying logic is q_h6_denominator_pass.py's, cleared at [4447]).

A_|valence| ( §Q2 ): wmean(|z_valence| of qualifying fallers) −
wmean(|z_valence| of qualifying risers), weights |delta| — pairs_d.py's
construction, C's wmean. Formed per cell, differenced per key, meaned per
stem, and only the DISPERSION of those stem means ever leaves.

Refusals name COUNTS and measure names only; every expected count below is
public ([4319]/[4321]/[4449]/[4450]).
"""
import collections
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# run from the malign-logits repo root; the repo's own path bootstrap:
ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "meta", "M01_displacement", "scripts"))

PAIRS = os.path.join(ROOT, "meta", "M01_displacement", "results",
                     "population_d_684.json")

SENTINEL = re.compile(r"^<<<.*>>>$")
CJK = re.compile(r"[一-鿿]")

# Public known answers; any mismatch is a refusal.
EXPECT_STEMS = 684
EXPECT_PAIR_TEXTS = 1368
EXPECT_EDGES = 43                    # distinct (base, aligned) transitions
EXPECT_BOTHSIDES_ANALYSED = 24606    # [4319], reproduced [4449]
EXPECT_ONESIDED_ANALYSED = 1253
EXPECT_BOTHSIDES_A = 15152           # [4449]
EXPECT_STEMS_H6 = 626                # [4449]
# The published H1 pair this pass must reproduce or refuse ([4337]):
EXPECT_H1_SD = 0.014765              # to 6 decimals
EXPECT_H1_K = 684


def refuse(msg):
    raise SystemExit("REFUSING: %s" % msg)


def wmean(vals, wts):
    s = sum(wts)
    return sum(v * w for v, w in zip(vals, wts)) / s if s > 0 else None


def main():
    from malign_logits.movement import CANONICAL
    import m01_concentration as CC
    import m01_norms as N
    import m01_registration_b as B
    from malign_logits.prompts import Prompts

    byid = {str(p.id): p for p in Prompts().all()}
    stems = json.load(open(PAIRS))["ids"]
    if len(stems) != EXPECT_STEMS:
        refuse("population carries %d stems, not %d" % (len(stems), EXPECT_STEMS))

    pair = {}
    for s in stems:
        m_, u_ = byid.get(s + "_M"), byid.get(s + "_U")
        if m_ is None or u_ is None:
            continue
        t_m, t_u = m_.text, u_.text
        if SENTINEL.match(t_m) or CJK.search(t_m):
            continue
        if SENTINEL.match(t_u) or CJK.search(t_u):
            continue
        pair[s] = (t_m, t_u)
    if 2 * len(pair) != EXPECT_PAIR_TEXTS:
        refuse("pair map covers %d texts, not %d" % (2 * len(pair), EXPECT_PAIR_TEXTS))

    _p, mods, _h, _d = CC.frozen_population()
    edges_raw, _drop = CC.operation_edges(mods)

    def mid(o):
        return getattr(o, "id", None) or getattr(o, "model_id", None) or str(o)

    # §Q3: family labels collapse — one Step per distinct (base, aligned).
    steps = {}
    for _fam, _pos, step in edges_raw:
        steps.setdefault((mid(step.pre), mid(step.post)), step)
    if len(steps) != EXPECT_EDGES:
        refuse("%d distinct transitions, not %d" % (len(steps), EXPECT_EDGES))

    norms, _f, _r = N.load_norms(verify=True)
    tabs = {d: norms[("en", d, "primary")] for d in ("arousal", "valence")}

    def measures(step, text):
        """-> (analysed?, tail_excess, departed, A_absval_or_None).
        Values are formed here and consumed by the differencing below;
        none is stored beyond its stem mean and none egresses."""
        c = step.cell(text)
        if not c.is_present:
            return False, None, None, None
        try:
            dec = c.decompose(None)
        except Exception:
            return False, None, None, None
        if not dec:
            return False, None, None, None
        try:
            roles = N.cell_roles(c, CANONICAL)
        except Exception:
            roles = None
        if roles is None or not any(r == "faller" for _w, _wt, r in roles):
            return False, None, None, None
        ws_f, zs_f, ws_r, zs_r = [], [], [], []
        for w, wt, role in roles:
            k = N.norm_key(w, "en", fold=False)
            if N.is_function_word(k, "en"):
                continue
            zv = {}
            for dim in ("arousal", "valence"):
                val, _src = N.lookup(tabs[dim], k.casefold(), "en")
                zv[dim] = val
            if any(x is None for x in zv.values()):
                continue
            if role == "faller":
                ws_f.append(wt); zs_f.append(abs(zv["valence"]))
            else:
                ws_r.append(wt); zs_r.append(abs(zv["valence"]))
        a_val = None
        if len(ws_f) >= B.QUALIFYING_MIN and len(ws_r) >= B.QUALIFYING_MIN:
            mf, mr = wmean(zs_f, ws_f), wmean(zs_r, ws_r)
            if mf is not None and mr is not None:
                a_val = mf - mr
        return True, float(dec["tail_excess"]), float(dec["departed"]), a_val

    per_stem = {m: collections.defaultdict(list) for m in ("H1", "H5", "H6")}
    n_both_an = n_one_an = n_both_a = 0
    for ei, ((_b, _a), step) in enumerate(sorted(steps.items()), 1):
        for s, (t_m, t_u) in pair.items():
            an_m, te_m, dp_m, av_m = measures(step, t_m)
            an_u, te_u, dp_u, av_u = measures(step, t_u)
            if an_m and an_u:
                n_both_an += 1
                per_stem["H1"][s].append(te_m - te_u)
                per_stem["H5"][s].append(dp_m - dp_u)
                if av_m is not None and av_u is not None:
                    n_both_a += 1
                    per_stem["H6"][s].append(av_m - av_u)
            elif an_m or an_u:
                n_one_an += 1
        print("  [%2d/%d] keys so far: analysed %d, A %d"
              % (ei, len(steps), n_both_an, n_both_a), flush=True)

    if n_both_an != EXPECT_BOTHSIDES_ANALYSED:
        refuse("both-sides analysed keys = %d, not %d"
               % (n_both_an, EXPECT_BOTHSIDES_ANALYSED))
    if n_one_an != EXPECT_ONESIDED_ANALYSED:
        refuse("one-sided analysed keys = %d, not %d"
               % (n_one_an, EXPECT_ONESIDED_ANALYSED))
    if n_both_a != EXPECT_BOTHSIDES_A:
        refuse("both-sides A keys = %d, not %d" % (n_both_a, EXPECT_BOTHSIDES_A))

    out = {}
    for meas, expect_k in (("H1", EXPECT_H1_K), ("H5", EXPECT_H1_K),
                           ("H6", EXPECT_STEMS_H6)):
        d = [sum(v) / len(v) for v in per_stem[meas].values() if v]
        k = len(d)
        if k != expect_k:
            refuse("%s stems = %d, not %d" % (meas, k, expect_k))
        mean = sum(d) / k                     # formed; does not egress
        sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (k - 1))
        out[meas] = (sd, k)

    h1_sd, h1_k = out["H1"]
    if round(h1_sd, 6) != EXPECT_H1_SD or h1_k != EXPECT_H1_K:
        refuse("H1 known answer failed: this pass's H1 (sd to 6dp, k) does "
               "not equal the published (0.014765, 684) — the machinery and "
               "the artifact disagree, and neither new number may be read")

    print("known answers verified: pairs, edges, analysed keys (both/one-"
          "sided), A keys, and all three stem counts match the public "
          "figures.", flush=True)
    print("H1  REPRODUCED the published pair: sd = 0.014765, k = 684  "
          "(known answer — not a new number)")
    print("H5  sd_d = %.6f   k = %d" % out["H5"])
    print("H6  sd_d = %.6f   k = %d" % out["H6"])


if __name__ == "__main__":
    main()
