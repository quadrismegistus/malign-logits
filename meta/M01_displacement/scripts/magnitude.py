"""M01 MAGNITUDE — the producer for `m01_magnitude_registration.md`
@ `efbab15841eae4c2` (FROZEN, three signatures).

    §3  d(L,p) = Q(MARKED) - Q(UNMARKED), pairs where BOTH members fire
        D(L)   = MEDIAN of d over L's pairs
    §4  SIGN-FLIP PERMUTATION over the 34 unit summaries, one-sided upper,
        alpha 0.05, 100,000 draws, resolution limit 1e-5 reported
    §5  PRIMARY `departed`   SECONDARY `concentration` (scale-free)
        description-only: arrived, selectivity, captured, js_*
    §6.2 per-unit both-fire / onlyM / onlyU / skew AS A COLUMN; the nine
        |skew| > 0.25 units NAMED and a declared drop-nine sensitivity
    §7  base top-mass and mover count, both members, AS A COLUMN

INPUT IS DECLARED BY PATH AND NOTHING ELSE IS ACCEPTED.
`departed` is ALSO a column in `data/c1_neutral_floor.csv` and
`data/c1_uncertainty_shape.csv` — Registration C artifacts keyed by
family/text, with no marked/unmarked structure and no relation to this
quantity ([3124].1). **A quantity resolved by COLUMN NAME across two
registrations is the pair_role polysemy with a different field.** This module
reads the twp store through the cache manager and the frozen site rule, and
refuses to take `departed` from anywhere a name match might find it.

WHY A SIGN-FLIP PERMUTATION AND NOT A SIGN TEST (§4, re-derived independently
by malign at [3127]): at n=34 the sign test needs a standardised effect of
0.599 and the permutation 0.426-0.431 — a 29% smaller detectable effect, for
the same alpha, power and exchangeable object. The rate test reduced 663 paired
measurements per unit to ONE BIT each.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # .../M01_displacement/scripts
CAMPAIGN = os.path.dirname(HERE)                          # .../M01_displacement
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))         # the repository root
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

REGISTRATION = "efbab15841eae4c2"
AMENDMENT = "1356aa2ff274b796"
FROZEN_SITES = "b8fd9a52cd5c794b"
#: §6.2 — declared BEFORE any magnitude is read.
SKEW_FLAG = 0.25
PERM_DRAWS = 100_000
PERM_SEED = 20260803
ALPHA = 0.05


# ── §4 the statistic ─────────────────────────────────────────────────────
def sign_flip_p(values, draws=PERM_DRAWS, seed=PERM_SEED):
    """One-sided upper p by sign-flip permutation. Distribution-free.

    The null is that each UNIT's summary is equally likely to carry either
    sign — the unit is the exchangeable object, which is what keeps the
    pseudo-replication guard intact while using the magnitudes the sign test
    threw away.

    Monte Carlo, not exhaustive: 2^34 is 1.7e10. The resolution limit is
    1/draws and is REPORTED, never elided — a p smaller than it is reported
    AS the limit, not as a smaller number.
    """
    obs = sum(values)
    rng = random.Random(seed)
    n = len(values)
    hits = 0
    for _ in range(draws):
        s = 0.0
        for v in values:
            s += v if rng.getrandbits(1) else -v
        if s >= obs:
            hits += 1
    p = (hits + 1) / (draws + 1)          # add-one: p is never exactly 0
    return {"statistic": obs, "n": n, "draws": draws,
            "p_value": p, "resolution_limit": 1.0 / draws,
            "at_resolution_limit": hits == 0,
            "reject": p <= ALPHA}


def cohen_d(values):
    """Observed standardised effect, for comparison against §8's MDE 0.426.

    POPULATION SD (`pstdev`), DECLARED ([3132].4). At n=34 `stdev/pstdev` is
    1.0150, so the observed d shifts 1.5% on this choice while the registered
    threshold 0.426 is fixed — small until an observed d lands near 0.42.

    Population is the right convention here because it is the one BOTH power
    derivations used: the closed form `(z_.95+z_.80)/sqrt(n)` and malign's
    Monte Carlo over `N(d,1)`. **They agreed at [3127] under this convention;
    stating it turns that agreement from luck into declaration.**
    """
    if len(values) < 2:
        return None
    sd = st.pstdev(values)
    return (st.mean(values) / sd) if sd > 0 else None


# ── the per-pair quantity ────────────────────────────────────────────────
def pair_quantity(S, gb, ga, rb, ra, text):
    """decompose() for one member, plus §7's base-side diagnostic.

    None when the cell does not fire — §3 requires BOTH members firing, and a
    non-firing member has no magnitude to compare.
    """
    from malign_logits.movement import decompose, CANONICAL
    if text not in gb or text not in ga:
        return None
    labs = S.classify(gb[text], ga[text], frozenset())
    if "FREE" not in labs:
        return None
    ob, pb = gb[text]
    d = decompose(pb, ga[text][1], CANONICAL,
                  residual_pre=rb.get(text, 0.0), residual_post=ra.get(text, 0.0))
    #: §7 — the confound diagnostic, as a COLUMN. A transgressive prompt may
    #: have a PEAKIER base distribution and so more mass available to move;
    #: `departed` cannot distinguish that from displacement, and
    #: `concentration` (scale-free) is the partner that survives it.
    d["_base_top_mass"] = pb[ob[0]] if ob else None
    d["_base_movers"] = len(ob)
    return d


def build(units_edges, grid, resid, pairs, S):
    """-> per-unit records. Pure apart from decompose(). No I/O."""
    out = {}
    for base, arms in units_edges.items():
        per_arm = []
        for arm in arms:
            gb, ga = grid.get(base, {}), grid.get(arm, {})
            rb, ra = resid.get(base, {}), resid.get(arm, {})
            dep, con = [], []
            both = onlyM = onlyU = 0
            base_top_m, base_top_u = [], []
            for pid, mem in pairs.items():
                M = pair_quantity(S, gb, ga, rb, ra, mem["MARKED"])
                U = pair_quantity(S, gb, ga, rb, ra, mem["UNMARKED"])
                if M is not None and U is not None:
                    both += 1
                    dep.append(M["departed"] - U["departed"])
                    if M["concentration"] is not None and U["concentration"] is not None:
                        con.append(M["concentration"] - U["concentration"])
                    base_top_m.append(M["_base_top_mass"])
                    base_top_u.append(U["_base_top_mass"])
                elif M is not None:
                    onlyM += 1
                elif U is not None:
                    onlyU += 1
            if not dep:
                continue
            per_arm.append({
                "arm": arm, "both": both, "onlyM": onlyM, "onlyU": onlyU,
                "skew": (onlyM - onlyU) / both if both else None,
                "D_departed": st.median(dep),
                "D_concentration": st.median(con) if con else None,
                "base_top_M": st.median(base_top_m), "base_top_U": st.median(base_top_u),
                "n_pairs": len(dep),
            })
        if not per_arm:
            continue
        med = lambda k: st.median([a[k] for a in per_arm if a[k] is not None]) \
            if any(a[k] is not None for a in per_arm) else None
        out[base] = {
            "arms": [a["arm"] for a in per_arm],
            "D_departed": med("D_departed"),
            "D_concentration": med("D_concentration"),
            "both": sum(a["both"] for a in per_arm) // len(per_arm),
            "onlyM": sum(a["onlyM"] for a in per_arm) // len(per_arm),
            "onlyU": sum(a["onlyU"] for a in per_arm) // len(per_arm),
            "skew": med("skew"),
            "base_top_M": med("base_top_M"), "base_top_U": med("base_top_U"),
        }
    return out


def selftest(verbose=False):
    ok = []
    def case(n, c):
        ok.append(bool(c))
        if verbose:
            print("  %-58s %s" % (n, "ok" if c else "FAIL"))

    # -- §4 KNOWN ANSWER: the permutation test on constructed inputs --------
    # A FAILING CASE CORRECTED ITSELF HERE. The first version asserted that
    # all-positive input lands AT the resolution limit. It does not at n=10:
    # the all-heads flip has probability 2^-10 = 1/1024, so in 2,000 draws it
    # occurs ~2 times and `hits` is not 0. The CASE encoded an assumption the
    # arithmetic does not support — the same shape as the alternation table.
    r = sign_flip_p([1.0] * 10, draws=2000)
    case("all-positive at n=10 REJECTS", r["reject"])
    case("  and p tracks 2^-n, not the draw count", 0.0005 < r["p_value"] < 0.006)
    case("  so it is NOT at the resolution limit at this n",
         not r["at_resolution_limit"])
    r_big = sign_flip_p([1.0] * 25, draws=2000)
    case("all-positive at n=25 IS at the resolution limit (2^-25 << 1/2000)",
         r_big["at_resolution_limit"])

    # -- THE ADD-ONE, WHICH NOTHING TESTED ([3132].3). The resolution-limit
    #    case above checks `hits == 0`, TRUE with or without the correction —
    #    it verifies the FLAG and never the VALUE the flag is about. Without
    #    Phipson-Smyth a fully-separated result reports p = 0.0: a claim of
    #    infinite evidence from 100,000 draws, which is the unearned precision
    #    the whole resolution-limit machinery exists to prevent.
    case("add-one: p at full separation is EXACTLY 1/(draws+1), not 0",
         r_big["p_value"] == 1.0 / (2000 + 1))
    case("  so p is strictly positive on every reachable input",
         r_big["p_value"] > 0 and r["p_value"] > 0)
    r2 = sign_flip_p([1.0, -1.0] * 8, draws=2000)
    case("symmetric input does not reject", not r2["reject"])
    r3 = sign_flip_p([0.5] * 3 + [-0.4] * 3, draws=4000)
    case("near-symmetric input: p well above alpha", r3["p_value"] > 0.2)
    case("resolution limit is REPORTED", r3["resolution_limit"] == 1/4000)

    # -- MAGNITUDE IS USED, WHICH IS THE WHOLE POINT ------------------------
    signs_same_a = sign_flip_p([0.1, 0.1, 0.1, 0.1, -0.09], draws=4000)
    signs_same_b = sign_flip_p([0.1, 0.1, 0.1, 0.1, -5.0], draws=4000)
    case("SAME SIGNS, different magnitudes -> DIFFERENT p (a sign test cannot)",
         abs(signs_same_a["p_value"] - signs_same_b["p_value"]) > 0.05)
    case("  and the big negative is the one that fails",
         signs_same_a["p_value"] < signs_same_b["p_value"])

    # -- the observed standardised effect -----------------------------------
    case("cohen_d of a constant vector is None (sd 0)", cohen_d([2.0]*5) is None)
    case("cohen_d matches hand arithmetic",
         abs(cohen_d([1.0, -1.0, 1.0, -1.0]) - 0.0) < 1e-12)

    # -- §6.2 the skew flag, declared at 0.25 -------------------------------
    case("SKEW_FLAG declared before any read", SKEW_FLAG == 0.25)
    case("PERM draws and seed are declared constants",
         PERM_DRAWS == 100_000 and PERM_SEED == 20260803)

    n_ok = sum(ok)
    print("selftest %d/%d" % (n_ok, len(ok)))
    return n_ok == len(ok)


def main(a):
    import m05_sites as S
    for f, want, name in ((os.path.join(HERE, "m05_sites.py"), FROZEN_SITES, "site rule"),
                          (os.path.join(CAMPAIGN, "registration_g_magnitude.md"),
                           REGISTRATION, "registration"),
                          (os.path.join(CAMPAIGN, "registration_f_within_pair_amendment_a.md"),
                           AMENDMENT, "amendment")):
        h = hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]
        print("%-14s %s  %s" % (name, h, "OK" if h == want else "MISMATCH -> REFUSING"))
        if h != want:
            return 1
    print()
    import within_pair as W
    import collapse as C
    pairs, _dom = W.m01_pairs()
    texts = {t for v in pairs.values() for t in v.values()}
    from malign_logits.cache import CacheManager
    cm = CacheManager()
    grid = collections.defaultdict(dict)
    resid = collections.defaultdict(dict)
    for k, v in cm.iter_items("true_word_probs"):
        if k["prompt"] not in texts:
            continue
        rs = (v or {}).get("rows") or []
        if rs:
            grid[k["model"]][k["prompt"]] = S.prepare(rs)
            resid[k["model"]][k["prompt"]] = float(
                ((v or {}).get("residual") or {}).get("total", 0.0))
    grid, resid = dict(grid), dict(resid)

    lm = json.load(open(os.path.join(ROOT, "data/lineage_map_models.json")))
    m2b, m2s = lm["model_to_base"], lm["model_to_stage"]
    import m04_producer as P
    n2s = {}
    for _m, _s in m2s.items():
        n2s.setdefault(P.norm(_m), _s)
    def arm_of(cp):
        s = m2s.get(cp) or n2s.get(P.norm(cp))
        return None if s is None else ("base" if s == "base" else "aligned")
    lineages, _ = S.pairs_from_map(grid, m2b, arm_of)

    #: §2 — Amendment A's edge, applied here exactly as `m01_collapse` applies it
    edges = collections.defaultdict(list)
    for b, al in lineages:
        if b in W.EXCLUDED_CHECKPOINTS or al in W.EXCLUDED_CHECKPOINTS:
            continue
        if m2s.get(al) == C.EDGE_STAGE and not C.is_reasoning(al):
            edges[b].append(al)
    print("units=%d field=model_to_base edge=%s/non-reasoning lineages=%d"
          % (len(edges), C.EDGE_STAGE, len(lineages)))

    units = build(edges, grid, resid, pairs, S)
    print("units with a magnitude: %d\n" % len(units))

    flagged = sorted(k for k, v in units.items()
                     if v["skew"] is not None and abs(v["skew"]) > SKEW_FLAG)
    dep = [v["D_departed"] for v in units.values() if v["D_departed"] is not None]
    con = [v["D_concentration"] for v in units.values() if v["D_concentration"] is not None]

    pr = sign_flip_p(dep)
    print("PRIMARY  `departed`   n %d  median %+0.5f  d %s" %
          (len(dep), st.median(dep),
           ("%.3f" % cohen_d(dep)) if cohen_d(dep) is not None else "n/a"))
    print("  p %.5f   resolution %.0e   REJECT %s%s"
          % (pr["p_value"], pr["resolution_limit"], pr["reject"],
             "   (AT THE RESOLUTION LIMIT)" if pr["at_resolution_limit"] else ""))
    sc = sign_flip_p(con)
    print("SECONDARY `concentration`  n %d  median %+0.5f  p %.5f  REJECT %s"
          % (len(con), st.median(con), sc["p_value"], sc["reject"]))

    print("\n§6.2 DECLARED SENSITIVITY — drop the %d units with |skew| > %.2f"
          % (len(flagged), SKEW_FLAG))
    for f in flagged:
        print("   %-44s skew %+0.3f" % (f, units[f]["skew"]))
    kept = [v["D_departed"] for k, v in units.items()
            if k not in flagged and v["D_departed"] is not None]
    ps = sign_flip_p(kept)
    print("   PRIMARY without them: n %d  p %.5f  REJECT %s"
          % (len(kept), ps["p_value"], ps["reject"]))

    print("\n§7 CONFOUND DIAGNOSTIC — base-side top mass, both members")
    bm = [v["base_top_M"] for v in units.values() if v["base_top_M"] is not None]
    bu = [v["base_top_U"] for v in units.values() if v["base_top_U"] is not None]
    print("   base top-mass  MARKED median %.4f   UNMARKED median %.4f   diff %+.4f"
          % (st.median(bm), st.median(bu), st.median(bm) - st.median(bu)))
    print("   (if this diff is large and same-signed with the primary, `departed`")
    print("    is not interpretable as displacement — §7)")

    if a.out:
        json.dump({"registration": REGISTRATION, "primary": pr, "secondary": sc,
                   "sensitivity_drop_flagged": ps, "flagged": flagged,
                   "units": units}, open(a.out, "w"), indent=2)
        print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    _a = ap.parse_args()
    if _a.selftest:
        sys.exit(0 if selftest(verbose=True) else 1)
    sys.exit(main(_a))
