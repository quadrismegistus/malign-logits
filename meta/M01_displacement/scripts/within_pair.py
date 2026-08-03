"""M01 WITHIN-PAIR DISPLACEMENT — the computation for registration
`8ff56206deac048e`, plus the BINDING RIDER of [3047].

WHAT IT COMPUTES (registration §2, §3)

    at-risk(L)   pairs whose BOTH members are SCORED on BOTH arms of L
    rate_M(L)    fraction of at-risk pairs whose MARKED member fires a site
    rate_U(L)    fraction whose UNMARKED member fires
    Delta(L)     rate_M(L) - rate_U(L)
    PRIMARY      SIGN TEST over lineages, ONE-SIDED UPPER, alpha 0.05
    SECONDARY    DEPTH: where BOTH fire, rank of the aligned arm's top word
                 in the base arm's ordering; within-pair difference of means

THE FROZEN RULE IS IMPORTED, NEVER REIMPLEMENTED. `prepare`, `top_word`,
`classify` and `pairs_from_map` come from `m05_sites.py` @ b8fd9a52cd5c794b and
this module refuses to run against any other hash.

DEPTH IS PINNED TO THE FROZEN RULE'S OWN VERDICT. `classify` computes
`avail = ob.index(wa)` and calls the cell FREE when `avail <= AVAIL_MAX`.
Depth is that same `avail`, and `_depth_of` ASSERTS the equivalence
(`depth <= AVAIL_MAX` iff FREE) on every cell it measures. A secondary that
drifted from the primary's own availability test would be a second definition
of displacement wearing the first one's name.

THE RIDER, VERBATIM ([3047]):
    THIS EXCLUSION IS APPLIED BY THE COMPUTATION AND MUST NOT BE INHERITED
    FROM THE STORE. ... The producer therefore names the two checkpoints
    explicitly and ASSERTS their absence from the admitted set; it must not
    rely on an empty-rows filter to drop them incidentally, because a filter
    that happens to exclude is not an exclusion anyone can audit.

So `EXCLUDED_CHECKPOINTS` is a DECLARED CONSTANT, applied by name, and
`_assert_rider` fails loudly if either checkpoint reaches the admitted set by
any route. The empty-rows filter still exists — it is how a cell is judged
UNSCORED — but it is not what performs this exclusion, and the assertion is
what proves the difference.
"""
import argparse
import collections
import hashlib
import json
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))          # .../M01_displacement/scripts
CAMPAIGN = os.path.dirname(HERE)                          # .../M01_displacement
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))         # the repository root
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

FROZEN_SITES = "b8fd9a52cd5c794b"
REGISTRATION = "8ff56206deac048e"
CAT = os.path.join(ROOT, "data/prompt_categorisation.json")

#: §4.2 + [3047] RIDER. Named, never inferred from emptiness.
EXCLUDED_CHECKPOINTS = frozenset({
    "tiiuae/Falcon-H1-7B-Base",
    "tiiuae/Falcon-H1-7B-Instruct",
})
#: §4.1. The `assistant` collision: these PAIRS on these MODELS only.
ASSISTANT_PAIRS = frozenset({"nps_18", "r2bpw_003", "r2bpw_031"})
ASSISTANT_MODELS = frozenset({
    "tiiuae/Falcon3-Mamba-7B-Instruct",
    "tiiuae/falcon-mamba-7b-instruct",
})
AT_RISK_FLOOR = 20          # §4.3
ALPHA = 0.05
POWER = 0.80


# ── the statistic ────────────────────────────────────────────────────────
def binom_sf(k, n, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def sign_test(deltas, alpha=ALPHA):
    """(n, positives, critical k, achieved size, p-value). ZEROES ARE DROPPED.

    A Delta of exactly 0 is no evidence either way and inflating n with ties
    would make the test anti-conservative. The count dropped is REPORTED.
    """
    nz = [d for d in deltas if d != 0.0]
    n = len(nz)
    pos = sum(1 for d in nz if d > 0)
    k = next((i for i in range(n + 1) if binom_sf(i, n, 0.5) <= alpha), None)
    return {"n": n, "ties_dropped": len(deltas) - n, "positives": pos,
            "critical_k": k,
            "achieved_size": binom_sf(k, n, 0.5) if k is not None else None,
            "p_value": binom_sf(pos, n, 0.5) if n else None,
            "reject": (k is not None and pos >= k)}


def mde(n, alpha=ALPHA, power=POWER):
    k = next((i for i in range(n + 1) if binom_sf(i, n, 0.5) <= alpha), None)
    if k is None:
        return None, None
    lo, hi = 0.5, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if binom_sf(k, n, mid) >= power:
            hi = mid
        else:
            lo = mid
    return k, hi


# ── corpus ───────────────────────────────────────────────────────────────
def m01_pairs(cat=CAT):
    rows = json.load(open(cat))["prompts"]
    rows = list(rows.values()) if isinstance(rows, dict) else rows
    out = collections.defaultdict(dict)
    dom = {}
    for r in rows:
        if (r.get("pair_role")
                and r.get("contrast_type") == "transgressive_swap"
                and str(r.get("source", "")).startswith("M01_PAIRS")):
            out[r["pair_id"]][r["pair_role"]] = str(r["prompt"])
            dom[r["pair_id"]] = r.get("domain")
    return ({k: v for k, v in out.items() if len(v) == 2}, dom)


def _depth_of(S, base_row, algn_row):
    """The frozen rule's own `avail`, with its FREE verdict asserted.

    None when the cell does not fire. Otherwise `ob.index(wa)` — the rank of
    the aligned top word in the base ordering — and the assertion binds it to
    `classify`'s decision so the secondary cannot drift from the primary.
    """
    labs = S.classify(base_row, algn_row, frozenset())
    ob, _pb = base_row
    oa, pa = algn_row
    wa = S.top_word(oa, pa)
    if "FREE" not in labs:
        return None
    d = ob.index(wa)
    assert d <= S.AVAIL_MAX, (
        "depth %d exceeds AVAIL_MAX %d on a cell classify called FREE — the "
        "secondary has drifted from the primary's availability test" % (d, S.AVAIL_MAX))
    return d


def _tail_diag(pb, pa, rb, ra):
    """DIAGNOSTIC COLUMN, ratified [3052].a. Disclosure, never a readout.

    The registered rate metric fires when the TOP WORD CHANGES, and a top word
    changes two ways it cannot distinguish:

        SUBSTITUTION  mass moves to a named substitute
        DEFLECTION    mass drains into the unresolved tail and the runner-up
                      floats up

    `tail_excess` separates them BY SIGN — positive is dispersal into
    unresolved mass, negative is the tail giving mass up to nameable words.
    `tail_share` is js_tail/js_total, the comparability gate.

    THIS CHANGES NO DECLARED QUANTITY. It reports what the primary includes.
    Returns None if the decomposition cannot be formed, which is recorded as
    `diag_unavailable` rather than silently skipped.
    """
    from malign_logits.movement import decompose, CANONICAL
    try:
        d = decompose(pb, pa, CANONICAL, residual_pre=rb, residual_post=ra)
    except Exception:
        return None
    return (d.get("tail_share"), d.get("tail_excess"))


def _assert_rider(admitted_models):
    """[3047]. Asserts the ADMITTED SET, never the grid.

    THE GRID LEGITIMATELY CONTAINS THE EXCLUDED CHECKPOINTS — they are in the
    store, correctly keyed and structurally perfect, which is the whole reason
    the rider exists. What must not happen is their reaching the admitted set.
    An earlier version asserted on the grid and would have refused to run at
    all; that is not the property the rider names.

    NOTE ON ITS OWN REACHABILITY: `compute` drops these lineages BY NAME before
    admission, so this guard cannot fire through the normal path — which is
    correct and makes it untestable through `compute`. It is therefore
    exercised DIRECTLY in the selftest. A guard that no test can reach is
    indistinguishable from one that does not work.
    """
    leaked = EXCLUDED_CHECKPOINTS & set(admitted_models)
    if leaked:
        raise AssertionError(
            "RIDER VIOLATION: %s reached the admitted set. The registration "
            "excludes these BY NAME; a filter that happens to exclude is not "
            "an exclusion anyone can audit." % sorted(leaked))
    return True


def compute(grid, lineages, pairs, domains, S, resid=None):
    """-> (per-lineage records, ledger). No I/O, fully injectable."""
    led = collections.Counter()
    recs = []
    for b, a in lineages:
        if b in EXCLUDED_CHECKPOINTS or a in EXCLUDED_CHECKPOINTS:
            led["lineage_excluded_by_name"] += 1
            continue
        gb, ga = grid.get(b, {}), grid.get(a, {})
        at_risk = fires_m = fires_u = 0
        dm, du = [], []
        by_dom = collections.Counter()
        tshare, disp, subst, diag_na = [], 0, 0, 0
        for pid, mem in pairs.items():
            if pid in ASSISTANT_PAIRS and (b in ASSISTANT_MODELS or a in ASSISTANT_MODELS):
                led["pair_excluded_assistant"] += 1
                continue
            mk, um = mem["MARKED"], mem["UNMARKED"]
            if not all(t in gb and t in ga for t in (mk, um)):
                led["pair_not_at_risk"] += 1
                continue
            at_risk += 1
            by_dom[domains.get(pid)] += 1
            d_m = _depth_of(S, gb[mk], ga[mk])
            d_u = _depth_of(S, gb[um], ga[um])
            if d_m is not None:
                fires_m += 1
            if d_u is not None:
                fires_u += 1
            #: [3052].a diagnostic -- on FIRING cells only, both members
            if resid is not None:
                for t_, d_ in ((mk, d_m), (um, d_u)):
                    if d_ is None:
                        continue
                    got = _tail_diag(gb[t_][1], ga[t_][1],
                                     resid.get(b, {}).get(t_, 0.0),
                                     resid.get(a, {}).get(t_, 0.0))
                    if got is None or got[0] is None:
                        diag_na += 1
                        continue
                    tshare.append(got[0])
                    if got[1] > 0:
                        disp += 1
                    elif got[1] < 0:
                        subst += 1
            if d_m is not None and d_u is not None:      # §3: BOTH fire
                dm.append(d_m)
                du.append(d_u)
        #: §4.3 — the floor decides ADMISSION, never whether a lineage is
        #: measured. A below-floor lineage is "named individually, never
        #: summarised", and a name without its rates cannot be inspected: a
        #: reader must be able to see WHAT was dropped, not just that it was.
        admitted = at_risk >= AT_RISK_FLOOR
        if not admitted:
            led["lineage_below_floor"] += 1
        if at_risk == 0:
            recs.append({"base": b, "aligned": a, "at_risk": 0,
                         "admitted": False, "rate_M": None, "rate_U": None,
                         "delta": None, "n_both_fire": 0, "depth_M": None,
                         "depth_U": None, "depth_delta": None, "domains": {}})
            continue
        recs.append({
            "base": b, "aligned": a, "at_risk": at_risk, "admitted": admitted,
            "rate_M": fires_m / at_risk, "rate_U": fires_u / at_risk,
            "delta": fires_m / at_risk - fires_u / at_risk,
            "n_both_fire": len(dm),
            "depth_M": (sum(dm) / len(dm)) if dm else None,
            "depth_U": (sum(du) / len(du)) if du else None,
            "depth_delta": ((sum(dm) / len(dm)) - (sum(du) / len(du))) if dm else None,
            "domains": dict(by_dom),
            #: DIAGNOSTIC, not a readout. [3052].a
            "diag_n": len(tshare),
            "diag_unavailable": diag_na,
            "tail_share_mean": (sum(tshare) / len(tshare)) if tshare else None,
            "fires_dispersed": disp,
            "fires_substituted": subst,
        })
    _assert_rider([m for r in recs if r["admitted"]
                   for m in (r["base"], r["aligned"])])
    return recs, led


# ── known-answer FIRST ───────────────────────────────────────────────────
def selftest(verbose=False):
    import m05_sites as S
    ok = []

    def case(name, cond):
        ok.append((name, bool(cond)))
        if verbose:
            print("  %-58s %s" % (name, "ok" if cond else "FAIL"))

    # -- KNOWN ANSWER: a fixture whose Delta is known by construction --------
    def row(words):
        return S.prepare([{"word": w, "p": p} for w, p in words])
    #  base prefers X; aligned prefers Y which sits at rank 1 in base -> FREE
    fires = (row([("x", .6), ("y", .3)]), row([("y", .7), ("x", .2)]))
    same = (row([("x", .6), ("y", .3)]), row([("x", .7), ("y", .2)]))
    pairs = {"p%d" % i: {"MARKED": "m%d" % i, "UNMARKED": "u%d" % i} for i in range(4)}
    doms = {k: "taboo" for k in pairs}
    #  MARKED fires on 3 of 4, UNMARKED on 1 of 4  ->  Delta = 0.75 - 0.25 = 0.5
    gb, ga = {}, {}
    for i in range(4):
        gb["m%d" % i], ga["m%d" % i] = (fires if i < 3 else same)
        gb["u%d" % i], ga["u%d" % i] = (fires if i < 1 else same)
    grid = {"B": gb, "A": ga}
    recs, led = compute(grid, [("B", "A")], pairs, doms, S)
    r = recs[0]
    case("KNOWN ANSWER: at_risk == 4", r["at_risk"] == 4)
    case("KNOWN ANSWER: rate_M == 0.75", abs(r["rate_M"] - 0.75) < 1e-12)
    case("KNOWN ANSWER: rate_U == 0.25", abs(r["rate_U"] - 0.25) < 1e-12)
    case("KNOWN ANSWER: Delta == 0.50", abs(r["delta"] - 0.50) < 1e-12)
    case("KNOWN ANSWER: below floor -> not admitted", r["admitted"] is False)

    # -- the denominator is AT-RISK, never 'both fire' ----------------------
    case("denominator counts non-firing pairs (4 at risk, 1 both-fire)",
         r["at_risk"] == 4 and r["n_both_fire"] == 1)

    # -- depth is the frozen rule's own avail -------------------------------
    case("depth of a rank-1 substitute == 1", _depth_of(S, *fires) == 1)
    case("depth is None when the cell does not fire", _depth_of(S, *same) is None)

    # -- §4.2 RIDER ---------------------------------------------------------
    raised = False
    try:
        _assert_rider(["B", "tiiuae/Falcon-H1-7B-Base"])
    except AssertionError:
        raised = True
    case("RIDER: guard RAISES when an excluded name reaches admitted", raised)
    case("RIDER: guard PASSES a clean admitted set", _assert_rider(["B", "A"]))
    case("RIDER: the grid may legitimately CONTAIN them",
         compute({"tiiuae/Falcon-H1-7B-Base": {}, "B": gb, "A": ga},
                 [("B", "A")], pairs, doms, S) is not None)
    recs2, led2 = compute({"B": gb, "tiiuae/Falcon-H1-7B-Instruct": ga},
                          [("B", "tiiuae/Falcon-H1-7B-Instruct")], pairs, doms, S)
    case("RIDER: excluded lineage is dropped BY NAME, and counted",
         led2["lineage_excluded_by_name"] == 1)
    case("RIDER: exclusion is not the empty-rows path (grid was non-empty)",
         len(ga) > 0)

    # -- §4.1 assistant pairs, per model ------------------------------------
    _r3, led3 = compute({"B": gb, "tiiuae/falcon-mamba-7b-instruct": ga},
                        [("B", "tiiuae/falcon-mamba-7b-instruct")],
                        dict(pairs, nps_18={"MARKED": "m0", "UNMARKED": "u0"}),
                        dict(doms, nps_18="sexual"), S)
    case("assistant pair excluded on an affected model",
         led3["pair_excluded_assistant"] == 1)
    _r4, led4 = compute(grid, [("B", "A")],
                        dict(pairs, nps_18={"MARKED": "m0", "UNMARKED": "u0"}),
                        dict(doms, nps_18="sexual"), S)
    case("assistant pair KEPT on an unaffected model",
         led4["pair_excluded_assistant"] == 0)

    # -- DIAGNOSTIC COLUMN: a branch I added, so a case reaches it ----------
    # Constructed so the SIGN of tail_excess is known by hand.
    #   base {x .6, y .3} resid .1 ; x falls to .2 -> R = .8, S = .4, ratio 2,
    #   so the tail's renormalised expectation is .2.
    #     aligned resid .1  -> excess -.1  NEGATIVE = SUBSTITUTED
    #     aligned resid .45 -> excess +.25 POSITIVE = DISPERSED
    subst_pre, subst_post = row([("x", .6), ("y", .3)]), row([("y", .7), ("x", .2)])
    disp_post = row([("y", .35), ("x", .2)])
    gb2 = {"m0": subst_pre, "u0": subst_pre, "m1": subst_pre, "u1": subst_pre}
    ga2 = {"m0": subst_post, "u0": subst_post, "m1": disp_post, "u1": disp_post}
    rb2 = {"B": {"m0": .1, "u0": .1, "m1": .1, "u1": .1}}
    rb2["A"] = {"m0": .1, "u0": .1, "m1": .45, "u1": .45}
    p2 = {"p0": {"MARKED": "m0", "UNMARKED": "u0"},
          "p1": {"MARKED": "m1", "UNMARKED": "u1"}}
    d2 = {"p0": "taboo", "p1": "taboo"}
    r2, _l2 = compute({"B": gb2, "A": ga2}, [("B", "A")], p2, d2, S,
                      resid={"B": rb2["B"], "A": rb2["A"]})
    rr = r2[0]
    case("DIAGNOSTIC: the branch is REACHED (4 firing cells decomposed)",
         rr["diag_n"] == 4)
    case("DIAGNOSTIC: 2 cells SUBSTITUTED (tail_excess < 0)",
         rr["fires_substituted"] == 2)
    case("DIAGNOSTIC: 2 cells DISPERSED (tail_excess > 0)",
         rr["fires_dispersed"] == 2)
    case("DIAGNOSTIC: absent when no residual map is passed",
         compute({"B": gb2, "A": ga2}, [("B", "A")], p2, d2, S)[0][0]["diag_n"] == 0)
    case("DIAGNOSTIC changes NO declared quantity (delta identical either way)",
         compute({"B": gb2, "A": ga2}, [("B", "A")], p2, d2, S)[0][0]["delta"]
         == rr["delta"])

    # -- the sign test ------------------------------------------------------
    t = sign_test([1.0] * 10 + [-1.0] * 10)
    case("sign test: 10/20 positive does not reject", not t["reject"])
    t2 = sign_test([1.0] * 20)
    case("sign test: 20/20 rejects", t2["reject"])
    t3 = sign_test([1.0, -1.0, 0.0, 0.0])
    case("sign test: exact zeros DROPPED, not counted", t3["n"] == 2 and t3["ties_dropped"] == 2)
    case("achieved size is reported and <= alpha",
         t2["achieved_size"] is not None and t2["achieved_size"] <= ALPHA)
    k59, p59 = mde(59)
    case("MDE at n=59 reproduces the registered 0.670", abs(p59 - 0.670) < 0.001)

    n_ok = sum(1 for _, v in ok if v)
    print("selftest %d/%d" % (n_ok, len(ok)))
    return n_ok == len(ok)


def main(a):
    import m05_sites as S
    h = hashlib.sha256(open(os.path.join(HERE, "m05_sites.py"), "rb")
                       .read()).hexdigest()[:16]
    if h != FROZEN_SITES:
        print("REFUSING: site rule is %s, not the frozen %s" % (h, FROZEN_SITES))
        return 1
    rh = hashlib.sha256(open(os.path.join(
        CAMPAIGN, "registration_f_within_pair.md"), "rb").read()).hexdigest()[:16]
    print("site rule    %s  OK" % h)
    print("registration %s  %s" % (rh, "OK" if rh == REGISTRATION else "MISMATCH"))
    if rh != REGISTRATION:
        return 1

    pairs, domains = m01_pairs()
    texts = {t for v in pairs.values() for t in v.values()}
    from malign_logits.cache import CacheManager
    cm = CacheManager()
    grid = collections.defaultdict(dict)
    resid = collections.defaultdict(dict)
    for k, v in cm.iter_items("true_word_probs"):
        if k["prompt"] not in texts:
            continue
        rs = (v or {}).get("rows") or []
        if rs:                                  # UNSCORED, not zero
            grid[k["model"]][k["prompt"]] = S.prepare(rs)
            resid[k["model"]][k["prompt"]] = float(
                ((v or {}).get("residual") or {}).get("total", 0.0))
    grid = dict(grid)
    resid = dict(resid)

    lm = json.load(open(os.path.join(ROOT, "data/lineage_map_models.json")))
    m2b, m2s = lm["model_to_base"], lm["model_to_stage"]
    import m04_producer as P
    n2s = {}
    for _m, _s in m2s.items():
        n2s.setdefault(P.norm(_m), _s)

    def arm_of(cp):
        s = m2s.get(cp) or n2s.get(P.norm(cp))
        return None if s is None else ("base" if s == "base" else "aligned")

    lineages, _bad = S.pairs_from_map(grid, m2b, arm_of)
    recs, led = compute(grid, lineages, pairs, domains, S, resid=resid)
    adm = [r for r in recs if r["admitted"]]

    print("\nlineages %d   admitted %d   below floor %d   excluded by name %d"
          % (len(lineages), len(adm), led["lineage_below_floor"],
             led["lineage_excluded_by_name"]))
    for r in recs:
        if not r["admitted"]:
            print("   BELOW FLOOR  %-44s at_risk %d" % (r["aligned"], r["at_risk"]))

    prim = sign_test([r["delta"] for r in adm])
    k, p = mde(prim["n"])
    print("\nPRIMARY — within-pair displacement-rate difference")
    print("  n %d   ties dropped %d   positives %d   critical k %s"
          % (prim["n"], prim["ties_dropped"], prim["positives"], prim["critical_k"]))
    print("  achieved size %.4f   p %.4f   REJECT %s"
          % (prim["achieved_size"], prim["p_value"], prim["reject"]))
    print("  MDE at this n: p >= %.3f" % p)

    dd = [r["depth_delta"] for r in adm if r["depth_delta"] is not None]
    sec = sign_test(dd)
    print("\nSECONDARY — displacement DEPTH (both members fire)")
    print("  lineages with depth %d   positives %d   p %s"
          % (sec["n"], sec["positives"],
             ("%.4f" % sec["p_value"]) if sec["p_value"] is not None else "n/a"))

    dn = sum(r["diag_n"] for r in adm)
    dsp = sum(r["fires_dispersed"] for r in adm)
    sub = sum(r["fires_substituted"] for r in adm)
    ts = [r["tail_share_mean"] for r in adm if r["tail_share_mean"] is not None]
    print("\nDIAGNOSTIC COLUMN ([3052].a) — disclosure, NOT a readout")
    print("  firing cells decomposed        %d   unavailable %d"
          % (dn, sum(r["diag_unavailable"] for r in adm)))
    print("  tail_excess POSITIVE (dispersed / DEFLECTION)  %d  (%.1f%%)"
          % (dsp, 100 * dsp / max(1, dsp + sub)))
    print("  tail_excess NEGATIVE (SUBSTITUTED)             %d  (%.1f%%)"
          % (sub, 100 * sub / max(1, dsp + sub)))
    print("  mean tail_share over lineages  %s"
          % (("%.4f" % (sum(ts) / len(ts))) if ts else "n/a"))
    print("  READ: a 'fire' is not necessarily a substitution; the share above")
    print("        is what the registered rate metric cannot distinguish.")

    if a.out:
        json.dump({"registration": rh, "site_rule": h, "primary": prim,
                   "secondary": sec, "mde": p, "ledger": dict(led),
                   "lineages": recs}, open(a.out, "w"), indent=2)
        print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    _a = ap.parse_args()
    #: [3059]: the previous form had 0 in BOTH branches of the outer ternary,
    #: so `--selftest` exited 0 whatever the suite found. The cases were doing
    #: their job and the exit code said "pass" regardless — a gate that cannot
    #: fail, and everything reading the status was blind: CI, wrappers, and a
    #: mutation harness that consequently scored three rider mutants SURVIVED.
    #:
    #: The one-line repair proposed in the audit (`else 1`) fixes --selftest
    #: and BREAKS the normal path: `main()` returns an int, so a successful run
    #: returns 0, and `0 is True` is False — a clean run would exit 1. The two
    #: paths return different TYPES and must not share a predicate.
    if _a.selftest:
        sys.exit(0 if selftest(verbose=True) else 1)
    sys.exit(main(_a))
