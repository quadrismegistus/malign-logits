#!/usr/bin/env python3
"""Registration M's tests. Frozen text `3506032d552438e4`.

**THIS FILE IS THE TEST AND ONLY THE TEST.** `p_aligned_gold` and everything
beside it is lacan's column (`result_m_column.json` @ `dec26603e9ac1826`); this
producer computes nothing that column already holds. The describer/tester split
is what makes a falsifier for lacan's own hypothesis worth running ([3699]).

    §M3   the OVERSHOOT primary, and why the sign is inverted
    §M3a  the EXACT rank-sum p, ties, undefined rho, the floor
    §M3b  the escapes arm
    §M3c  the pre-check, VOID-IF-NULL, and forbidden as evidence of contraction
    §M3d  formula deciles
    §M3e  the priced feasibility and the declared fallback
"""

import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
sys.path.insert(0, HERE)

REGISTRATION_SHA16 = "3506032d552438e4"
COLUMN = os.path.join(CAMPAIGN, "results", "result_m_column.json")
#: PIN UPDATED UNDER [3828], protocol [3830], upstream hash posted at [3840].
#: The superseded value STAYS: it is the only record that a swap happened.
#: AND THE REASON THE COLUMN MOVED AT ALL IS WORTH THE LINE — an import analysis
#: said M was independent of the repair (no `movement`, no `decompose`, no
#: `cell_roles` in m_column.py) and it was WRONG: M reads `c.pre.probs`, which
#: comes through `word_probs()`. **A dependency graph over IMPORT STATEMENTS
#: does not see a shared ACCESSOR.** 82% of rows changed; sized diff: REAL none.
COLUMN_SHA16_SUPERSEDED = "dec26603e9ac1826"   #: pre-repair, escrowed at [3840]
COLUMN_SHA16 = "daf11fc743456f42"   #: post-repair re-run, [3840]
CLUSTER_FLOOR = 20                       #: §M3a


# ══════════════════════════════════════════════════════════════════════════
# the rank machinery -- DEFINITION, not library call (§M3a, [3686].3)
# ══════════════════════════════════════════════════════════════════════════
def midranks(v):
    """Ranks with ties averaged. The whole tie correction lives here."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman_tc(x, y):
    """Tie-corrected Pearson on midranks. None when either side is constant."""
    rx, ry = midranks(x), midranks(y)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None                      #: §M3a UNDEFINED RHO
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return sxy / math.sqrt(sxx * syy)


def exact_ranksum_p(s_pos, s_neg):
    """EXACT two-sided rank-sum p. §M3a's frozen calibration rule.

    **A Spearman with a BINARY arm IS the rank-sum statistic**, so its null is
    closed and exact at every k. The asymptotic Spearman p runs ~36%
    anticonservative at k = 1 -- in exactly the thin families, in the direction
    that manufactures significance. No asymptotic p enters the Stouffer.
    """
    from scipy.stats import mannwhitneyu
    if not s_pos or not s_neg:
        return None
    try:
        return float(mannwhitneyu(s_pos, s_neg, alternative="two-sided",
                                  method="exact").pvalue)
    except ValueError:
        #: §M3a EXACT-TIE REVERSION -- `s` is continuous so this is
        #: measure-zero; if it fires the count PRINTS.
        return ("REVERTED", float(mannwhitneyu(
            s_pos, s_neg, alternative="two-sided",
            method="asymptotic").pvalue))


def _ppf(q):
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


def z_from(p, rho):
    """Signed z from the EXACT p, sign from rho. §M3a."""
    if p is None or rho is None:
        return None
    if rho == 0:
        return 0.0
    return abs(_ppf(p / 2)) * (1 if rho > 0 else -1)


def stouffer(zs):
    zs = [z for z in zs if z is not None]
    if not zs:
        return None, None, 0
    Z = sum(zs) / math.sqrt(len(zs))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(Z) / math.sqrt(2))))
    return Z, p, len(zs)


# ══════════════════════════════════════════════════════════════════════════
def load():
    blob = open(COLUMN, "rb").read()
    got = hashlib.sha256(blob).hexdigest()[:16]
    if got != COLUMN_SHA16:
        raise SystemExit(f"REFUSING: column hashes {got}, expected "
                         f"{COLUMN_SHA16}. The test does not run on "
                         "unidentified cells.")
    print(f"column gate PASSED; result_m_column.json OBSERVED {got}")
    d = json.loads(blob)
    return d["rows"] if isinstance(d, dict) else d


def base_of_family():
    sys.path.insert(0, os.path.join(os.path.dirname(CAMPAIGN), "scripts"))
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(CAMPAIGN)), "scripts"))
    import m01_concentration as CC
    _p, models, _h, _d = CC.frozen_population()
    edges, _ = CC.operation_edges(models)
    return {fam: (getattr(st.pre, "id", None) or str(st.pre))
            for fam, _pos, st in edges}


def arm(rows, side):
    """side='overshoot' -> s = margin - d_null, words the null says SURVIVE.
       side='escape'    -> s = d_null - margin, words the null says are EVICTED.
    """
    per = {}
    for r in rows:
        s = (r["margin"] - r["d_null"]) if side == "overshoot" \
            else (r["d_null"] - r["margin"])
        if s <= 0:
            continue
        per.setdefault(r["family"], {"s": [], "y": []})
        per[r["family"]]["s"].append(s)
        per[r["family"]]["y"].append(1 if r["gold_evicted"] else 0)
    return per


def run_arm(rows, side, fb):
    per = arm(rows, side)
    fam, reverted, undefined = {}, 0, []
    for f, d in sorted(per.items()):
        rho = spearman_tc(d["y"], d["s"])
        if rho is None:
            undefined.append(f)
            fam[f] = {"n": len(d["s"]), "k": sum(d["y"]), "rho": None, "z": None}
            continue
        pos = [s for s, y in zip(d["s"], d["y"]) if y == 1]
        neg = [s for s, y in zip(d["s"], d["y"]) if y == 0]
        p = exact_ranksum_p(pos, neg)
        if isinstance(p, tuple):
            reverted += 1
            p = p[1]
        fam[f] = {"n": len(d["s"]), "k": sum(d["y"]), "rho": rho,
                  "p_exact": p, "z": z_from(p, rho)}
    clust = {}
    for f, rec in fam.items():
        if rec["z"] is None:
            continue
        clust.setdefault(fb.get(f, f"?{f}"), []).append(rec["z"])
    cz = {b: sum(v) / len(v) for b, v in clust.items()}
    Z, p, k = stouffer(list(cz.values()))
    return {"side": side, "families": fam, "n_families_with_z": len(fam) - len(undefined),
            "undefined": undefined, "reverted_to_asymptotic": reverted,
            "clusters": cz, "n_clusters": k, "Z": Z, "p": p,
            "UNDERPOWERED": k < CLUSTER_FLOOR}


def precheck(rows, fb):
    """§M3c. McNemar observed-vs-null eviction. VOID-IF-NULL, and FORBIDDEN
    as evidence of contraction: R counts LOSSES ONLY and the null spreads that
    mass over gainers too, so b > c under ANY non-uniform perturbation."""
    per = {}
    for r in rows:
        obs = 1 if r["gold_evicted"] else 0
        nul = 1 if r["margin"] < r["d_null"] else 0
        d = per.setdefault(r["family"], {"b": 0, "c": 0})
        if obs and not nul:
            d["b"] += 1
        elif nul and not obs:
            d["c"] += 1
    out = {}
    for f, d in per.items():
        n = d["b"] + d["c"]
        if n == 0:
            out[f] = {"b": 0, "c": 0, "z": None}
            continue
        k = min(d["b"], d["c"])
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
        z = 0.0 if d["b"] == d["c"] else abs(_ppf(p / 2)) * (1 if d["b"] > d["c"] else -1)
        out[f] = {"b": d["b"], "c": d["c"], "p": p, "z": z}
    clust = {}
    for f, rec in out.items():
        if rec["z"] is None:
            continue
        clust.setdefault(fb.get(f, f"?{f}"), []).append(rec["z"])
    cz = {b: sum(v) / len(v) for b, v in clust.items()}
    Z, p, k = stouffer(list(cz.values()))
    return {"families": out, "n_clusters": k, "Z": Z, "p": p}


def deciles(rows):
    """§M3d. EQUAL-n deciles of s over the pooled s > 0 population."""
    pts = [(r["margin"] - r["d_null"], 1 if r["gold_evicted"] else 0)
           for r in rows if (r["margin"] - r["d_null"]) > 0]
    pts.sort()
    n = len(pts)
    out = []
    for i in range(10):
        lo, hi = i * n // 10, (i + 1) * n // 10
        chunk = pts[lo:hi]
        if not chunk:
            continue
        out.append({"decile": i + 1, "n": len(chunk),
                    "s_lo": chunk[0][0], "s_hi": chunk[-1][0],
                    "evicted": sum(y for _, y in chunk),
                    "rate": sum(y for _, y in chunk) / len(chunk)})
    return out


RESULT = os.path.join(CAMPAIGN, "results", "result_m_primary.json")
ESCROW = os.path.join(CAMPAIGN, "results", "superseded")


def emit(pc, arms, bands):
    """WRITE THE ARTIFACT. [3849]: this producer computed and printed and wrote
    NOTHING, so the artifact of record on disk kept declaring `_column`
    `dec26603e9ac1826` — an input that had been superseded — permanently.

    **AN ARTIFACT DECLARING AN INPUT HASH THAT NO LONGER EXISTS ON DISK IS A
    SELF-EVIDENT STALENESS MARKER**, and it is only self-evident if someone
    re-emits when the input moves. A run whose output lives in a terminal is a
    run nobody else can test — the same sentence `run_l_found_prose.py` opens
    with, and this file was the counter-example to it.

    Escrow BEFORE write, read-only, per [3830].
    """
    payload = {
        "_what": "Registration M's tests (§M3-§M3e). Frozen text %s." % REGISTRATION_SHA16,
        "_registration": REGISTRATION_SHA16,
        "_column": COLUMN_SHA16,
        "_column_superseded": COLUMN_SHA16_SUPERSEDED,
        "_denominator_note": "3,610 rows; one cell refused at lambda <= 0 per "
                             "§M2 -- the declared refusal FIRED",
        "_sign": "rho NEGATIVE = evictions concentrate at LOW headroom = "
                 "BOUNDARY BLUR (§M4). Contraction predicts evictions "
                 "PERSISTING at large s.",
        "precheck": pc, "overshoot": arms["overshoot"],
        "escapes": arms["escape"], "bands": bands,
    }
    if os.path.exists(RESULT):
        os.makedirs(ESCROW, exist_ok=True)
        prior = open(RESULT, "rb").read()
        h = hashlib.sha256(prior).hexdigest()[:16]
        dst = os.path.join(ESCROW, "result_m_primary.PREFIX-%s.json" % h)
        if not os.path.exists(dst):
            with open(dst, "wb") as fh:
                fh.write(prior)
            os.chmod(dst, 0o444)
        print(f"  escrowed prior artifact @ {h} -> {os.path.basename(dst)}")
        #: THE ARTIFACT IS chmod a-w AND THAT IS THE POINT. Unlock AFTER the
        #: escrow exists and never before: if the write fails between unlock and
        #: lock, the escrow is the copy that survives. Announced, never silent —
        #: an unlock nobody sees is the lock not being there.
        print(f"  UNLOCKING {os.path.basename(RESULT)} for the re-emit "
              f"(escrow already read-only)")
        os.chmod(RESULT, 0o644)
    with open(RESULT, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    os.chmod(RESULT, 0o444)
    print(f"  wrote {os.path.basename(RESULT)} @ "
          f"{hashlib.sha256(open(RESULT,'rb').read()).hexdigest()[:16]}  "
          f"_column {COLUMN_SHA16}  RE-LOCKED a-w")


if __name__ == "__main__":
    rows = load()
    fb = base_of_family()
    print(f"  rows {len(rows)}   (denominator 3,610 -- one cell refused at "
          f"lambda <= 0, §M2)\n")

    pc = precheck(rows, fb)
    print(f"PRE-CHECK (§M3c, VOID-IF-NULL, NOT evidence of contraction)")
    print(f"  clusters {pc['n_clusters']}   Stouffer Z {pc['Z']:+.4f}   p {pc['p']:.6g}")
    if pc["p"] is not None and pc["p"] > 0.05:
        print("  *** NULL: the observed perturbation IS a uniform shrink. "
              "The exercise is VOID (§M3c). ***")

    arms = {}
    for side in ("overshoot", "escape"):
        r = run_arm(rows, side, fb)
        arms[side] = r
        label = "PRIMARY -- OVERSHOOT" if side == "overshoot" else "ESCAPES (§M3b)"
        print(f"\n{label}")
        print(f"  families with a z {len([f for f,v in r['families'].items() if v['z'] is not None])}"
              f"   undefined {len(r['undefined'])}"
              f"   reverted-to-asymptotic {r['reverted_to_asymptotic']}")
        print(f"  clusters {r['n_clusters']} (floor {CLUSTER_FLOOR})"
              f"   Stouffer Z {r['Z']:+.4f}   p {r['p']:.6g}"
              f"{'   *** UNDERPOWERED ***' if r['UNDERPOWERED'] else ''}")
        if side == "overshoot":
            print("  per-family eviction counts (k) beside n, [3683].4:")
            for f, v in sorted(r["families"].items(), key=lambda kv: -kv[1]["k"])[:8]:
                rho_s = "  --  " if v["rho"] is None else f"{v['rho']:+.4f}"
                print(f"    {f:<24} n {v['n']:>3}  k {v['k']:>3}  rho {rho_s}")

    bands = deciles(rows)
    print(f"\nBAND TABLE (§M3d, equal-n deciles of s):")
    print(f"  {'dec':>4}{'n':>6}{'s_lo':>9}{'s_hi':>9}{'evict':>7}{'rate':>8}")
    for b in bands:
        print(f"  {b['decile']:>4}{b['n']:>6}{b['s_lo']:>9.3f}{b['s_hi']:>9.3f}"
              f"{b['evicted']:>7}{100*b['rate']:>7.2f}%")
    print("\nNO MDE (§M4). Read the rho WITH the band table: rho alone cannot "
          "separate 'evictions at all headrooms' from 'no evictions at all'.")

    print("\nARTIFACT (§[3849] — this producer used to write nothing):")
    emit(pc, arms, bands)
