#!/usr/bin/env python3
"""Registration C producer v3 — the role-membership battery, VECTORISED.

FROZEN SPEC: registration_c_delta_v6.md  06f0272d7f21b901  ([1559])
BASE:        registration_b_spec_v13.md  06186c42f9ff46e0

v3 supersedes v2 f2aeb28893b5198c: IDENTICAL STATISTICS, IDENTICAL NULLS, the
permutation loops moved from Python into numpy. v2 was audited-correct and
computationally intractable -- 48 permutation loops x 10,000 draws x thousands
of cells, recomputing every contrast in pure Python. Nothing about what is
measured changes; only how the same draws are generated.

Supersedes m01_registration_c.py f0fd1a1b285dabc3, which was written to v4 —
the wrong apparatus. v4's centred within-role statistic is ALGEBRAICALLY BLIND
to a class-level shift, so it would have returned null on a true hypothesis.

THE UNIT IS THE CELL, not the (cell, role). RH's hypotheses are about WHICH
WORDS FALL AND WHICH RISE, so the membership null must permute labels across
both roles of one cell -- which requires holding the cell whole.
"""
import argparse, collections, math, os, statistics as st, sys
import numpy as np

for _root in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
    if os.path.isfile(os.path.join(_root, "m01_registration_b.py")):
        sys.path.insert(0, _root); break
else:                                    # pragma: no cover
    sys.exit("needs m01_registration_b.py beside it")
import m01_registration_b as B

SPEC_SHA = "06f0272d7f21b901"
DIMS = ("valence", "dominance")
STRATA = ("displacing", "control", "gap")
TESTED = ("displacing", "control")
ORIGIN_Z = 0.0                      #: §C0. z is database-anchored.
MIN_PERMS = 20                      #: §C7. A null with too few attainable
                                    #: values is not a null. Refuse below this.

#: §C2. (dim, variant) -> (label, direction, arm)
REGISTERED = {
    ("valence",   "signed"):    ("H1", "down"),   # A_v NEGATIVE
    ("valence",   "extremity"): ("H2", "up"),     # A_|v| POSITIVE
    ("dominance", "extremity"): ("H3", "up"),     # A_|d| POSITIVE
}


def fit(xs, ys, quad):
    """§C0: GLOBAL over the qualifying population, never within cell."""
    X = np.column_stack([np.ones(len(xs)), xs] + ([np.asarray(xs) ** 2] if quad else []))
    return np.linalg.lstsq(X, np.asarray(ys), rcond=None)[0]


def value_of(z, dim, variant, coef):
    """§C4. ABSOLUTE VALUE FIRST for extremity, about the DECLARED ORIGIN --
    RH's hypothesis is distance from the scale's neutral point, not from a
    cell's mean. And the RESIDUALISED VARIABLE IS THE ONE THE HYPOTHESIS IS
    ABOUT: removing the signed coupling then folding leaves the extremity
    confound intact under a "residualised" label.
    """
    base = z[dim] if variant == "signed" else abs(z[dim] - ORIGIN_Z)
    if coef is None:
        return base
    a = z["arousal"]
    pred = coef[0] + coef[1] * a + (coef[2] * a * a if len(coef) > 2 else 0.0)
    return base - pred


def wmean(vals, wts):
    s = sum(wts)
    return sum(v * w for v, w in zip(vals, wts)) / s if s > 0 else float("nan")


def A_and_terms(vals, wts, roles):
    """§C6: FOUR numbers, not one. A alone recovers neither A nor Delta_E,
    and H1/H2 are told apart by the RISER TERM ALONE.
    """
    f = [(v, w) for v, w, r in zip(vals, wts, roles) if r == "faller"]
    r_ = [(v, w) for v, w, r in zip(vals, wts, roles) if r == "riser"]
    if not f or not r_:
        return None
    Mf, Mr = sum(w for _, w in f), sum(w for _, w in r_)
    wf = wmean([v for v, _ in f], [w for _, w in f])
    wr = wmean([v for v, _ in r_], [w for _, w in r_])
    return {"A": wf - wr, "Mf": Mf, "Mr": Mr, "wf": wf, "wr": wr,
            "dE": Mr * wr - Mf * wf}


def top_of(vals, wts, roles, want):
    """Top-movers arm: the centred value of the largest-|delta| word of a role."""
    s = [(v, w) for v, w, r in zip(vals, wts, roles) if r == want]
    if len(s) < B.QUALIFYING_MIN:
        return None
    return max(s, key=lambda x: x[1])[0] - st.mean(v for v, _ in s)


def n_perms(roles):
    """§C7: distinct label arrangements available in this cell."""
    nf = sum(1 for r in roles if r == "faller")
    return math.comb(len(roles), nf) if 0 < nf < len(roles) else 1


def collect(prompts, edges, norms, N, C):
    tabs = {d: norms[("en", d, "primary")] for d in ("arousal",) + DIMS}
    dep, raw, diag = collections.defaultdict(list), [], collections.Counter()
    for fam, pos, step in sorted(edges):
        for t in prompts:
            c = step.cell(t)
            if not c.is_present:
                diag["cell absent from the store (cut)"] += 1; continue
            if c.language != "en":
                diag["non-en, outside the declared population (cut)"] += 1; continue
            try:
                d = c.decompose(None)
            except RuntimeError:
                raise
            except Exception as e:
                diag[f"cell errored: {type(e).__name__} (code)"] += 1; continue
            if not d:
                diag["cell decomposed empty (data)"] += 1; continue
            dep[t].append(d["departed"]); raw.append((fam, t, c))

    disp = {t for t, v in dep.items() if v and st.median(v) >= N.DISPLACING_AT}
    ctrl = {t for t, v in dep.items() if v and st.median(v) < N.CONTROL_BELOW}

    cells = []
    for fam, t, c in raw:
        stratum = "displacing" if t in disp else "control" if t in ctrl else "gap"
        try:
            roles = N.cell_roles(c, C.RULE)
        except RuntimeError:
            raise
        except Exception as e:
            diag[f"roles errored: {type(e).__name__} (code)"] += 1; continue
        ws, zs, rs = [], [], []
        for w, wt, role in roles:
            k = N.norm_key(w, "en", fold=False)
            if N.is_function_word(k, "en"):
                diag["function word excluded (data)"] += 1; continue
            z = {}
            for dim in ("arousal",) + DIMS:
                v, _ = N.lookup(tabs[dim], k.casefold(), "en")
                z[dim] = v
            if any(v is None for v in z.values()):
                diag["missing a V/A/D rating (data)"] += 1; continue
            ws.append(wt); zs.append(z); rs.append(role)
        nf = sum(1 for r in rs if r == "faller")
        if nf < B.QUALIFYING_MIN or len(rs) - nf < B.QUALIFYING_MIN:
            diag[f"cell below the {B.QUALIFYING_MIN}-per-role bar (data)"] += 1
            continue
        cells.append({"family": fam, "prompt": t, "stratum": stratum,
                      "w": ws, "z": zs, "roles": rs})
    return cells, diag, len(dep), len(disp), len(ctrl)


def run_general(cells, dim, variant, coef, rng, n_perm, benchmark):
    """§C5 MEMBERSHIP NULL, VECTORISED. Permute the faller/riser LABEL within a
    cell, holding each word's value and BOTH ROLE SIZES fixed.

    The draws are identical to v2's Python loop: a permutation of the labels
    that preserves n_f is exactly a uniformly random SUBSET of size n_f, so
    per draw we choose n_f indices and sum over them. Permutations are
    independent across cells, so a cell's n_perm draws can be generated at once
    and accumulated -- which is what makes this tractable without changing the
    null.

        A = Sf/Wf - (S-Sf)/(W-Wf)

    so only the faller-side sums vary per draw; the rest is cell-constant.
    """
    terms, obs = [], []
    per_cell = []
    for c in cells:
        vals = np.asarray([value_of(z, dim, variant, coef) for z in c["z"]])
        w = np.asarray(c["w"], dtype=float)
        roles = np.asarray(c["roles"])
        t = A_and_terms(list(vals), list(w), list(roles))
        if t is None:
            continue
        terms.append(t); obs.append(t["A"])
        per_cell.append((vals * w, w, int((roles == "faller").sum())))
    if len(obs) < B.MIN_CELLS_TO_REPORT:
        return None
    A_obs = st.mean(obs)

    null = np.zeros(n_perm)
    for vw, w, nf in per_cell:
        n = len(w)
        S, W = vw.sum(), w.sum()
        r = rng.random((n_perm, n))
        idx = np.argpartition(r, nf - 1, axis=1)[:, :nf]      # uniform n_f-subset
        Sf = np.take_along_axis(np.broadcast_to(vw, (n_perm, n)), idx, 1).sum(1)
        Wf = np.take_along_axis(np.broadcast_to(w, (n_perm, n)), idx, 1).sum(1)
        Wr = W - Wf
        with np.errstate(invalid="ignore", divide="ignore"):
            null += np.where((Wf > 0) & (Wr > 0), Sf / Wf - (S - Sf) / Wr, 0.0)
    null /= len(per_cell)

    p_up = float((1 + int((null >= A_obs).sum())) / (1 + n_perm))
    p_dn = float((1 + int((null <= A_obs).sum())) / (1 + n_perm))
    return {"A": A_obs, "n": len(obs), "null": float(np.median(null)),
            "p_up": p_up, "p_dn": p_dn, "benchmark": benchmark,
            "beats": (A_obs < benchmark) if variant == "signed" else (A_obs > benchmark),
            "Mf": st.mean(t["Mf"] for t in terms), "Mr": st.mean(t["Mr"] for t in terms),
            "wf": st.mean(t["wf"] for t in terms), "wr": st.mean(t["wr"] for t in terms),
            "dE": st.mean(t["dE"] for t in terms),
            "minperm": min(n_perms(c["roles"]) for c in cells)}


def run_top(cells, dim, variant, coef, rng, n_perm, want):
    """§C5 MASS-ORDER NULL, VECTORISED. Permute |delta| WITHIN a role.

    The statistic depends on the weights ONLY through which word carries the
    largest one. Permuting the weights within a role places the maximum on a
    uniformly random position, so drawing a uniform index is the same null.
    (Exact for distinct weights; a tie would make argmax prefer the earlier
    index, a bias of order 1/n_perm on float weights.)
    """
    idxs, obs = [], []
    for c in cells:
        vals = [value_of(z, dim, variant, coef) for z in c["z"]]
        sel = [v for v, r in zip(vals, c["roles"]) if r == want]
        wts = [w for w, r in zip(c["w"], c["roles"]) if r == want]
        if len(sel) < B.QUALIFYING_MIN:
            continue
        mu = st.mean(sel)
        centred = np.asarray(sel) - mu
        obs.append(centred[int(np.argmax(wts))])
        idxs.append(centred)
    if len(obs) < B.MIN_CELLS_TO_REPORT:
        return None
    R = float(np.mean(obs))
    null = np.zeros(n_perm)
    for centred in idxs:
        null += centred[rng.integers(0, len(centred), size=n_perm)]
    null /= len(idxs)
    return {"R": R, "n": len(obs), "null": float(np.median(null)),
            "p_up": float((1 + int((null >= R).sum())) / (1 + n_perm)),
            "p_dn": float((1 + int((null <= R).sum())) / (1 + n_perm))}


def main(a):
    N, C = B._instrument()
    print("REGISTRATION C v2 — the role-membership battery")
    print(f"  SPEC      registration_c_delta_v6.md  {SPEC_SHA}  frozen [1559]")
    print("  SIDEDNESS one-sided directional per registered arm")
    print("  ORIGIN    database mean (z-anchored)")
    print("  NULLS     membership (general) / mass-order (top movers)")

    rng = np.random.default_rng(B.SEED)
    if not B.calibrate(rng):
        return 1
    if a.calibrate_only:
        print("\n  --calibrate-only: stopping. NO read of any kind has occurred.")
        return 0

    prompts, models, _hashes, drift = C.frozen_population()
    if drift:                            # [1521].1 -- the guard v1 dropped
        sys.exit(f"POPULATION DRIFT: {drift}. §C0 pins the population by hash; "
                 "a drifted population is not the registered one.")
    edges, _ = C.operation_edges(models)
    norms, _freqs, _ = N.load_norms()
    cells, diag, n_moved, n_disp, n_ctrl = collect(prompts, edges, norms, N, C)

    print(f"\n  POPULATION  {n_moved} prompts with movement; {n_disp} displacing, "
          f"{n_ctrl} control")
    print(f"  QUALIFYING  {len(cells)} cells with >= {B.QUALIFYING_MIN} rated "
          f"non-function words IN EACH ROLE")
    for k, v in diag.most_common():
        print(f"      {v:>7}  {k}")
    for s in STRATA:
        n = sum(1 for c in cells if c["stratum"] == s)
        print(f"      stratum {s:<11} {n:>6} cells"
              + ("" if s in TESTED else "   [§C8 PRINTED, NEVER TESTED]"))

    # --- §C0 global fits + §C3 per-stratum benchmarks -----------------------
    flat = [z for c in cells for z in c["z"]]
    Ar = [z["arousal"] for z in flat]
    coefs = {}
    print("\n  §C0 GLOBAL FITS (one per dim x variant, over the whole qualifying set)")
    for dim in DIMS:
        coefs[(dim, "signed")] = fit(Ar, [z[dim] for z in flat], quad=False)
        coefs[(dim, "extremity")] = fit(
            Ar, [abs(z[dim] - ORIGIN_Z) for z in flat], quad=True)
        for variant in ("signed", "extremity"):
            cf = coefs[(dim, variant)]
            print(f"    {dim:<10}{variant:<10} n={len(flat)}  " +
                  "  ".join(f"b{i}={v:+.4f}" for i, v in enumerate(cf)))
    print("    FOUR fits over the FULL population. A within-cell fit would print "
          "thousands\n    and would zero small cells silently ([1487].1).")

    for stratum in STRATA:
        sel = [c for c in cells if c["stratum"] == stratum]
        if len(sel) < B.MIN_CELLS_TO_REPORT:
            print(f"\n  ===== {stratum.upper()} — {len(sel)} cells, below the "
                  f"{B.MIN_CELLS_TO_REPORT} floor. UNDERPOWERED, NOT null. =====")
            continue
        nperm = a.perm if stratum in TESTED else a.perm_gap
        tag = ("" if stratum in TESTED
               else f"   [§C8 EXPLORATORY, NOT TESTED — {nperm} draws]")
        print(f"\n  ===== {stratum.upper()} — {len(sel)} cells ====={tag}")

        # §C3: the benchmark is PER STRATUM, from THIS stratum's own A_arousal
        av = [{"arousal": z["arousal"]} for c in sel for z in c["z"]]
        aw = [w for c in sel for w in c["w"]]
        ar_roles = [r for c in sel for r in c["roles"]]
        at = A_and_terms([x["arousal"] for x in av], aw, ar_roles)
        A_ar = at["A"] if at else float("nan")
        print(f"    A_arousal on THIS stratum = {A_ar:+.4f}   "
              f"(benchmarks below are computed FROM IT, not pooled)")

        for dim in DIMS:
            for variant in ("signed", "extremity"):
                reg = REGISTERED.get((dim, variant))
                cf = coefs[(dim, variant)]
                bench = cf[1] * A_ar + (cf[2] * A_ar * A_ar if len(cf) > 2 else 0.0)
                # §C8 WITHHELD WALL. ALL STRATA, not just the tested ones: the
                # gap value is the same quantity on the same population, and a
                # blind that leaks through an exploratory row is not a blind.
                # It suppresses the GENERAL arm ONLY -- H1's TOP-MOVERS arm was
                # never computed anywhere and malign is NOT blind to it.
                withheld = (dim == "valence" and variant == "signed"
                            and not a.show_h1_general)
                if withheld:
                    print(f"\n    {dim.upper()}/{variant.upper()} GENERAL — "
                          "[WITHHELD — malign holds this arm blind on the PAIRS "
                          "population]")
                    print("      §C8: emitted only under --show-h1-general, which "
                          "malign does not pass. The TOP-MOVERS arm below is NOT "
                          "withheld.")
                for kind, coef in ((("RAW", None), ("RESIDUALISED", cf))
                                   if not withheld else ()):
                    g = run_general(sel, dim, variant, coef, rng, nperm,
                                    0.0 if coef is not None else bench)
                    lbl = f"{dim}/{variant}/{kind}"
                    if g is None:
                        print(f"\n    {lbl:<34} below the reporting floor")
                        continue
                    tail = "p_dn" if (reg and reg[1] == "down") else "p_up"
                    print(f"\n    {lbl:<34}{'REGISTERED ' + reg[0] if reg else 'exploratory'}")
                    print(f"      A {g['A']:+.4f}   null {g['null']:+.4f}   "
                          f"{tail} {g[tail]:.4f}   cells {g['n']}")
                    print(f"      benchmark {g['benchmark']:+.4f}   "
                          f"{'BEATS' if g['beats'] else 'does NOT beat'} it"
                          + ("   (raw-beats-benchmark and residualised-beats-zero "
                             "are ONE test)" if coef is None else ""))
                    print(f"      M_f {g['Mf']:.4f}  M_r {g['Mr']:.4f}  "
                          f"wmean_f {g['wf']:+.4f}  wmean_r {g['wr']:+.4f}  "
                          f"Delta_E {g['dE']:+.4f}")
                    if g["minperm"] < MIN_PERMS:
                        print(f"      *** REFUSED: min per-cell arrangements "
                              f"{g['minperm']} < {MIN_PERMS}. A null with too few "
                              "attainable values is not a null. ***")
                for want in ("faller", "riser"):
                    t = run_top(sel, dim, variant, cf, rng, nperm, want)
                    if t:
                        print(f"      TOP-MOVERS {want:<7} R {t['R']:+.4f}  "
                              f"null {t['null']:+.4f}  p_up {t['p_up']:.4f}  "
                              f"p_dn {t['p_dn']:.4f}  cells {t['n']}")

    print("\n  §C6 READING RULE: H1 iff wmean_r EXCEEDS its null AND A beats its")
    print("  benchmark; H2 iff wmean_r does NOT exceed AND A_|v| beats its. Each")
    print("  hypothesis is INVISIBLE on the other's statistic — read the four")
    print("  numbers, never A alone. §C9: residualisation is the SOLE defence.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibrate-only", action="store_true")
    p.add_argument("--show-h1-general", action="store_true",
                   help="emit the WITHHELD arm. malign does not pass this.")
    p.add_argument("--perm", type=int, default=B.N_PERM)
    p.add_argument("--perm-gap", type=int, default=1000,
                   help="draws for the GAP stratum, which §C8 never tests. "
                        "Lower resolution there is not a spec deviation: no "
                        "registered claim reads it. Tested strata keep --perm.")
    sys.exit(main(p.parse_args()))
