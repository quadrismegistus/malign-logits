#!/usr/bin/env python3
"""Registration E producer — H2 on the GAP stratum.

FROZEN SPEC: registration_e_gap_v3.md  6b58842efad50e90  (commit 2807e3a)
DELTA ON:    registration_c_delta_v6.md 06f0272d7f21b901

    .venv/bin/python scripts/m01_registration_e.py

NOT YET AUDITED. Per [1825] this is written, then code-audited by lacan, then
cleared by the pen, and only then run. It has NOT been run against real data.

THE GATE IS STRUCTURAL, NOT DISCIPLINARY. `require_frozen_spec()` returns the
spec hash and every emitting function DEMANDS that hash as an argument, verifying
it again before it prints. A caller who has not passed the gate cannot obtain the
token, so the family decomposition is unprintable rather than merely forbidden
([1809].4). This is why the token is threaded rather than checked once in main().

TWO PLACES WHERE THIS PRODUCER DEPARTS FROM C's CODE, BOTH BECAUSE THE SPEC SAYS
SO, AND BOTH FLAGGED FOR THE AUDIT RATHER THAN DECIDED QUIETLY:

  (1) THE BENCHMARK IS CELL-AVERAGED, AND THIS IS A BOOKED CORRECTION LANDING,
      NOT A DRIFT. [1592].1 measured the mismatch and [1594].1 ruled it: C's
      benchmark POOLED every word of every cell into one list while the arm it
      benchmarks CELL-AVERAGES. Pooled ran ~20% HIGH throughout, so every C arm
      faced a bar that was too strict — CONSERVATIVE, which is why no C verdict
      moved. It was booked as a producer amendment for the next touch. E IS THE
      NEXT TOUCH and §E3's "arm's own estimator, cell-averaged, never pooled" is
      that amendment written into a spec deliberately.
      **E's AND C's BENCHMARKS ARE NOT COMPUTED THE SAME WAY. E's is the corrected
      one.** A reader comparing the two registrations must not have to discover
      that, so the declaration line says it ([1827].1).

  (2) THE GLOBAL FIT'S POPULATION IS AMBIGUOUS AND BOTH ARE PRINTED. §E6 inherits
      v6 §C0's "GLOBAL over the qualifying population, never within cell." E's
      declared population is the GAP stratum, so the natural reading is a fit over
      the gap. C fitted over ALL strata pooled. RULED AT [1829], REVERSING
      [1827].2: C's ALL-STRATA FIT IS PRIMARY. Three reasons, and the first is
      decisive on its own:
        - COMPARABILITY. §E3's reading rule instructs comparison against "the
          effect size C measured (0.0251 displacing / 0.0340 gap, both from C)".
          Those were produced under C's all-strata fit. Residualise E differently
          and E's number is not on that scale — the spec's own reading rule
          becomes unexecutable, and nothing in the spec says so.
        - E IS A DELTA. §E0: C's v6 "governs everything not named here." §E1 says
          "GLOBAL fit" and does NOT name a population, so it INHERITS. A delta
          that silently changes an inherited parameter is not a delta.
        - LEAKAGE. Fitting arousal-to-valence ON the gap and then testing valence
          ON the gap absorbs gap-specific arousal/valence structure into "the
          confound" and subtracts it. That is the base-axis pattern — 8/21
          reversal without leave-one-out, 11/21 nothing with it — and it pointed
          where the other results pointed, which is what made it believable.
      The superseded "fit where you correct" argument is not wrong in isolation;
      it is wrong against this project's history. The gap fit still prints as the
      declared sensitivity.
"""

import argparse
import collections
import hashlib
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np

import m01_registration_b as B
import m01_registration_c3 as C3
from m01_registration_e_gate import require_frozen_spec, SPEC_SHA256

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINEAGE_MAP = os.path.join(ROOT, "data", "lineage_map.json")
LINEAGE_SETTING = "siblings_merged"        #: §E4, the principled setting. NAMED.
DIM, VARIANT = "valence", "extremity"      #: H2
K_THRESHOLDS = (20, 30)                    #: §E4 SECONDARY, both declared in the spec
ALPHA = 0.05                               #: one-sided, §E1 SIDEDNESS

#: THE CONTESTED PARAMETER, NAMED SO THE SWAP IS ONE LINE AND SO IT CANNOT BECOME
#: UNDECLARED. [1827].2 ruled GAP ("fit where you correct"); [1828] ruled ALL
#: ("E is a delta and does not name the fit population; and §E3's reading rule
#: compares E's number to C's 0.0251/0.0340, which were produced under C's
#: all-strata fit -- a different fit puts E on a different scale and makes the
#: spec's own reading rule unexecutable"). BOTH ARE COMPUTED AND BOTH PRINT
#: whichever is primary. UNRESOLVED at the time of writing; the pen adjudicates.
FIT_POPULATION = "all"          #: "all" (C's, per [1828]) or "gap" (per [1827].2)


def arm_rng(arm):
    """§E6: per-arm seed from sha256 of the arm name, NEVER builtin hash().

    `hash()` is salted per process, so a seed derived from it is not reproducible
    across runs — the defect this rule exists to prevent.
    """
    h = int(hashlib.sha256(arm.encode()).hexdigest()[:8], 16)
    return np.random.default_rng([B.SEED, h])


def _check(token):
    """Every emitter re-verifies. The token cannot be forged by a caller who
    skipped the gate, because obtaining it IS passing the gate."""
    if token != SPEC_SHA256:
        raise RuntimeError(
            "EMIT REFUSED: no valid frozen-spec token. The gap's family "
            "decomposition is structurally unprintable until require_frozen_spec() "
            "has passed ([1809].4). This is not a check you may skip.")


def cell_A(cell, coef, dim=DIM, variant=VARIANT):
    vals = [C3.value_of(z, dim, variant, coef) for z in cell["z"]]
    return C3.A_and_terms(vals, cell["w"], cell["roles"])


def benchmark_cell_averaged(cells, coef):
    """§E3: the benchmark uses THE ARM'S OWN ESTIMATOR — cell-averaged.

    A_arousal is computed per cell and averaged, exactly as run_general averages
    A over cells. C's main() pools instead; see the module docstring, departure (1).
    """
    per = []
    for c in cells:
        t = C3.A_and_terms([z["arousal"] for z in c["z"]], c["w"], c["roles"])
        if t is not None:
            per.append(t["A"])
    A_ar = st.mean(per) if per else float("nan")
    b = coef[1] * A_ar + (coef[2] * A_ar * A_ar if len(coef) > 2 else 0.0)
    return A_ar, b, len(per)


def realized_mde(null, alpha=ALPHA, power=0.80):
    """The effect this arm COULD have detected, from its own null's spread.

    §E3: a null at an MDE above the effect C measured reads UNINFORMATIVE, not
    negative. That sentence is unusable without this number, so it is computed
    here rather than left to the reader.
    """
    from scipy import stats
    crit = float(np.quantile(null, 1 - alpha))
    return crit + stats.norm.ppf(power) * float(np.std(null, ddof=1))


def pooled_arm(cells, coef_fitted, token, label):
    """One arm. `label` is "raw" or "residualised" and it selects BOTH the value
    function and the bar, which are coupled.

    C's convention, read from m01_registration_c3.main() rather than inferred:

        run_general(..., coef, ..., 0.0 if coef is not None else bench)

        RAW           values UNRESIDUALISED, bar = the AROUSAL-INDUCED benchmark
        RESIDUALISED  values RESIDUALISED,   bar = ZERO

    and C's own comment: "raw-beats-benchmark and residualised-beats-zero are ONE
    test." They are two expressions of one question — does the faller/riser gap
    in valence-extremity exceed what arousal alone would induce — and the bar
    moves to zero precisely because residualising has already removed the thing
    the benchmark represents.

    The first version of this function applied the fitted benchmark to BOTH arms.
    That put the residualised arm against a bar the residualisation had already
    subtracted (too strict, and it printed "beats benchmark: False" on a value
    that clears its real bar), and it crashed the raw arm, whose coef is None.
    """
    _check(token)
    rng = arm_rng(f"pooled:{label}")
    A_ar, bench, n_ar = benchmark_cell_averaged(cells, coef_fitted)
    if label == "raw":
        coef_used, bar = None, bench
    else:
        coef_used, bar = coef_fitted, 0.0
    res = C3.run_general(cells, DIM, VARIANT, coef_used, rng, B.N_PERM, bar)
    if res is None:
        return None
    res["A_arousal"] = A_ar
    res["induced_benchmark"] = bench
    res["n_arousal_cells"] = n_ar
    return res


def family_effects(cells, coef, token):
    """§E4 CO-PRIMARY: naive family means and prompt-adjusted family terms."""
    _check(token)
    rows = []
    for c in cells:
        t = cell_A(c, coef)
        if t is not None:
            rows.append((c["family"], c["prompt"], t["A"]))
    fams = sorted({f for f, _, _ in rows})
    prompts = sorted({p for _, p, _ in rows})

    naive = {}
    for f in fams:
        v = [a for ff, _, a in rows if ff == f]
        naive[f] = float(np.mean(v))

    fi = {f: i for i, f in enumerate(fams)}
    pi = {p: i for i, p in enumerate(prompts)}
    F, P = len(fams), len(prompts)
    X = np.zeros((len(rows), F + P))
    y = np.zeros(len(rows))
    for r, (f, p, a) in enumerate(rows):
        X[r, fi[f]] = 1.0
        X[r, F + pi[p]] = 1.0
        y[r] = a
    con = np.zeros((1, F + P))
    con[0, F:] = 1.0                       # prompt effects sum to zero
    beta, *_ = np.linalg.lstsq(np.vstack([X, con * 100.0]),
                               np.concatenate([y, [0.0]]), rcond=None)
    adjusted = {f: float(beta[fi[f]]) for f in fams}
    counts = collections.Counter(f for f, _, _ in rows)
    return rows, naive, adjusted, counts


def thresholded_core(rows, k, token):
    """§E4 SECONDARY at the two declared k. Diagnostic of DISCARDING."""
    _check(token)
    per_prompt = collections.defaultdict(set)
    for f, p, _ in rows:
        per_prompt[p].add(f)
    keep = {p for p, fs in per_prompt.items() if len(fs) >= k}
    sub = [(f, p, a) for f, p, a in rows if p in keep]
    fams = sorted({f for f, _, _ in sub})
    means = {f: float(np.mean([a for ff, _, a in sub if ff == f])) for f in fams}
    return keep, means


def sign_test(vals):
    v = [x for x in vals if x != 0]
    n, k = len(v), sum(1 for x in v if x > 0)
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0
    return k, n, p


def lineage_of(families, token):
    """§E4: family counts CITE the published map and NAME the setting."""
    _check(token)
    lm = json.load(open(LINEAGE_MAP))
    d = lm[LINEAGE_SETTING]
    f2l = {f: lin for lin, fams in d["lineages"].items() for f in fams}
    missing = [f for f in families if f not in f2l]
    return f2l, missing, d["n_lineages"]


def main(a):
    token = require_frozen_spec()          # THE GATE. Nothing above this reads data.
    print("REGISTRATION E — H2 on the GAP stratum")
    print(f"  SPEC    registration_e_gap_v3.md  {token}   FROZEN, gate passed")
    print(f"  DELTA ON registration_c_delta_v6.md 06f0272d7f21b901")
    print(f"  SIDEDNESS one-sided, A_|valence| > 0, C's confirmed direction")
    print(f"  NULL    membership, {B.N_PERM} draws, per-arm sha256-derived seeds\n")

    N, C = B._instrument()
    fp, fm, _h, drift = C.frozen_population()
    if drift:
        sys.exit(f"POPULATION DRIFT: {drift}")
    edges, _ = C.operation_edges(fm)
    norms, _f, _ = N.load_norms()
    cells, diag, n_moved, n_disp, n_ctrl = C3.collect(fp, edges, norms, N, C)
    gap = [c for c in cells if c["stratum"] == "gap"]

    # --- selection diagnostic, §E3 ALONGSIDE ------------------------------
    print("SELECTION DIAGNOSTIC")
    print(f"  registered prompts {len(fp)} | with movement {n_moved} | "
          f"displacing {n_disp} | control {n_ctrl} | gap {n_moved-n_disp-n_ctrl}")
    print(f"  GAP qualifying: {len(gap)} cells | "
          f"{len({c['prompt'] for c in gap})} prompts | "
          f"{len({c['family'] for c in gap})} families")
    for k, v in diag.most_common(5):
        print(f"      {v:>7}  {k}")

    # --- global fit: E's population, with C's as declared sensitivity ------
    def fit_over(cs):
        flat = [z for c in cs for z in c["z"]]
        return C3.fit([z["arousal"] for z in flat],
                      [abs(z[DIM] - C3.ORIGIN_Z) for z in flat], quad=True), len(flat)
    coef_gap, n_gap_words = fit_over(gap)
    coef_all, n_all_words = fit_over(cells)
    coef = coef_all if FIT_POPULATION == "all" else coef_gap
    print(f"\nGLOBAL AROUSAL FIT — primary is FIT_POPULATION={FIT_POPULATION!r}")
    print(f"  over ALL strata (as C) n={n_all_words:>6}  " +
          "  ".join(f"b{i}={v:+.4f}" for i, v in enumerate(coef_all)) +
          ("   <- PRIMARY" if FIT_POPULATION == "all" else "   (sensitivity)"))
    print(f"  over the GAP stratum   n={n_gap_words:>6}  " +
          "  ".join(f"b{i}={v:+.4f}" for i, v in enumerate(coef_gap)) +
          ("   <- PRIMARY" if FIT_POPULATION == "gap" else "   (sensitivity)"))

    # --- POOLED ARM, §E3 ---------------------------------------------------
    print("\n" + "=" * 70)
    print("POOLED ARM (§E3) — CONFIRMATORY IN DESIGN, SIGHTED IN FACT (§E0)")
    print("=" * 70)
    for label in ("raw", "residualised"):
        r = pooled_arm(gap, coef, token, label)
        if r is None:
            print(f"  {label}: below the {B.MIN_CELLS_TO_REPORT}-cell floor")
            continue
        print(f"  {label.upper():<14} A = {r['A']:+.4f}   n={r['n']} cells")
        print(f"     four numbers   M_f {r['Mf']:+.4f}  M_r {r['Mr']:+.4f}  "
              f"wmean_f {r['wf']:+.4f}  wmean_r {r['wr']:+.4f}")
        print(f"     null median {r['null']:+.4f}   p_up {r['p_up']:.4g}")
        print(f"     bar {r['benchmark']:+.4f}   "
              f"{'BEATS' if r['beats'] else 'does NOT beat'} it")
        if label == "raw":
            print(f"     the bar is the AROUSAL-INDUCED benchmark, from A_arousal "
                  f"{r['A_arousal']:+.4f}")
            print(f"     CELL-AVERAGED over {r['n_arousal_cells']} cells per §E3 "
                  f"(C pooled and ran ~20% high, [1594].1)")
        else:
            print(f"     the bar is ZERO because residualisation already removed "
                  f"the induced component ({r['induced_benchmark']:+.4f})")
        print("     raw-beats-benchmark and residualised-beats-zero are ONE test")

    # --- SCOPE ARM, §E4 ----------------------------------------------------
    print("\n" + "=" * 70)
    print("SCOPE ARM (§E4) — GENUINELY BLIND. Co-primary; agreement on SIGN ALONE.")
    print("=" * 70)
    rows, naive, adjusted, counts = family_effects(gap, coef, token)
    passing = [f for f in sorted(naive) if counts[f] >= B.MIN_CELLS_TO_REPORT]
    thin = [f for f in sorted(naive) if counts[f] < B.MIN_CELLS_TO_REPORT]
    print(f"  {len(passing)} families at the {B.MIN_CELLS_TO_REPORT}-cell floor; "
          f"{len(thin)} below and PRINTED, not dropped: {thin or 'none'}")

    f2l, missing, n_lin_total = lineage_of(passing, token)
    n_arm = len({f2l.get(f, f) for f in passing})
    print(f"  lineage map: data/lineage_map.json setting '{LINEAGE_SETTING}'")
    print(f"     ROSTER lineages      {n_lin_total}   (the map's own partition, "
          f"NOT this arm's denominator)")
    print(f"     lineages IN THIS ARM {n_arm}   <- the denominator any 'k of N' uses")
    if missing:
        print(f"     UNMAPPED, each counted as its OWN lineage ({len(missing)}), which "
              f"OVERSTATES independence: {missing}")

    print(f"\n  {'arm':<34}{'k of N':>10}{'p one-sided':>14}")
    results = {}
    for name, est in (("naive family mean", naive), ("prompt-adjusted", adjusted)):
        for unit, grp in (("family", lambda f: f), ("lineage", lambda f: f2l.get(f, f))):
            agg = collections.defaultdict(list)
            for f in passing:
                agg[grp(f)].append(est[f])
            vals = [float(np.mean(v)) for v in agg.values()]
            k, n, p = sign_test(vals)
            results[(name, unit)] = (k, n, p)
            note = (f"   [{len(missing)} unmapped counted as own lineage]"
                    if unit == "lineage" and missing else "")
            print(f"  {name + ', ' + unit + ' unit':<34}{f'{k} of {n}':>10}{p:>14.4g}{note}")

    for k_thr in K_THRESHOLDS:
        keep, means = thresholded_core(rows, k_thr, token)
        vals = [means[f] for f in sorted(means) if counts[f] >= B.MIN_CELLS_TO_REPORT]
        kk, nn, pp = sign_test(vals)
        print(f"  {'thresholded core k=' + str(k_thr):<34}{f'{kk} of {nn}':>10}"
              f"{pp:>14.4g}   ({len(keep)} prompts retained)")

    signs = {kx: (v[0] > v[1] / 2) for kx, v in results.items()}
    agree = len(set(signs.values())) == 1
    # "THE ARMS AGREE" IS MEANINGLESS IF A WIRING BUG MADE THEM ONE COMPUTATION.
    # §E4 warns that two estimators differing by 0.3 points are one estimator
    # counted twice; the worse case is that they ARE one by defect. Print their
    # actual divergence so agreement is a finding about estimators and not about
    # my plumbing.
    dv = np.array([adjusted[f] - naive[f] for f in passing])
    nv = np.array([naive[f] for f in passing])
    av = np.array([adjusted[f] for f in passing])
    print(f"\n  ESTIMATOR DIVERGENCE (are these two distinct computations?)")
    print(f"     max |adjusted - naive| {np.abs(dv).max():.5f}   "
          f"mean |diff| {np.abs(dv).mean():.5f}")
    print(f"     corr(naive, adjusted)  {np.corrcoef(nv, av)[0,1]:.6f}   "
          f"identical values: {np.allclose(nv, av)}")
    print(f"     families whose SIGN differs between them: "
          f"{int((np.sign(nv) != np.sign(av)).sum())}")

    print(f"\n  AGREEMENT ON SIGN ALONE (§E4): "
          f"{'ARMS AGREE' if agree else 'ARMS DISAGREE — scope claim is ESTIMATOR-DEPENDENT'}")
    if not agree:
        print("  §E4 pre-declared disagreement as the INFORMATIVE outcome: at the")
        print("  measured ICC these should not disagree, so this indicts a modelling")
        print("  assumption and not the finding. REPORT LOUDLY.")

    print("\nDECLARATION LINE")
    print(f"  spec {token} | producer {hashlib.sha256(open(__file__,'rb').read()).hexdigest()[:16]}"
          f" | map {LINEAGE_SETTING} | seed {B.SEED} | perms {B.N_PERM}")
    print("  BENCHMARK: CELL-AVERAGED, per [1594].1. Registration C's benchmark was")
    print("  POOLED and ran ~20% HIGH (a bar too strict, hence conservative). THE TWO")
    print("  REGISTRATIONS' BENCHMARKS ARE NOT COMPUTED THE SAME WAY and E's is the")
    print("  corrected one. Do not compare E's benchmark to C's as like for like.")
    # DERIVED FROM THE CONSTANT, NEVER RESTATED. A declaration line that spells out
    # a parameter in prose goes stale the moment the parameter moves -- which is
    # exactly what happened to this line when FIT_POPULATION was reversed to "all"
    # and the text still read "over the GAP stratum, per [1827].2".
    _fit = ("ALL STRATA, as Registration C (the INHERITED parameter: E is a delta "
            "and §E1 does not name a fit population)" if FIT_POPULATION == "all"
            else "the GAP stratum only")
    print(f"  AROUSAL FIT: FIT_POPULATION={FIT_POPULATION!r} -- {_fit}, per [1829]")
    print("  reversing [1827].2. The other fit is printed above as a declared sensitivity.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate-only", action="store_true",
                    help="verify the freeze gate and exit without reading anything")
    args = ap.parse_args()
    if args.gate_only:
        print(f"gate passes: {require_frozen_spec()}")
        sys.exit(0)
    sys.exit(main(args) or 0)
