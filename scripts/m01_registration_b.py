"""Registration B producer: the high-mass decomposition. Written TO v13.

    uv run .venv/bin/python scripts/m01_registration_b.py --calibrate-only
    uv run .venv/bin/python scripts/m01_registration_b.py            # THE RUN
    ... --csv out.csv    per cell-role rows

SPEC: `registration_b_spec_v13.md`, sha256 06186c42f9ff46e0, frozen [1412].
Thirteen versions; the attack log is the spec's own §10. **Every constant below
names the section it implements; a constant with no section is a defect.**

THE REGISTERED QUESTION (§0): is there arousal structure along the mass ordering
BEYOND what baseline probability explains?

SEEN-GIVENS, so nothing here is re-sold as a finding: the riser concentration
profile; the weighted/unweighted gap and its size; corr(arousal, log P) = +0.08.
BLIND AT EVERY SEAT until this file runs: the mass-conditional arousal profile.
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import math
import os
import statistics as st
import sys

# THE INSTRUMENT IS IMPORTED LAZILY, AT THE DATA PATH ONLY.
# `m01_norms` and `m01_concentration` both `sys.exit` AT IMPORT TIME when the package
# is unreachable, so a module-level import made `--calibrate-only` impossible to run
# without the store — no arrangement of statements inside main() could have reached
# the gate, because the file could not be loaded. §8 is pure simulation and needs
# nothing from either module; it must be auditable on a machine with no data.
def _instrument():
    for _root in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        if os.path.isfile(os.path.join(_root, "m01_norms.py")):
            sys.path.insert(0, _root)
            break
    else:                                    # pragma: no cover - environment failure
        sys.exit("m01_norms.py must sit beside this file for the data path; "
                 "--calibrate-only does not need it")
    import m01_norms as N
    import m01_concentration as C
    # SEED is declared literally below so calibration needs no import. If the
    # instrument IS present, the two must agree or the producers have diverged.
    if N.PERM_SEED != SEED:                  # pragma: no cover - would be real
        sys.exit(f"SEED DIVERGENCE: this producer pins {SEED}, m01_norms pins "
                 f"{N.PERM_SEED}. The nulls would not be comparable.")
    return N, C

# --- §2 UNIT, RANKING, QUALIFYING -------------------------------------------
QUALIFYING_MIN = 3       #: §2. A 2-word role is PERMUTATION-INVARIANT under this
                         #: statistic — (a,-a) gives max|C|=|a| in either order — so it
                         #: adds the same constant to observed and null. Zero information
                         #: at any sample size, not merely low power.
RANKING = "|delta| = |Q - P|, descending, BOTH ARMS"
#: §2. ~81% a probability ranking (+0.830 faller / +0.600 riser). Used anyway: the
#: confound needs two links and the second is absent, corr(arousal, log P) = +0.08.
#: Ranking risers by excess inverts the coupling to -0.964 and is worse.

MIN_CELLS_TO_REPORT = 20 #: §4 IMPLEMENTATION, declared because it silently dropped
                         #: a stratum and a role before malign named it. Below this,
                         #: a stratum or role prints UNDERPOWERED with its count --
                         #: uninformative, NOT null. A cut that is not printed is not
                         #: declared.
REOPEN_BAR = 0.15        #: §2. |corr(arousal, log P)| >= this REOPENS the ranking
                         #: question. WIRED as a run-time comparison below, not prose:
                         #: 0.15 separates the observed cases (arousal 0.08 inert,
                         #: concreteness 0.177 live).

# --- §3 STATISTIC AND NULL ---------------------------------------------------
N_PERM = 10000           #: §3
MIN_PAIRS_FOR_REOPEN = 100   #: §2 -- the reopen bar needs this many
                             #: (word, P) pairs before a correlation on
                             #: them means anything. Was an undeclared
                             #: literal until [1440].3(a).
MIN_WORDS_FOR_FREQ = 100     #: §7(b), same reasoning, same former literal
SEED = 20260731          #: §3. DECLARED LITERALLY so §8 runs without the instrument;
                         #: `_instrument()` asserts it equals m01_norms.PERM_SEED
                         #: whenever the data path loads, so divergence still cannot
                         #: pass silently.
EXACT_ENUM_MAX = 6       #: §3. M_cell exact by enumeration at n <= 6 (<= 720 orderings);
                         #: estimated from draws above.

# --- §8 CALIBRATION ----------------------------------------------------------
GATE_TOL = 0.02          #: §8. E[s | n, spread] = 1.000 +/- this
GATE_K_MIN = 3.0         #: §8 PRECONDITION. The grid must place some spread at
                         #: least this many tolerances from 1.0, or the known-bad
                         #: control cannot fire and the gate stops discriminating
                         #: SILENTLY. Currently 26.5x — but that is a property of the
                         #: population's spread deciles, and populations move.
GATE_MIN_CORPORA = 4000  #: §8 IMPLEMENTATION, not a spec parameter — and it is not
                         #: tuning. The gate's TOLERANCE and its SAMPLE SIZE must be
                         #: matched: s has CV ~0.26 at n=6, so the standard error of
                         #: E[s] is 0.26/sqrt(N). At N=120 that is 0.024 — LARGER than
                         #: the +/-0.02 tolerance, so a correct statistic fails roughly
                         #: half the grid BY NOISE. Measured at n=6, spread 0.666:
                         #:     N=120  E[s] 1.0026  SE 0.0233
                         #:     N=480       0.9953     0.0118
                         #:     N=2000      1.0111     0.0057
                         #:     N=8000      1.0017     0.0029
                         #: It converges on 1.000; the first run's FAILs were my sample,
                         #: not the statistic. 4,000 gives SE ~0.004, five times inside
                         #: the tolerance. **A GATE WHOSE STANDARD ERROR EXCEEDS ITS
                         #: TOLERANCE TESTS THE SAMPLE SIZE, NOT THE STATISTIC.**
GATE_SIZES = (3, 4, 6, 10, 20, 40)                       #: §8
GATE_SPREADS = (0.470, 0.666, 0.842, 1.036, 1.252)       #: §8. The POPULATION's own
                         #: spread deciles, p10 through p90, measured on 8,340 qualifying
                         #: cell-roles. NOT invented: a gate is tested where the data
                         #: lives. A size-only grid would certify E_n.


def m_cell(vals, rng, n_draws=400):
    """§3. The cell's OWN permutation mean of max|cumsum|.

    Exact by enumeration at n <= EXACT_ENUM_MAX; estimated above. This is the divisor
    the spec adopted after four candidates: `sqrt(n)` mis-weighted by SIZE, `E_n` by
    SPREAD (2.66x across the population's own deciles), and `sd_cell x E_n_unit` is a
    Jensen-biased-low ESTIMATOR of this very quantity — the two correlate at 0.9986
    and are one thing, one computed and one estimated from n points.
    """
    n = len(vals)
    if n <= EXACT_ENUM_MAX:
        tot = 0.0
        cnt = 0
        for p in itertools.permutations(vals):
            c = 0.0
            m = 0.0
            for x in p:
                c += x
                if abs(c) > m:
                    m = abs(c)
            tot += m
            cnt += 1
        return tot / cnt
    import numpy as np
    v = np.asarray(vals, dtype=float)
    return float(np.mean([np.abs(np.cumsum(rng.permutation(v))).max()
                          for _ in range(n_draws)]))


def cusum_max(vals):
    """§3. max_j |C_j| over the given ORDER."""
    c = 0.0
    m = 0.0
    for x in vals:
        c += x
        if abs(c) > m:
            m = abs(c)
    return m


def calibrate(rng, n_corpora=GATE_MIN_CORPORA, n_perm=600):
    """§8. RUNS UNCONDITIONALLY IN THE MAIN PATH, never behind a flag.

    THE GATE CHECKS THE IMPLEMENTATION, NOT THE CHOICE. Under M_cell the pass is an
    IDENTITY — E[s] = 1.000 holds by construction — so a pass certifies only that the
    code computes what the spec says. **THE INFORMATIVE HALF IS THE E_n FIRE**: that
    the gate DISCRIMINATES is what makes it evidence rather than tautology, which is
    why E_n is retained here as the known-bad control.
    """
    # ===== THE PRECONDITION, ASSERTED BEFORE ANYTHING RUNS =================
    # E[s_En] = sigma EXACTLY, so whether the E_n column fires is DETERMINED BY THE
    # GRID, not observed from it. Crediting the fire as evidence credits an identity
    # -- the third such number this arc has retired. What is worth asserting is that
    # the grid still PLACES a spread far enough from 1.0 to make the fire unambiguous:
    # if the population's spreads ever tighten toward 1.0, the gate silently stops
    # discriminating and its pass would mean nothing. This fails loudly instead.
    margin = max(abs(sd - 1.0) for sd in GATE_SPREADS)
    k = margin / GATE_TOL
    print(f"\n  §8 PRECONDITION — the grid must be able to make E_n fire.")
    print(f"    max |spread - 1.0| = {margin:.3f} = {k:.1f}x the {GATE_TOL} tolerance"
          f"   (require >= {GATE_K_MIN}x)")
    if k < GATE_K_MIN:                        # pragma: no cover - would be real
        sys.exit(f"§8 CANNOT DISCRIMINATE: the spread grid spans only {margin:.3f} "
                 f"from 1.0, {k:.1f}x the tolerance. E[s_En] = spread, so a grid this "
                 f"tight cannot fire the known-bad control and the gate's pass would "
                 f"certify nothing. Re-measure the population's spread deciles.")

    import numpy as np
    EnU = {n: float(np.mean([np.abs(np.cumsum(x - x.mean())).max()
                             for x in rng.normal(0, 1, (4000, n))]))
           for n in GATE_SIZES}
    print("\n  §8 CALIBRATION — gate E[s|n,spread] = 1.000 +/- "
          f"{GATE_TOL}, sizes x POPULATION SPREAD DECILES")
    print(f"    {'n':>4}{'spread':>9}{'E[s] M_cell':>14}{'gate':>7}"
          f"{'E[s] E_n':>11}{'fires?':>8}")
    if n_corpora < GATE_MIN_CORPORA:
        # REFUSE rather than report a noisy verdict as a real one.
        sys.exit(f"§8 requires >= {GATE_MIN_CORPORA} corpora per grid cell: the "
                 f"tolerance is +/-{GATE_TOL} and the SE at {n_corpora} would be "
                 f"~{0.26 / (n_corpora ** 0.5):.3f}. A gate cannot be finer than its "
                 f"own sampling error.")
    ok = True
    fired = False
    for n in GATE_SIZES:
        for sd in GATE_SPREADS:
            sm, se = [], []
            for _ in range(n_corpora):
                v = rng.normal(0, sd, n)
                v = v - v.mean()
                obs = cusum_max(v)
                sm.append(obs / m_cell(list(v), rng, 250))
                se.append(obs / EnU[n])
            mm, me = float(np.mean(sm)), float(np.mean(se))
            passes = abs(mm - 1.0) <= GATE_TOL
            fires = abs(me - 1.0) > GATE_TOL
            ok &= passes
            fired |= fires
            print(f"    {n:>4}{sd:>9.3f}{mm:>14.3f}{'PASS' if passes else 'FAIL':>7}"
                  f"{me:>11.3f}{'FIRES' if fires else '-':>8}")
    print(f"\n    M_cell passes every cell: {ok}")
    print(f"    E_n fires somewhere:      {fired}")
    print("    §8 IS A DIFFERENTIAL IMPLEMENTATION CHECK. Its content is TWO COLUMNS")
    print("    MATCHING TWO INDEPENDENTLY PREDICTED SHAPES — M_cell flat at 1.000,")
    print("    E_n tracking the grid value-for-value. A wrong divisor, a mis-centred")
    print("    CUSUM, an out-of-cell permutation or a mis-scaled table breaks one")
    print("    column or the other. THE FIRE ITSELF IS DETERMINED BY THE GRID and is")
    print("    asserted above, not credited here. That is what §8 is worth: no more.")
    if not (ok and fired):                    # pragma: no cover - would be real
        sys.exit("§8 CALIBRATION FAILED — the statistic is not implemented as specified, "
                 "or the gate no longer discriminates. Refusing to measure.")
    return True


def collect(prompts, edges, norms, freqs, N, C):
    """§2/§7. One pass: qualifying cell-roles, their strata, and the control inputs.

    Returns (rows, diag) where each row is a qualifying (cell, role) with its rated
    words in MASS ORDER, centred arousal, and the per-word probability/frequency the
    §7 controls need. NOTHING is computed here — collection only, so the statistic
    and its null read one object.
    """
    tab = norms[("en", "arousal", "primary")]
    fq = freqs.get("en", {})
    dep = collections.defaultdict(list)
    cells = []
    diag = collections.Counter()
    for fam, pos, step in sorted(edges):
        for t in prompts:
            c = step.cell(t)
            if not c.is_present:
                diag["cell absent from the store (cut)"] += 1
                continue
            if c.language != "en":
                diag["non-en, outside the declared population (cut)"] += 1
                continue
            try:
                d = c.decompose(None)
            except RuntimeError:
                # ENVIRONMENT FAULT — never a data fact. Route (a) made the
                # store-missing error a RuntimeError, which `except Exception`
                # SWALLOWS where the old `sys.exit` could not be caught at all
                # ([1434].2). A silently-swallowed environment failure would
                # thin the population with no counter and no crash.
                raise
            except ValueError as e:
                diag["cell refused: mixed rule_version (data integrity)"
                     if "rule_version" in str(e)
                     else "cell errored: ValueError (code)"] += 1
                continue
            except Exception as e:
                diag[f"cell errored: {type(e).__name__} (code)"] += 1
                continue
            if not d:
                # THIRD UNCOUNTED DROP, found while executing [1435].1 and not
                # flagged by either seat. It removes the cell from `cells`, hence
                # from `rows`, hence from T -- the same population-thinning class
                # as site 258, one loop earlier and in the DATA bucket.
                diag["cell decomposed empty (data)"] += 1
                continue
            dep[t].append(d["departed"])
            cells.append((fam, t, c))
    disp = {t for t, v in dep.items() if v and st.median(v) >= N.DISPLACING_AT}
    ctrl = {t for t, v in dep.items() if v and st.median(v) < N.CONTROL_BELOW}

    rows = []
    for fam, t, c in cells:
        stratum = ("displacing" if t in disp else
                   "control" if t in ctrl else "gap")
        try:
            roles = N.cell_roles(c, C.RULE)
        except RuntimeError:
            raise                            # environment fault, as above
        except Exception as e:
            # COUNTED. This site dropped cells with no counter and no print until
            # [1434].3. It matters more than an ordinary silent drop because
            # T = sum of s OVER THE POPULATION: a drop changes T and the null's
            # dimensionality, while every printed diagnostic stays internally
            # consistent with the reduced set. NO NUMBER ON THE PRINTOUT WOULD
            # LOOK WRONG. The declaration doctrine applied to FAILURES, not just
            # to cuts.
            diag[f"roles errored: {type(e).__name__} (code)"] += 1
            continue
        byrole = collections.defaultdict(list)
        for w, wt, role in roles:
            k = N.norm_key(w, "en", fold=False)
            if N.is_function_word(k, "en"):
                diag["function word excluded (data)"] += 1
                continue
            z, _ = N.lookup(tab, k.casefold(), "en")
            if z is None:
                diag["no arousal rating (data)"] += 1
                continue
            p = c.pre.probs.get(w, 0.0)
            f = None
            for cand in (k.casefold(),):
                if cand in fq:
                    f = fq[cand]
            byrole[role].append((wt, z, p, f, w))   # z is RAW; centring is below
        for role, ws in byrole.items():
            if len(ws) < QUALIFYING_MIN:
                diag[f"role below the {QUALIFYING_MIN}-word bar (data)"] += 1
                continue
            ws.sort(key=lambda x: -x[0])          # §2 MASS ORDER, |delta| descending
            mu = st.mean(x[1] for x in ws)
            rows.append({
                "family": fam, "prompt": t, "role": role, "stratum": stratum,
                "n": len(ws),
                "centred": [x[1] - mu for x in ws],   # §3 centred, in mass order
                # RAW arousal retained: §2's reopen bar is DEFINED on raw z (its
                # calibration values +0.08 / +0.177 are raw correlations), and
                # computing it on centred z strips the between-cell covariance --
                # attenuating toward zero, which is the direction in which a guard
                # FAILS TO FIRE. The first draft discarded x[1] and the bar could
                # not have been evaluated on its own quantity.
                "raw": [x[1] for x in ws],
                "probs": [x[2] for x in ws],
                "freqs": [x[3] for x in ws],
                "words": [x[4] for x in ws],
            })
    return rows, diag, len(dep), len(disp), len(ctrl)


def joint_null(rows, rng, n_perm=N_PERM):
    """§3. ONE joint permutation: mass order permuted independently within EVERY
    qualifying cell, simultaneously. One p at 1/n_perm resolution, size-independent.

    There is no combining step and therefore nothing to assume. Fisher and Stouffer
    both require p ~ U(0,1); per-cell permutation p-values live on a lattice whose
    spacing is set by the DISTINCT statistic values — generically 2^(n-2) — so no
    weighting repairs them. This sidesteps combining entirely.
    """
    import numpy as np
    vs = [np.asarray(r["centred"], dtype=float) for r in rows]
    ms = [m_cell(list(v), rng, 400) for v in vs]
    obs_s = [cusum_max(v) / m for v, m in zip(vs, ms)]
    T_obs = float(sum(obs_s))
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = sum(cusum_max(rng.permutation(v)) / m for v, m in zip(vs, ms))
    # (1 + k) / (1 + n_perm): the exact permutation p is bounded below by
    # 1/(n_perm+1), and the naive mean admits 0.0000 -- a resolution the test
    # does not have.
    p = float((1 + int((null >= T_obs).sum())) / (1 + n_perm))
    return T_obs, float(np.median(null)), p, obs_s


def readouts(rows, rng, n_perm=N_PERM):
    """§6. R_faller / R_riser under the SAME joint null.

    B1 IS THE HARDER ARM AND THE SPEC SAYS SO IN ADVANCE: fallers are more coupled to
    probability (+0.830) than risers (+0.600), so the faller ordering carries more
    probability structure to see past.
    """
    import numpy as np
    out = {}
    for role in ("faller", "riser"):
        sel = [r for r in rows if r["role"] == role]
        if len(sel) < MIN_CELLS_TO_REPORT:
            out[role] = (len(sel), None, None, None)      # UNDERPOWERED, printed
            continue
        vs = [np.asarray(r["centred"], dtype=float) for r in sel]
        obs = float(np.mean([v[0] for v in vs]))       # top-mass word, centred
        null = np.empty(n_perm)
        for i in range(n_perm):
            null[i] = float(np.mean([rng.permutation(v)[0] for v in vs]))
        pct = float((1 + int((null < obs).sum())) / (1 + n_perm))
        out[role] = (len(sel), obs, float(np.median(null)), pct)
    return out


def main(a):
    import numpy as np
    rng = np.random.default_rng(SEED)

    # §8 RUNS FIRST — BEFORE frozen_population(), NOT MERELY BEFORE load_norms().
    # The first draft calibrated after the population read and the announcement
    # claimed "no data contact at all": precise about load_norms() and false about
    # the claim. `--calibrate-only` now touches nothing at all, so the gate can be
    # audited on a machine with no store.
    print("REGISTRATION B — the high-mass decomposition")
    print(f"  SPEC      registration_b_spec_v13.md  06186c42f9ff46e0  frozen [1412]")
    calibrate(rng, n_corpora=a.cal_corpora, n_perm=a.cal_perm)
    if a.calibrate_only:
        print("\n  --calibrate-only: stopping. NO read of any kind has occurred.")
        return 0

    N, C = _instrument()
    prompts, models, (ph, mh), drift = C.frozen_population()
    print(f"  RANKING   {RANKING}")
    print(f"  DIVISOR   M_cell, the cell's own permutation mean "
          f"(exact by enumeration at n <= {EXACT_ENUM_MAX})")
    print(f"  QUALIFY   >= {QUALIFYING_MIN} rated non-function words of the role")
    print(f"  NULL      ONE joint permutation, {N_PERM} draws, seed {SEED}")
    print(f"  FROZEN    prompts {len(prompts)} {ph[:16]}...  models {len(models)} "
          f"{mh[:16]}...")
    if drift:
        print("\n  *** POPULATION DRIFT — refusing ***")
        for d in drift:
            print(f"      {d}")
        return 1

    norms, freqs, _ = N.load_norms()
    edges, _ = C.operation_edges(models)
    rows, diag, n_moved, n_disp, n_ctrl = collect(prompts, edges, norms, freqs, N, C)

    print(f"\n  POPULATION  {n_moved} prompts with movement; "
          f"{n_disp} displacing, {n_ctrl} control")
    print(f"  QUALIFYING  {len(rows)} cell-roles at the >= {QUALIFYING_MIN} bar")
    print(f"  EVERY DROP BELOW IS COUNTED. T is a SUM OVER THIS POPULATION, so an")
    print(f"  uncounted drop would change T and the null's dimensionality with no")
    print(f"  printed number looking wrong. Buckets are SPLIT: (data) is a sentence")
    print(f"  about the corpus; (code) is a sentence about this program; (cut) is")
    print(f"  the declared population boundary of §1. A (code) count above zero is")
    print(f"  a DEFECT REPORT, not a population statement.")
    for k, v in diag.most_common():
        print(f"      {v:>7}  {k}")

    # --- §2's REOPEN BAR, WIRED, on THIS run's own population -----------------
    tab = norms[("en", "arousal", "primary")]
    fq = freqs.get("en", {})
    pairs = [(z, math.log10(p)) for r in rows
             for z, p in zip(r["raw"], r["probs"]) if p > 0]
    if len(pairs) <= MIN_PAIRS_FOR_REOPEN:
        # GUARD-(a), [1440].3(a): a guard ABSENT from the output is
        # indistinguishable from a guard that PASSED. Below the floor this
        # check used to skip in silence and the run proceeded to T with no
        # line anywhere saying the ranking question went untested.
        print(f"\n  §2 REOPEN BAR — NOT RUN. n={len(pairs)} (word, P) pairs, "
              f"below MIN_PAIRS_FOR_REOPEN={MIN_PAIRS_FOR_REOPEN}.")
        print("      THE RANKING CHECK DID NOT EXECUTE. T below is reported with")
        print("      the frequency-confound question OPEN, not settled.")
    else:
        rr = st.correlation([x for x, _ in pairs], [y for _, y in pairs])
        print(f"\n  §2 REOPEN BAR — corr(arousal, log P) on THIS run's population: "
              f"{rr:+.3f}")
        print(f"      bar |r| >= {REOPEN_BAR}: "
              f"{'REOPENS — ranking question is live' if abs(rr) >= REOPEN_BAR else 'inert, ranking stands'}")
        if abs(rr) >= REOPEN_BAR:
            print("      The ranking variable is confounded with the outcome on this")
            print("      population. Residualisation returns as construction; the spec")
            print("      says so and this run does not proceed past it.")
            return 1

    for stratum in ("displacing", "control"):
        sel = [r for r in rows if r["stratum"] == stratum]
        if len(sel) < MIN_CELLS_TO_REPORT:
            print(f"\n  {stratum}: {len(sel)} cell-roles, below the declared "
                  f"{MIN_CELLS_TO_REPORT} floor — UNDERPOWERED, NOT null")
            continue
        T, nul, p, obs_s = joint_null(sel, rng, a.perm)
        print(f"\n  ===== {stratum.upper()} — {len(sel)} qualifying cell-roles =====")
        print(f"    T = {T:.2f}   null median {nul:.2f}   p = {p:.4f}"
              f"   ({a.perm} joint draws)")

        # §4 per-size contribution and banded aggregate
        band = collections.defaultdict(list)
        for r, sv in zip(sel, obs_s):
            n = r["n"]
            key = 3 if n <= 3 else 4 if n <= 5 else 6 if n <= 9 else 10
            band[key].append(sv)
        print(f"    {'role size':>11}{'cells':>7}{'mean s':>9}{'CV':>7}")
        for k in sorted(band):
            lbl = {3: "3", 4: "4-5", 6: "6-9", 10: "10+"}[k]
            v = band[k]
            print(f"    {lbl:>11}{len(v):>7}{st.mean(v):>9.3f}"
                  f"{(st.pstdev(v)/st.mean(v)):>7.3f}")
        print("    CV RISES with size under M_cell; the low CV at n=3 is CONSTRAINT,")
        print("    not precision — a 3-word role admits only two distinct values.")

        # §6 directional readouts, same joint null
        for role, (n, obs, nm, pct) in readouts(sel, rng, a.perm).items():
            if obs is None:
                print(f"    R_{role:<7} n={n:>5}  below the {MIN_CELLS_TO_REPORT} "
                      f"floor — UNDERPOWERED, NOT null")
                continue
            print(f"    R_{role:<7} n={n:>5}  observed {obs:+.4f}  "
                  f"null median {nm:+.4f}  percentile {pct:.4f}")
        print("    B1 (faller > 0) is the HARDER arm: fallers are more coupled to")
        print("    probability (+0.830) than risers (+0.600).")

    # ===== §5 THE CURVE, and §7(b)/(c) THE CONTROLS =====================
    disp_rows = [r for r in rows if r["stratum"] == "displacing"]
    if disp_rows:
        print("\n  §5 CURVE — mean CENTRED arousal by POSITION in the mass ordering,")
        print("  stratified by role size. NO SINGLE BAND IS THE RESULT.")
        bands = {"3": lambda n: n == 3, "4-5": lambda n: 4 <= n <= 5,
                 "6-9": lambda n: 6 <= n <= 9, "10+": lambda n: n >= 10}
        print(f"    {'role size':>10}{'cells':>7}" +
              "".join(f"{'pos ' + str(i + 1):>9}" for i in range(5)))
        for lbl, f in bands.items():
            sel = [r for r in disp_rows if f(r["n"])]
            if not sel:
                print(f"    {lbl:>10}{0:>7}" + "".join(f"{'-':>9}" for _ in range(5)))
                continue          # [1423].1 promised a '-', not a vanished row
            cols = []
            for i in range(5):
                v = [r["centred"][i] for r in sel if len(r["centred"]) > i]
                cols.append(f"{st.mean(v):>+9.3f}" if len(v) >= 20 else f"{'-':>9}")
            print(f"    {lbl:>10}{len(sel):>7}" + "".join(cols))
        print("    position 1 is the TOP-MASS word. '-' is a band with fewer than")
        print(f"    {MIN_CELLS_TO_REPORT} cells at that position — declared, not dropped.")

        # THE THIRD CURVE: arousal over BASELINE PROBABILITY. §5's stated purpose --
        # "so a reader can see whether arousal tracks mass beyond tracking P".
        pb = [(math.log10(p_), z) for r in disp_rows
              for p_, z in zip(r["probs"], r["centred"]) if p_ > 0]
        if len(pb) <= MIN_WORDS_FOR_FREQ:
            print(f"\n  §5 THIRD CURVE — NOT RUN. n={len(pb)} words, below "
                  f"MIN_WORDS_FOR_FREQ={MIN_WORDS_FOR_FREQ}.")
        else:
            pb.sort()
            k = len(pb) // 5
            print("\n  §5 THIRD CURVE — mean centred arousal over BASELINE PROBABILITY")
            print(f"    {'log P quintile':>16}{'words':>8}{'mean centred arousal':>22}")
            for i in range(5):
                chunk = pb[i * k:(i + 1) * k] if i < 4 else pb[4 * k:]
                print(f"    {i + 1:>16}{len(chunk):>8}"
                      f"{st.mean(z for _, z in chunk):>+22.3f}")
            print("    Flat here and structured in the §5 curve = structure along MASS")
            print("    that is not structure along PROBABILITY, which is §0's question.")

        # ===== §7(b) FREQUENCY, WITHIN THE HIGH-MASS SET AND WITHIN THE TAIL ====
        hi = [(r["raw"][0], r["freqs"][0]) for r in disp_rows if r["freqs"][0]]
        tl = [(z, f) for r in disp_rows
              for z, f in zip(r["raw"][1:], r["freqs"][1:]) if f]
        print("\n  §7(b) FREQUENCY CONTROL — computed SEPARATELY in each set, because")
        print("  a selection check clears the cut it was run on and no other.")
        print(f"    {'set':>12}{'words':>8}{'mean logfreq':>15}{'corr(arousal,logfreq)':>24}")
        for lbl, S in (("high-mass", hi), ("tail", tl)):
            if len(S) < MIN_WORDS_FOR_FREQ:
                print(f"    {lbl:>12}{len(S):>8}   CONTROL NOT RUN, below "
                      f"MIN_WORDS_FOR_FREQ={MIN_WORDS_FOR_FREQ}")
                continue
            lf = [math.log10(f + 1.0) for _, f in S]
            az = [z for z, _ in S]
            print(f"    {lbl:>12}{len(S):>8}{st.mean(lf):>15.3f}"
                  f"{st.correlation(az, lf):>+24.3f}")
        if len(hi) < MIN_WORDS_FOR_FREQ or len(tl) < MIN_WORDS_FOR_FREQ:
            print(f"    A_logfreq — NOT RUN. high-mass n={len(hi)}, tail n={len(tl)}; "
                  f"one is below MIN_WORDS_FOR_FREQ={MIN_WORDS_FOR_FREQ}.")
        else:
            d = st.mean(math.log10(f + 1.0) for _, f in hi) - \
                st.mean(math.log10(f + 1.0) for _, f in tl)
            print(f"    A_logfreq (high-mass minus tail): {d:+.3f}")
            print("    High-mass words are structurally more frequent -- |delta| is")
            print("    bounded by probability. The question is whether that reaches")
            print("    AROUSAL, which the two correlations above answer.")

        # ===== §7(c) DENSE-STRATUM SCOPE, printed with every claim =============
        print("\n  §7(c) SCOPE: role size couples to concentration at -0.756, and the")
        print(f"  >= {QUALIFYING_MIN} bar selects on role size. T is therefore measured")
        print("  disproportionately on DENSE, LESS-CONCENTRATED cells. Any claim from")
        print("  this run carries that sentence; the banded table above is how a reader")
        print("  checks whether the effect lives in one stratum.")

    print("\n  §9 FALSIFIER: if T's p is not significant there is no arousal structure")
    print("  along the mass ordering beyond probability, and the [1250].3 sign")
    print("  observation DIES as evidence. No magnitude was predicted.")

    if a.csv and rows:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["family", "prompt", "role", "stratum",
                                               "n", "words", "centred"])
            w.writeheader()
            for r in rows:
                w.writerow({k: (r[k] if k not in ("words", "centred")
                                else "|".join(map(str, r[k]))) for k in
                            ("family", "prompt", "role", "stratum", "n",
                             "words", "centred")})
        print(f"\nwrote {a.csv}  {len(rows)} cell-roles")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv")
    p.add_argument("--perm", type=int, default=N_PERM)
    p.add_argument("--calibrate-only", action="store_true",
                   help="run §8 and stop before any data contact")
    p.add_argument("--cal-corpora", type=int, default=GATE_MIN_CORPORA)
    p.add_argument("--cal-perm", type=int, default=600)
    sys.exit(main(p.parse_args()))
