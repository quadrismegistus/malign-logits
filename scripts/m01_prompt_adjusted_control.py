"""POSITIVE CONTROL for the prompt-adjusted family estimator, BEFORE the freeze.

Docket [1806]/[1808].2 propose `A ~ family + prompt` as the PRIMARY family-generality
arm of the new gap registration, replacing a thresholded common core whose free
parameter moves p from 0.0015 to 0.43. The argument is sound — unbalanced two-way
estimation is the standard tool for cells present in some rows and not others.

BUT NOBODY HAS CHECKED THAT IT IS IDENTIFIABLE AT *THIS* COVERAGE. Family and
prompt effects are separable only if the incidence graph is connected enough to
tie them together. Our coverage is extreme: exactly ONE prompt is common to all
37 families, pairwise Jaccard median 0.61, minimum 0.13. A two-way model on a
nearly-disjoint design can be formally estimable and practically useless.

[1808].3 requires a positive control before freeze. This is it, and it runs now
because the answer must be known BEFORE the spec commits to the estimator.

    .venv/bin/python scripts/m01_prompt_adjusted_control.py

BLINDNESS IS PRESERVED ABSOLUTELY. This uses the (family, prompt) INCIDENCE of the
gap and displacing strata — WHICH cells exist — and never their values. Incidence
is outcome-blind and was already reported publicly at [1805]. Every `A` here is
SYNTHETIC, generated from known family and prompt effects. Nothing about H2's real
statistic on the gap is computed, glanced at, or estimated.

WHAT IT ANSWERS: given the true family effects, does the prompt-adjusted estimator
recover their SIGNS better than the naive per-family mean does? If it does not, the
remedy is not a remedy and the spec must not freeze on it.
"""

import os
import sys
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import m01_registration_b as B  # noqa: E402
import m01_registration_c3 as C3  # noqa: E402

N_SIM = 200
TRUE_POSITIVE_FRACTION = 0.70     # 70% of families genuinely positive
FAMILY_SD = 0.05                  # spread of true family effects
NOISE_SD = 0.40                   # within-cell noise, ~ the observed within-family SD

#: THE PROMPT SD IS NOT A FREE CHOICE -- it is fixed by the MEASURED prompt ICC,
#: and getting it wrong is the difference between a control and a formality. A
#: first pass used PROMPT_SD = 0.05 against NOISE_SD = 0.40, which is an induced
#: ICC of 0.015 -- it asked whether adjusting for a NEGLIGIBLE nuisance helps, and
#: of course it does not. rho = s_p^2/(s_p^2+s_e^2)  =>  s_p = s_e*sqrt(rho/(1-rho)).
#: Measured prompt ICC on comparable contrasts runs +0.05 to +0.14 ([1794]/[1761]),
#: so the control SWEEPS that range and beyond rather than picking a point.
ICC_SWEEP = (0.015, 0.05, 0.14, 0.30, 0.50)


def prompt_sd_for(rho):
    return NOISE_SD * (rho / (1.0 - rho)) ** 0.5


def incidence():
    """WHICH (family, prompt) cells exist. Values are never touched."""
    N, C = B._instrument()
    fp, fm, _h, _d = C.frozen_population()
    edges, _ = C.operation_edges(fm)
    norms, _f, _ = N.load_norms()
    cells, _diag, _nm, _nd, _nc = C3.collect(fp, edges, norms, N, C)
    out = {}
    for s in ("displacing", "gap"):
        pairs = [(c["family"], c["prompt"]) for c in cells if c["stratum"] == s]
        cnt = collections.Counter(f for f, _ in pairs)
        keep = {f for f, n in cnt.items() if n >= B.MIN_CELLS_TO_REPORT}
        out[s] = [(f, p) for f, p in pairs if f in keep]
    return out


def simulate(pairs, rng, prompt_sd):
    """Synthetic A with KNOWN family effects on the real incidence graph."""
    fams = sorted({f for f, _ in pairs})
    prompts = sorted({p for _, p in pairs})
    n_pos = int(round(TRUE_POSITIVE_FRACTION * len(fams)))
    true = rng.normal(FAMILY_SD, FAMILY_SD, len(fams))
    order = np.argsort(true)
    true[order[:len(fams) - n_pos]] = -np.abs(true[order[:len(fams) - n_pos]])
    true[order[len(fams) - n_pos:]] = np.abs(true[order[len(fams) - n_pos:]])
    fe = dict(zip(fams, true))
    pe = dict(zip(prompts, rng.normal(0, prompt_sd, len(prompts))))
    A = np.array([fe[f] + pe[p] + rng.normal(0, NOISE_SD) for f, p in pairs])
    return fams, prompts, fe, A


def naive_signs(pairs, A, fams):
    acc = collections.defaultdict(list)
    for (f, _), a in zip(pairs, A):
        acc[f].append(a)
    return {f: float(np.mean(acc[f])) for f in fams}


def adjusted_signs(pairs, A, fams, prompts):
    """A ~ family + prompt, least squares, family term read out.

    Sum-to-zero on prompts so the family coefficients are interpretable as
    effects rather than as contrasts against an arbitrary reference cell.
    """
    fi = {f: i for i, f in enumerate(fams)}
    pi = {p: i for i, p in enumerate(prompts)}
    n, F, P = len(pairs), len(fams), len(prompts)
    X = np.zeros((n, F + P))
    for r, (f, p) in enumerate(pairs):
        X[r, fi[f]] = 1.0
        X[r, F + pi[p]] = 1.0
    # prompt effects sum to zero: one extra row of constraint, heavily weighted
    con = np.zeros((1, F + P))
    con[0, F:] = 1.0
    Xc = np.vstack([X, con * 100.0])
    yc = np.concatenate([A, [0.0]])
    beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    return {f: float(beta[fi[f]]) for f in fams}


def main():
    inc = incidence()
    rng = np.random.default_rng(0)
    print("POSITIVE CONTROL — can the prompt-adjusted estimator recover family signs")
    print("at THIS coverage? Synthetic A on the real incidence graph.\n")
    print(f"  {N_SIM} simulations | {int(TRUE_POSITIVE_FRACTION*100)}% of families truly positive")
    print(f"  family SD {FAMILY_SD}, noise SD {NOISE_SD}; prompt SD set from each ICC\n")

    for stratum, pairs in inc.items():
        fams = sorted({f for f, _ in pairs})
        prompts = sorted({p for _, p in pairs})
        print(f"  {stratum.upper()}: {len(pairs)} cells | {len(fams)} families | "
              f"{len(prompts)} prompts")
        print(f"     {'prompt ICC':>11}{'prompt SD':>11}{'naive':>9}{'adjusted':>10}"
              f"{'gain':>8}")
        for rho in ICC_SWEEP:
            ps_sd = prompt_sd_for(rho)
            hit_n, hit_a = [], []
            for _ in range(N_SIM):
                fs, ps, true, A = simulate(pairs, rng, ps_sd)
                nv = naive_signs(pairs, A, fs)
                aj = adjusted_signs(pairs, A, fs, ps)
                tv = np.array([true[f] for f in fs])
                hit_n.append(np.mean(np.sign([nv[f] for f in fs]) == np.sign(tv)))
                hit_a.append(np.mean(np.sign([aj[f] for f in fs]) == np.sign(tv)))
            d = np.mean(hit_a) - np.mean(hit_n)
            flag = "  <- MEASURED" if rho == 0.14 else ""
            print(f"     {rho:>11.3f}{ps_sd:>11.3f}{np.mean(hit_n):>9.3f}"
                  f"{np.mean(hit_a):>10.3f}{d:>+8.3f}{flag}")
        print()

    print("  If the adjusted estimator does not beat the naive mean at this coverage,")
    print("  the remedy is not a remedy and the spec must not freeze on it as PRIMARY.")


if __name__ == "__main__":
    main()
