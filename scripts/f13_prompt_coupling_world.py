"""Does within-prompt normalisation absorb a PROMPT-LEVEL confound?

    uv run .venv/bin/python scripts/f13_prompt_coupling_world.py

Frozen at docket [586], amended at [587].1. Synthetic only; no model, no GPU.

WHY THIS EXISTS AND WHOSE PROPOSAL IT THREATENS. The per-prompt permutation is
dead at this corpus's sizes -- median 8 risers per cell, 1% of cells at >=25,
none at >=40 ([564]) -- so I proposed pooling risers across prompts with
within-prompt normalisation. That pooling is a real weakening and it is mine:
the per-prompt design absorbed prompt-level confounds by construction and a
pooled one does not. This world measures whether z-scoring absorbs them instead.

THE FAILURE IT HUNTS: prompts differing in BOTH mean similarity and mean excess,
with NO within-prompt coupling at all. If the pooled test rejects there, my
proposal manufactures (B) out of prompt heterogeneity and must be withdrawn.
BETA=0 is the negative control; the licensing table was declared before the run.

REPLICATE FLOORS ARE THE PEN'S ARITHMETIC, NOT MINE ([587].1). I wrote n=60 and
was wrong by ~7x: at n=60 an observed 3/60 has an exact upper bound of 0.146 and
does not clear 0.10, and a perfectly calibrated test clears the bar only ~20% of
the time -- a rule that fails the innocent four times in five. A VERDICT RULE HAS
ITS OWN POWER ANALYSIS, which I did not run on my own rule.
"""
import os
import sys

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_PROMPTS = 200          # [564]: per-edge cell counts 73-566, median ~205
SIM_MEAN, SIM_SD = 0.90, 0.036      # [562], the real amber edge's printed values
TAU = 0.08               # [546]: measured between-prompt offset 0.05-0.11
BETAS = (0.0, 0.5, 1.0, 2.0)
N_REP = {0.0: 400}       # [587].1: 400 for the negative control,
DEFAULT_REP = 100        #          100 for the swept rows
ALPHA = 0.05
N_PERM = 400
STRATA = 8
RNG = np.random.default_rng(20260729)


def riser_counts(n):
    """Empirical shape from [564]: median 8, 1% at >=25, 0% at >=40.
    Lognormal matched to those three points, floored at 4 (the analysis's own
    minimum cell) and hard-capped at 39 because ZERO real cells reached 40."""
    v = np.exp(RNG.normal(np.log(8.0), 0.62, size=n)).round().astype(int)
    return np.clip(v, 4, 39)


def one_world(bta):
    """Returns p from the pooled, within-prompt-normalised, stratified test."""
    sims, excs, freqs, pid = [], [], [], []
    for j in range(N_PROMPTS):
        k = int(riser_counts(1)[0])
        off = RNG.normal(0, TAU)                  # the prompt-level offset
        f = RNG.normal(0, 1, k)                   # frequency proxy
        s = SIM_MEAN + off + RNG.normal(0, SIM_SD, k)
        # excess depends on frequency and on the PROMPT offset -- never on s
        e = 0.5 * f + bta * off + RNG.normal(0, 1, k)
        sims.append(s); excs.append(e); freqs.append(f); pid.append(np.full(k, j))
    return pooled_test(np.concatenate(sims), np.concatenate(excs),
                       np.concatenate(freqs), np.concatenate(pid))


def _z(v):
    s = v.std()
    return (v - v.mean()) / s if s > 0 else np.zeros_like(v)


def pooled_test(sim, exc, freq, pid):
    """MY [564] PROPOSAL, implemented exactly as proposed: rank->z within prompt,
    pool, then permute excess within p_pre strata over the pooled set."""
    zs, ze = np.empty_like(sim), np.empty_like(exc)
    for j in np.unique(pid):
        m = pid == j
        zs[m] = _z(np.argsort(np.argsort(sim[m])).astype(float))
        ze[m] = _z(np.argsort(np.argsort(exc[m])).astype(float))
    obs = spearmanr(zs, ze).statistic
    order = np.argsort(freq)
    strat = np.empty(len(freq), int)
    strat[order] = (np.arange(len(freq)) * STRATA) // len(freq)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        p = ze.copy()
        for b in range(STRATA):
            m = strat == b
            if m.sum() > 1:
                p[m] = RNG.permutation(p[m])
        null[i] = spearmanr(zs, p).statistic
    return (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)


def main():
    print("PROMPT-COUPLING VALIDATION WORLD  (frozen [586], amended [587].1)")
    print(f"n_prompts {N_PROMPTS} | tau {TAU} | strata {STRATA} | perms {N_PERM}")
    print("no within-prompt sim-excess coupling in ANY row\n")
    print(f"{'BETA':>6}{'reps':>7}{'reject':>9}{'95% upper':>11}{'median p':>10}"
          f"{'verdict':>10}")
    for bta in BETAS:
        n = N_REP.get(bta, DEFAULT_REP)
        ps = np.array([one_world(bta) for _ in range(n)])
        rate = float((ps < ALPHA).mean())
        k = int((ps < ALPHA).sum())
        # exact Clopper-Pearson upper bound, the [561].3 criterion
        ub = 1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k))
        med = float(np.median(ps))
        ok = (ub <= 0.10) and (0.35 <= med <= 0.65)
        print(f"{bta:>6.1f}{n:>7}{rate:>9.3f}{ub:>11.3f}{med:>10.3f}"
              f"{('PASS' if ok else 'FAIL'):>10}")
    print("\nLICENSING, declared at [586].4 before this ran:")
    print("  rejects only at the BETA=0 rate -> z-scoring absorbs prompt coupling,")
    print("                                     the pooled design stands")
    print("  rejection RISES with BETA       -> it does not; MY [564] PROPOSAL IS")
    print("                                     WITHDRAWN and (B) needs another null")
    print("  median p at an extreme          -> degenerate, same disease as the")
    print("                                     per-prompt design; no runnable null")


if __name__ == "__main__":
    main()
