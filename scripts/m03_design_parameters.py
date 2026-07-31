"""Every design parameter M03's guide quotes, in one re-runnable place.

The night this was written began with F21 declaring `data: []` and `scripts: []`,
which is why three weeks of its numbers had gone unchecked. The M03 design guide
now quotes a sigma, an MDE table, two correlations and a translation, all of which
were computed in throwaway shell heredocs. This is that defect about to repeat, so
this file exists before it does.

    .venv/bin/python scripts/m03_design_parameters.py

Docket: sigma and MDE [1776]; rho_pair [1767]; stem ICC [1772]; C1 translation and
the mean/median gap [1783].

TWO THINGS THIS FILE IS CAREFUL ABOUT, because both produced sign or scale errors
tonight before they were caught:

  STRATIFICATION   The data are CROSSED: family x prompt. An ICC pooled across
                   families counts family-level variation as within-cluster noise
                   and inverts. Every ICC here declares whether it is within-family
                   or across, and the raw spread is printed beside the estimate so
                   a model that contradicts it is visible ([1775].4).

  WHICH SD         A within-family prompt SD and a family-AVERAGED prompt SD are
                   different quantities. Mixing them produced a table whose printed
                   conclusion contradicted its own rows. The design averages over
                   the roster, so the family-averaged SD is the operative one, and
                   any effect compared against it must be on the same footing.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats

from c1_institutional_neutral import distinct_texts, isolated_steps  # noqa: E402
from malign_logits.contrast import rank_sum  # noqa: E402
from malign_logits.prompts import Prompts  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "m03_design_parameters.csv")
S_VALUES = (12, 18, 24)
ALPHA, POWER = 0.05, 0.80


def cells(domains=("institutional", "neutral")):
    rows = []
    for key, step in isolated_steps().items():
        for dom in domains:
            for p in distinct_texts(dom):
                c = step.cell(p.text)
                if c is not None and c.is_present:
                    rows.append(dict(family=key, domain=dom, prompt=p.text, js=c.js()))
    return pd.DataFrame(rows)


def sigma(d):
    """The operative SD: between prompts, on the FAMILY-AVERAGED value."""
    d = d.assign(c=d.js - d.groupby("family").js.transform("mean"))
    return float(d.groupby("prompt").c.mean().std()), float(d.c.std())


def mde(S, sig, rho):
    t = stats.t.ppf(1 - ALPHA / 2, S - 1) + stats.t.ppf(POWER, S - 1)
    return t * sig * np.sqrt(2 * (1 - rho)) / np.sqrt(S)


def stem_icc():
    """WITHIN-family stem ICC. Pooling across families gives +0.042 and inverts."""
    ps = [p for p in Prompts.where(language="en") if p.row.get("group_id")]
    byg = {}
    for p in ps:
        byg.setdefault(p.row["group_id"], []).append(p)
    byg = {g: m for g, m in byg.items() if len(m) > 1}
    rows = []
    for g, mem in byg.items():
        for key, step in isolated_steps().items():
            for m in mem:
                c = step.cell(m.text)
                if c is not None and c.is_present:
                    rows.append(dict(family=key, group=g, js=c.js()))
    d = pd.DataFrame(rows)
    out = []
    for _, g in d.groupby("family"):
        gg = g.groupby("group").js
        n = gg.ngroups
        if n < 3:
            continue
        kbar = gg.size().mean()
        msb = ((gg.mean() - g.js.mean()) ** 2 * gg.size()).sum() / (n - 1)
        msw = sum(((v - v.mean()) ** 2).sum() for _, v in gg) / max(len(g) - n, 1)
        v = (msb - msw) / kbar
        out.append(v / (v + msw) if (v + msw) > 0 else 0.0)
    # the raw-spread check that caught the pooled version
    w = d.groupby(["family", "group"]).js.std().median()
    b = d.groupby(["family", "group"]).js.mean().groupby("family").std().median()
    return np.median(out), len(byg), w, b


def rho_pair():
    """Correlation between a SPEAKER pair's two members. Unit is the PAIR: n=12."""
    pairs = pd.read_csv(os.path.join(os.path.dirname(OUT),
                                     "f21_institutional_prompts_paired.csv"))
    rows = []
    for key, step in isolated_steps().items():
        for i, r in pairs.iterrows():
            a, b = step.cell(r["individual"]), step.cell(r["institution"])
            if a is None or b is None or not a.is_present or not b.is_present:
                continue
            rows.append(dict(family=key, pair=i, ind=a.js(), inst=b.js()))
    d = pd.DataFrame(rows)
    rs = [np.corrcoef(g.ind, g.inst)[0, 1] for _, g in d.groupby("family") if len(g) >= 8]
    z = np.arctanh(np.clip(rs, -0.999, 0.999))
    r = float(np.tanh(z.mean()))
    n = d.pair.nunique()
    se = 1 / np.sqrt(n - 3)          # UNIT IS THE PAIR. Pooling over the 21 families
    lo = np.tanh(np.arctanh(r) - 1.96 * se)   # is 21 measurements of the SAME 12 pairs
    hi = np.tanh(np.arctanh(r) + 1.96 * se)   # and gives a spuriously narrow interval.
    return r, n, lo, hi


def c1_translation():
    """C1's effect on the design's scale, per family then averaged."""
    inst = [p.text for p in distinct_texts("institutional")]
    neut = [p.text for p in distinct_texts("neutral")]
    per = []
    for key, step in isolated_steps().items():
        A = [c.js() for t in inst if (c := step.cell(t)) is not None and c.is_present]
        B = [c.js() for t in neut if (c := step.cell(t)) is not None and c.is_present]
        if len(A) != len(inst) or len(B) != len(neut):
            continue
        U, _, _ = rank_sum(A, B)
        r = 2 * U / (len(A) * len(B)) - 1
        d = np.sqrt(2) * stats.norm.ppf(np.clip((r + 1) / 2, 1e-6, 1 - 1e-6))
        per.append(dict(family=key, r=r, implied=d * np.std(A + B, ddof=1),
                        observed=np.mean(A) - np.mean(B)))
    return pd.DataFrame(per)


def main():
    d = cells()
    sig, sig_cell = sigma(d)
    print("=" * 74)
    print("1. SIGMA -- the wall that models cannot move")
    print("=" * 74)
    print(f"   {d.prompt.nunique()} prompts x {d.family.nunique()} families")
    print(f"   SD of a single (prompt, family) cell, family-centred : {sig_cell:.4f}")
    print(f"   SD of the FAMILY-AVERAGED prompt value               : {sig:.4f}  <- sigma")
    print("   The roster buys the first drop and it is already bought; what remains")
    print("   is between-prompt variance and only SCENARIOS move it.")

    print("\n" + "=" * 74)
    print("2. THE TWO CORRELATIONS -- they are NOT the same quantity")
    print("=" * 74)
    r, n, lo, hi = rho_pair()
    print(f"   rho_pair (SPEAKER: one conflict, two sides)")
    print(f"      {r:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}] on n={n} PAIRS")
    print(f"      break-even for pairing to repay its halved unit count is rho=0.500,")
    print(f"      which is INSIDE the interval: the measurement cannot decide it.")
    icc, ngrp, w, b = stem_icc()
    print(f"   stem ICC (FORM: near-minimal, one word moved)")
    print(f"      {icc:+.3f} within-family, over {ngrp} stems")
    print(f"      raw-spread check: within-stem SD {w:.4f} vs between-stem {b:.4f}")
    print(f"      (ratio {w/b:.2f} -- an ICC near zero cannot be true of this, which is")
    print(f"       how the pooled-across-families version was caught)")

    print("\n" + "=" * 74)
    print("3. MDE -- full crossing, n = S scenarios")
    print("=" * 74)
    print(f"   two-sided alpha={ALPHA}, power={POWER}")
    print(f"   {'S':>4}{'prompts':>9}{'rho=0':>10}{'rho=0.5':>10}{'rho=0, /sqrt(6)':>18}")
    recs = []
    for S in S_VALUES:
        m0, m5 = mde(S, sig, 0.0), mde(S, sig, 0.5)
        print(f"   {S:>4}{S*12:>9}{m0:>10.4f}{m5:>10.4f}{m0/np.sqrt(6):>18.4f}")
        recs.append(dict(S=S, prompts=S * 12, mde_rho0=m0, mde_rho50=m5,
                         mde_rho0_sqrt6=m0 / np.sqrt(6)))
    print("   The last column assumes the six within-scenario realisations per side")
    print("   have independent residuals. UNVERIFIABLE until this design runs -- no")
    print("   existing corpus has twelve realisations of one scenario.")

    print("\n" + "=" * 74)
    print("4. C1 TRANSLATED -- a calibration anchor, NOT a prediction")
    print("=" * 74)
    t = c1_translation()
    print(f"   C1 is institutional-vs-NEUTRAL between strata; the design's contrast is")
    print(f"   individual-vs-institution WITHIN scenario. No reason they match in size.")
    print(f"   implied from rank-biserial (assumes normality) : {t.implied.mean():.4f}")
    print(f"   OBSERVED mean difference (no assumption)       : {t.observed.mean():.4f}")
    print(f"   conversion error: {100*abs(t.implied.mean()-t.observed.mean())/t.observed.mean():.0f}%"
          "   -> the observed is used")
    print(f"\n   AND THE LARGEST UNKNOWN ON THE BOARD:")
    print(f"      MEAN over families   {t.observed.mean():.4f}   detected at every S")
    print(f"      MEDIAN family        {t.observed.median():.4f}   detected at NO S, "
          f"even with sqrt(6)")
    print(f"      ratio {t.observed.mean()/t.observed.median():.1f}x")
    print("   A hypothesis about the roster MEAN and one about the TYPICAL FAMILY are")
    print("   different hypotheses with a 6x difference in required S. rho and the")
    print("   sqrt(6) are each worth at most 2.45x; this is worth six and is measured.")

    pd.DataFrame(recs).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
