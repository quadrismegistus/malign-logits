"""The instrument's base rate: how much a step concentrates its distributions.

    from malign_logits.sharpening import sharpening, reduces_to_sharpening

    sharpening(step)                      # entropy, top-1 and residual, pre vs post
    reduces_to_sharpening(effects, steps)  # does the claimed effect vanish where
                                           # sharpening vanishes?

WHY THIS EXISTS. On 2026-07-31 a measured effect -- alignment pushes prompt distributions
apart, significant at 1e-06 to 1e-27 across five families -- turned out to be entirely
this. Two peaked distributions over different supports have high JS almost mechanically,
so if a step concentrates every distribution then pairwise divergence rises with no
increase in what the model distinguishes. **The confound and the claim are observationally
identical on JS alone.**

    family          top-1 pre->post   entropy pre->post   prompts diverged?
    amber            0.236 -> 0.336    3.750 -> 3.107      yes, +0.119
    beaver           0.286 -> 0.360    3.317 -> 2.705      yes, +0.059
    tulu             0.169 -> 0.213    4.007 -> 3.735      yes, +0.088
    olmo             0.192 -> 0.215    3.666 -> 3.439      yes, +0.063
    olmo-think       0.160 -> 0.170    3.701 -> 3.655      weakly, +0.018
    archangel-dpo    0.155 -> 0.154    4.172 -> 4.177      NO, +0.0006 ns

**ARCHANGEL IS THE ROSTER'S NATURAL NULL: the one family whose distributions do not sharpen
is the one whose prompts do not diverge.** That is what killed the claim, and it is why the
check is cheap -- it needs no simulation, only a family the operation barely touches.

THE CHECK IS CORRELATIONAL, NOT GENERATIVE, AND DELIBERATELY SO. A generative null would
temperature-scale each pre distribution to the observed post entropy and ask whether the
real divergence exceeds it. That cannot be done honestly here: `word_probs` gives a
TRUNCATED head plus one residual lump, and a lump of 0.1 made of a thousand words at
0.0001 sharpens nothing like a single atom at 0.1. Any temperature fit over the visible
support alone would be a null model for a distribution this instrument never observed.
So the check asks the question that actually caught it: **does the effect vanish where the
sharpening vanishes?**

THE BASE RATE IS A GATE, NOT A FOOTNOTE. Every distributional claim on this instrument
must be shown to exceed what sharpening alone produces, and the gate applies retroactively
to clauses booked before it existed. "Reduces to sharpening" is a named rival for all of
them.
"""
from __future__ import annotations

import math
import statistics as st


def _entropy(probs):
    return -sum(v * math.log2(v) for v in probs.values() if v > 0)


def sharpening(step, language="en", texts=None):
    """Concentration of `step`'s two arms over a common prompt set.

    Returns medians and their deltas. Medians rather than means because these
    distributions are skewed roughly 3:1 and a mean would track the tail.

        entropy_delta   NEGATIVE = the step concentrated. This is the headline number:
                        0.2-0.6 bits across the roster, and every pairwise-divergence
                        effect measured on 2026-07-31 followed from it.
        top1_delta      POSITIVE = concentrated.
        residual_delta  NEGATIVE = mass moved OUT of the unresolved tail into named words.

    `n` is the cells actually measured, which is not the prompt count: a cell absent from
    either arm is dropped, and a rate without its population is not a number.
    """
    from .movement import word_probs
    from .prompts import Prompts

    if texts is None:
        texts = [p.text for p in Prompts.where(language=language)]

    H0, H1, T0, T1, R0, R1 = [], [], [], [], [], []
    for t in texts:
        a = word_probs(step.pre.id, t)
        b = word_probs(step.post.id, t)
        if a is None or b is None or not a.probs or not b.probs:
            continue
        H0.append(_entropy(a.probs)); H1.append(_entropy(b.probs))
        T0.append(max(a.probs.values())); T1.append(max(b.probs.values()))
        R0.append(a.residual); R1.append(b.residual)
    if not H0:
        return None

    out = {
        "step": step.label, "family": step.family, "n": len(H0),
        "entropy_pre": st.median(H0), "entropy_post": st.median(H1),
        "top1_pre": st.median(T0), "top1_post": st.median(T1),
        "residual_pre": st.median(R0), "residual_post": st.median(R1),
    }
    out["entropy_delta"] = out["entropy_post"] - out["entropy_pre"]
    out["top1_delta"] = out["top1_post"] - out["top1_pre"]
    out["residual_delta"] = out["residual_post"] - out["residual_pre"]
    #: A step that does not concentrate cannot produce divergence for free, so it is the
    #: control the whole check rests on. The threshold is deliberately loose: this flags a
    #: candidate null, it does not certify one.
    out["is_flat"] = abs(out["entropy_delta"]) < 0.02 and abs(out["top1_delta"]) < 0.005
    return out


def reduces_to_sharpening(effects, steps, language="en", texts=None):
    """Does a claimed per-family effect vanish where the sharpening vanishes?

        effects   {family_key: measured effect size}
        steps     {family_key: Step}

    Returns the paired table plus a Spearman correlation between effect and entropy drop.
    **A HIGH CORRELATION IS NOT PROOF AND A LOW ONE IS NOT ACQUITTAL** -- with a roster of
    six the coefficient carries almost no power. The column that decides is `is_flat`: if a
    family whose distributions do not concentrate ALSO shows no effect, the effect is a
    candidate for reduction; if a flat family shows the full effect, it is not.

    Reported, never inferred. The caller reads the table.
    """
    rows = []
    for k, s in steps.items():
        sh = sharpening(s, language=language, texts=texts)
        if sh is None or k not in effects:
            continue
        rows.append({"family": k, "effect": effects[k],
                     "entropy_delta": sh["entropy_delta"],
                     "top1_delta": sh["top1_delta"], "is_flat": sh["is_flat"],
                     "n": sh["n"]})
    rho = None
    if len(rows) >= 3:
        def rank(vals):
            order = sorted(range(len(vals)), key=lambda i: vals[i])
            r = [0.0] * len(vals); i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                    j += 1
                for k2 in range(i, j + 1):
                    r[order[k2]] = (i + j) / 2 + 1
                i = j + 1
            return r
        a = rank([r["effect"] for r in rows])
        b = rank([-r["entropy_delta"] for r in rows])   # more sharpening = larger
        ma, mb = st.mean(a), st.mean(b)
        sa = math.sqrt(sum((x - ma) ** 2 for x in a))
        sb = math.sqrt(sum((y - mb) ** 2 for y in b))
        if sa and sb:
            rho = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)

    flat = [r for r in rows if r["is_flat"]]
    return {"rows": rows, "spearman": rho, "n_families": len(rows),
            "flat_families": [r["family"] for r in flat],
            "flat_effects": {r["family"]: r["effect"] for r in flat}}


def table(rows, title=""):
    """Print sharpening baselines. `flat` marks the roster's natural nulls."""
    if not rows:
        print("  no steps")
        return
    if title:
        print(f"\n{title}")
    print(f"  {'family':<16}{'step':<12}{'n':>5}{'entropy':>18}{'top-1':>16}"
          f"{'residual':>16}   flat")
    print(f"  {'':<16}{'':<12}{'':>5}{'pre':>9}{'post':>9}{'pre':>8}{'post':>8}"
          f"{'pre':>8}{'post':>8}")
    for r in rows:
        if r is None:
            continue
        print(f"  {str(r['family'])[:15]:<16}{r['step']:<12}{r['n']:>5}"
              f"{r['entropy_pre']:>9.3f}{r['entropy_post']:>9.3f}"
              f"{r['top1_pre']:>8.3f}{r['top1_post']:>8.3f}"
              f"{r['residual_pre']:>8.3f}{r['residual_post']:>8.3f}"
              f"   {'FLAT' if r['is_flat'] else ''}")
