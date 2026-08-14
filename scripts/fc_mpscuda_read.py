#!/usr/bin/env python
"""fc_mpscuda_read.py — the MPS-vs-CUDA pair-level gap, WRITTEN BEFORE THE RUN
LANDS. Commissioned by lacan at [4911].1, amended [4913].3(a).

    scripts/fc_mpscuda_read.py

WHAT IT MEASURES. `allenai/OLMo-2-0425-1B > allenai/OLMo-2-0425-1B-Instruct` is
in the 32-pair roster and its pass-1 beams were generated on an A100. The same
420 undisturbed units were regenerated on MPS into a NON-CANONICAL stash
(`beam_fc_mpscheck`) with **identical keys** — which is what makes the two
comparable, and also why the canonical store had to be kept out of it: the unit
key has no device field, so writing there would have collided and resume-by-key
would have skipped every unit, and the measurement would silently not happen.

WHY IT IS NEEDED. Tonight established hardware agreement in two places and
neither covers this: same GPU model across boxes is bit-identical (446/446), and
different CUDA GPUs agree at the pair level to <0.25% of the effect. **Both are
CUDA-to-CUDA.** MPS is the device known to diverge most at the beam level (0 of
460 identical beams) and was never checked at the pair level. And the roster
itself is 17 CUDA / 15 MPS, so this is not a property of the probes alone.

**PER-SITE DISTRIBUTION IS PRINTED, NOT ONLY THE PAIR MEAN** — lacan asked for
it explicitly, and the pair mean is exactly the statistic that would hide a
large symmetric per-site spread.

THIS FILE PRINTS NUMBERS AND NO CONCLUSIONS. Four times on 7 Aug an
interpretation went into a print statement ahead of the value it described; the
fourth was correct, which is worse, because an unchecked assertion that turns
out right teaches that the practice is safe. The reading goes in a post where it
can be disagreed with, never in an artifact that re-emits it every run.
"""
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

PAIR = "allenai/OLMo-2-0425-1B>allenai/OLMo-2-0425-1B-Instruct"
CHECK_STASH = "beam_fc_mpscheck"


def per_site(st, pair):
    """{prompt: asymmetry} over the undisturbed arm, the fc_analyse formula."""
    per = {}
    for k in st.keys():
        if not isinstance(k, dict) or k.get("type") != "fc_v1":
            continue
        if k.get("pair") != pair or k.get("arm") != "undisturbed":
            continue
        rec = st[k]
        sb, sa = rec.get("scored_by_base"), rec.get("scored_by_aligned")
        if not sb or not sa:
            continue
        first, second = (sb, sa) if k["role"] == "base" else (sa, sb)
        v = [x - y for r1, r2 in zip(first, second)
             for i, (x, y) in enumerate(zip(r1, r2)) if i > 0]
        if v:
            per.setdefault(k["prompt"], {})[k["role"]] = statistics.mean(v)
    return {p: (d["base"] - d["aligned"]) / 2 for p, d in per.items() if len(d) == 2}


def main():
    from malign_logits.cache import get_cache
    cm = get_cache()
    cuda = per_site(cm._stash("beam_fc"), PAIR)
    mps = per_site(cm._stash(CHECK_STASH), PAIR)
    common = sorted(set(cuda) & set(mps))
    print("pair    %s" % PAIR)
    print("sites   CUDA %d | MPS %d | shared %d" % (len(cuda), len(mps), len(common)))
    if len(common) < 5:
        print("shared sites < 5 — the MPS run has not landed")
        return
    c = [cuda[p] for p in common]
    m = [mps[p] for p in common]
    d = [mps[p] - cuda[p] for p in common]
    mc, mm = statistics.mean(c), statistics.mean(m)
    print()
    print("PAIR MEAN")
    print("  CUDA (A100, pass 1)   %+.6f" % mc)
    print("  MPS  (this run)       %+.6f" % mm)
    print("  MPS - CUDA           %+.6f" % (mm - mc))
    print("  as %% of the CUDA value %.2f%%" % (100 * abs(mm - mc) / abs(mc)))
    print("  as %% of the roster mean (-0.1381)  %.2f%%" % (100 * abs(mm - mc) / 0.1381))
    print()
    print("PER-SITE DIFFERENCE (MPS - CUDA), n=%d" % len(d))
    ds = sorted(d)
    print("  mean %+.6f | sd %.6f" % (statistics.mean(d), statistics.pstdev(d)))
    for q, lab in ((0.0, "min"), (0.05, "p05"), (0.25, "p25"), (0.50, "median"),
                   (0.75, "p75"), (0.95, "p95"), (1.0, "max")):
        print("  %-7s %+.6f" % (lab, ds[min(len(ds) - 1, int(q * (len(ds) - 1)))]))
    print("  |diff| > 0.01  %d/%d" % (sum(1 for x in d if abs(x) > 0.01), len(d)))
    print("  |diff| > 0.05  %d/%d" % (sum(1 for x in d if abs(x) > 0.05), len(d)))
    print("  identical to 4dp  %d/%d" % (sum(1 for x in d if abs(x) < 5e-5), len(d)))
    print("  sign agreement of the per-site asymmetry  %d/%d"
          % (sum(1 for a, b in zip(c, m) if (a < 0) == (b < 0)), len(common)))
    print()
    print("FOR REFERENCE, not computed here:")
    print("  between-pair device coefficient  -0.034 (drop) / -0.060 (drop+size)")
    print("  largest per-site CUDA-CUDA hardware effect observed   0.048")
    print("  census anti-competitor margins   0.0884 (OLMo-2) / 0.2255 (MiniCPM5)")


if __name__ == "__main__":
    main()
