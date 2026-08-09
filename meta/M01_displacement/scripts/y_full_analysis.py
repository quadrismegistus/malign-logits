#!/usr/bin/env python
"""Every Y analysis we actually return to, from the two tables. Nothing re-derived.

    y_full_analysis.py                 # all sections
    y_full_analysis.py --only diegetic # one section
    y_full_analysis.py --list

Reads ONLY `results/y_passages.parquet` and `results/y_tokens/`, built by
`y_build_tables.py`. No raw generations, no tokenisers, no span relocation --
those happened once, at build time, so every section below is provably talking
about the same spans.

## WHAT THIS REPLACES AND WHAT IT DOES NOT

It supersedes the analysis paths of `y_diegetic.py`, `y_span_surprisal.py`,
`y_span_analysis.py` and `y_lexical_divergence.py`. **Those files stay**: they
produced numbers that are published in the finding documents, and a script that
generated a published number is a record, not a draft. When this file and one of
them disagree, that is a finding and not a merge conflict.

It does NOT replace `y_span_agreement.py` (needs both coders' raw output),
`y_field_analysis.py` (needs the USAS lexicon over span TEXT, which the tables
do not carry), or the manifest/delivery infrastructure.

## NESTING, AND THE ONE PLACE THE TABLE IS LOSSY

`layer1` and `layer2` are independent columns, so a `<guilt>` token inside
`<story>` carries both, and "ordinary narration" is `layer1=='story' AND layer2
is null` -- which is the correct baseline for a layer-2 contrast.

`layer2` is a SINGLE column, so layer-2 overlapping layer-2 collapses to the
last tag written (order: sexual, moral, guilt, consent, resist). Measured on
6,571 passages: **254 of 102,640 layer-2 tokens, 0.25%**, mostly guilt+moral and
moral+resist. Globally negligible; concentrated on `moral`, which loses about 3%
of its tokens to overwrite. `moral` is null in every contrast anyway, but the
number is here so nobody has to rediscover it.

## CONVENTIONS

Unit is the PAIR unless a section says otherwise: a rate or mean is computed
inside a pair, then aligned-minus-base across pairs. Never pooled over rows --
this corpus has repeatedly produced readings that were one model. Every clearing
cell carries how many pairs agree in sign.
"""
import argparse
import collections
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

LAYER2 = ("sexual", "moral", "guilt", "consent", "resist")
LAYER1 = ("story", "refusal", "noise", "meta", "web")
COMPOSITES = ("SUPEREGO_IN_SCENE", "CLEAN_SCENE", "EXIT", "MORAL_UTTERED")
FIELDS = ("sexual_scene", "consummation", "guilt_or_shame", "moralisation_in_scene",
          "consent_hesitation", "assistant_refusal", "frame_exit", "noise_present")
SECTIONS = ("coverage", "diegetic", "route", "words", "regions", "prompt")
MIN_N, MIN_PAIRS = 20, 8


def paired(P, key, comp=False, sub=None):
    """-> per-pair (aligned - base) in percentage points."""
    D = P if sub is None else sub
    out = []
    for _p, g in D.groupby("pair", observed=True):
        b, a = g[g.arm == "base"], g[g.arm == "aligned"]
        if len(b) < MIN_N or len(a) < MIN_N:
            continue
        f = (lambda x: (x[key] == True).mean()) if comp else (lambda x: (x[key] == "YES").mean())
        out.append((100 * f(a) - 100 * f(b), 100 * f(b), 100 * f(a)))
    return out


def report(rows, label, boot_ci, wilcoxon):
    if len(rows) < MIN_PAIRS:
        print("  %-24s (below the %d-pair floor)" % (label, MIN_PAIRS))
        return
    d = [x[0] for x in rows]
    lo, hi = boot_ci(d)
    p, _ = wilcoxon(d)
    med = statistics.median(d)
    print("  %-24s %5d %8.2f %9.2f %+9.2f  [%+6.2f,%+6.2f] %8.1e %3d/%-2d%s"
          % (label, len(d), statistics.mean(x[1] for x in rows),
             statistics.mean(x[2] for x in rows), med, lo, hi, p,
             sum(1 for x in d if (x > 0) == (med > 0)), len(d),
             "  <=" if (lo > 0 or hi < 0) else ""))


def head():
    print("  %-24s %5s %8s %9s %9s %18s %8s %6s"
          % ("measure", "pairs", "base %", "aligned %", "delta pp", "boot 95% CI", "p", "sign"))
    print("  " + "-" * 96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", choices=SECTIONS)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("sections: " + "  ".join(SECTIONS))
        return 0
    want = (lambda s: not a.only or s in a.only)

    import pandas as pd
    from y_paired_tests import boot_ci, wilcoxon

    P = pd.read_parquet(os.path.join(CAMP, "results", "y_passages.parquet"))
    TOKDIR = os.path.join(CAMP, "results", "y_tokens")

    if want("coverage"):
        print("=" * 100)
        print("COVERAGE  %s passages, %d pairs, %d models, %d prompts"
              % (format(len(P), ","), P.pair.nunique(), P.model.nunique(), P.prompt_id.nunique()))
        print("=" * 100)
        print("  arms %s" % dict(P.arm.value_counts()))
        print("  span location %.1f%% (%s of %s)"
              % (100 * P.spans_located.sum() / P.spans_total.sum(),
                 format(int(P.spans_located.sum()), ","), format(int(P.spans_total.sum()), ",")))
        print("  rt_band %s" % dict(P.rt_band.value_counts()))
        print("  forced %s / undisturbed %s"
              % (format(int(P.forced_word.notna().sum()), ","),
                 format(int(P.forced_word.isna().sum()), ",")))

    if want("diegetic"):
        print("\n" + "=" * 100)
        print("DIEGETIC vs EXTRA-DIEGETIC   (Y_diegetic_superego.md 1-3)")
        print("=" * 100)
        head()
        for k in FIELDS:
            report(paired(P, k), k, boot_ci, wilcoxon)
        for k in COMPOSITES:
            report(paired(P, k, comp=True), k, boot_ci, wilcoxon)
        sx = P[P.sexual_scene == "YES"]
        print("\n  GIVEN A SEXUAL SCENE  (%s passages)" % format(len(sx), ","))
        head()
        for k in ("guilt_or_shame", "moralisation_in_scene", "consent_hesitation"):
            report(paired(P, k, sub=sx), k, boot_ci, wilcoxon)
        for k in ("SUPEREGO_IN_SCENE", "CLEAN_SCENE"):
            report(paired(P, k, comp=True, sub=sx), k, boot_ci, wilcoxon)

    if want("route"):
        print("\n" + "=" * 100)
        print("AVOIDANCE vs THE SUPEREGO   (Y_diegetic_superego.md 6)")
        print("  is avoidance removable by forcing the word? does the superego care how it got there?")
        print("=" * 100)
        for lab, sub in (("UNDISTURBED", P[P.forced_word.isna()]), ("FORCED", P[P.forced_word.notna()])):
            print("\n  %s  (%s passages)" % (lab, format(len(sub), ",")))
            head()
            report(paired(P, "sexual_scene", sub=sub), "sexual_scene", boot_ci, wilcoxon)
            report(paired(P, "SUPEREGO_IN_SCENE", comp=True, sub=sub), "SUPEREGO_IN_SCENE", boot_ci, wilcoxon)
            report(paired(P, "EXIT", comp=True, sub=sub), "EXIT", boot_ci, wilcoxon)
            s2 = sub[sub.sexual_scene == "YES"]
            report(paired(P, "SUPEREGO_IN_SCENE", comp=True, sub=s2),
                   "  ...given sexual scene", boot_ci, wilcoxon)

    if want("words"):
        print("\n" + "=" * 100)
        print("WHICH WORDS DIVIDE THE TWO MODELS   (arm mean gap subtracted, so this is lexical)")
        print("=" * 100)
        D = pd.read_parquet(TOKDIR, columns=["mid", "token", "base_surprisal", "aligned_surprisal"])
        D = D.merge(P[["mid", "arm"]], on="mid", how="left")
        D["gap"] = D.aligned_surprisal - D.base_surprisal
        D["w"] = D.token.str.strip().str.lower()
        D = D[D.w.str.isalpha() & (D.w.str.len() >= 3)]
        for arm in ("base", "aligned"):
            g = D[D.arm == arm]
            mg = g.gap.mean()
            r = g.groupby("w", observed=True).gap.agg(["mean", "size"])
            r = r[r["size"] >= 400]
            r["centred"] = r["mean"] - mg
            print("\n  %s-WRITTEN  %s tokens, arm mean gap %+.3f subtracted"
                  % (arm.upper(), format(len(g), ","), mg))
            for lab, asc in (("ALIGNED more surprised", False), ("BASE more surprised", True)):
                print("   --- %s ---" % lab)
                for w, row in r.sort_values("centred", ascending=asc).head(10).iterrows():
                    print("   %-16s %+8.3f  n=%d" % (w, row.centred, row["size"]))

    if want("regions"):
        print("\n" + "=" * 100)
        print("SURPRISAL BY REGION.  'story (plain)' = layer1 story AND no layer-2 tag,")
        print("  which is the correct baseline for a layer-2 contrast.")
        print("=" * 100)
        D = pd.read_parquet(TOKDIR, columns=["mid", "layer1", "layer2", "base_surprisal", "aligned_surprisal"])
        D = D.merge(P[["mid", "arm"]], on="mid", how="left")
        D["region"] = D.layer2.where(D.layer2.notna(),
                                     D.layer1.where(D.layer1.isna() | (D.layer1 != "story"), "story (plain)"))
        D["region"] = D.region.fillna("untagged")
        print("  %-16s %-8s %10s %10s %10s %10s"
              % ("region", "arm", "tokens", "base", "aligned", "gap"))
        for arm in ("base", "aligned"):
            for reg in ["story (plain)"] + list(LAYER2) + ["meta", "noise", "web"]:
                s = D[(D.arm == arm) & (D.region == reg)]
                if len(s) < 5000:
                    continue
                print("  %-16s %-8s %10s %10.3f %10.3f %+10.3f"
                      % (reg, arm, format(len(s), ","), s.base_surprisal.mean(),
                         s.aligned_surprisal.mean(),
                         (s.aligned_surprisal - s.base_surprisal).mean()))
            print()

    if want("prompt"):
        print("\n" + "=" * 100)
        print("BY PROMPT and BY FORCED WORD   (gender: liminal_6 'she/her' vs liminal_7 'he/his')")
        print("=" * 100)
        print("  %-20s %8s %10s %10s %10s %10s"
              % ("prompt", "n", "sex base", "sex algn", "SUP base", "SUP algn"))
        for pid in sorted(P.prompt_id.dropna().unique()):
            g = P[P.prompt_id == pid]
            b, al = g[g.arm == "base"], g[g.arm == "aligned"]
            print("  %-20s %8d %10.1f %10.1f %10.1f %10.1f"
                  % (pid, len(g), 100 * (b.sexual_scene == "YES").mean(),
                     100 * (al.sexual_scene == "YES").mean(),
                     100 * (b.SUPEREGO_IN_SCENE == True).mean(),
                     100 * (al.SUPEREGO_IN_SCENE == True).mean()))
        print("\n  %-12s %-20s %8s %10s %10s" % ("word", "prompt", "n", "sex base", "sex algn"))
        lim = P[P.prompt_id.astype(str).str.contains("liminal")]
        for w in sorted(lim.forced_word.dropna().unique()):
            for pid in sorted(lim.prompt_id.dropna().unique()):
                g = lim[(lim.forced_word == w) & (lim.prompt_id == pid)]
                b, al = g[g.arm == "base"], g[g.arm == "aligned"]
                if len(b) < MIN_N or len(al) < MIN_N:
                    continue
                print("  %-12s %-20s %8d %10.1f %10.1f"
                      % (w, pid, len(g), 100 * (b.sexual_scene == "YES").mean(),
                         100 * (al.sexual_scene == "YES").mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
