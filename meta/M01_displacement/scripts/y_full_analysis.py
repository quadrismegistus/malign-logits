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

## MULTIPLICITY

Not Bonferroni. It controls the family-wise error rate, which is the wrong
target for a descriptive table, and it assumes independence these cells do not
have -- they share passages, spans and pairs. Where a correction is wanted the
`--only story` section reports Benjamini-Hochberg, which controls the
proportion of false positives among rejections. Where one is not, the family
size is printed and the reader sees every cell.

Y's own convention governs: **the CI is the claim, the p ranks.** A result whose
argument depends on which correction you chose is a lead to register, not a
finding to quote.

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
SECTIONS = ("coverage", "diegetic", "route", "words", "regions", "prompt",
            "tags", "story", "reproduce")
MIN_N, MIN_PAIRS = 20, 8


def boot_mean(d, reps=4000, seed=4946):
    import random
    rng = random.Random(seed)
    n = len(d)
    out = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return out[int(.025 * reps)], out[int(.975 * reps)]


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
    ap.add_argument("--pass", dest="passfilter", default="A", choices=["A", "B", "all"],
                    help="A (default, and what every published number is), B, or all")
    ap.add_argument("--include-unparsed", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("sections: " + "  ".join(SECTIONS))
        return 0
    want = (lambda s: not a.only or s in a.only)

    import pandas as pd
    from y_paired_tests import boot_ci, wilcoxon

    P = pd.read_parquet(os.path.join(CAMP, "results", "y_passages.parquet"))
    #: THE TABLE HOLDS EVERYTHING; THE FILTER IS HERE AND IT IS NOT OPTIONAL.
    #: y_passages carries pass A AND B and unparsed rows, deliberately. Every
    #: published Y number is pass A, parsed. Reading the table unfiltered moved
    #: all nine reproduction cells -- assistant_refusal from 1.14 to 3.47 on the
    #: aligned arm, because refusal is a short-passage phenomenon (Y_superego 9)
    #: and pass B is where it lives. Defaulting to A makes the published path
    #: the default path; --pass B / all opts into the other population.
    n0 = len(P)
    if a.passfilter != "all":
        P = P[P["pass"] == a.passfilter]
    if not a.include_unparsed:
        P = P[P.parsed]
    print("population: pass=%s parsed=%s -> %s of %s rows"
          % (a.passfilter, not a.include_unparsed, format(len(P), ","), format(n0, ",")))
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

    if want("tags"):
        print("\n" + "=" * 100)
        print("TAG FREQUENCY, ONSET AND LENGTH   (Y_superego.md 4)")
        print("=" * 100)
        T = pd.read_parquet(TOKDIR, columns=["mid", "layer1", "layer2", "token_num", "l2_span_id"])
        T = T.merge(P[["mid", "arm", "pair"]], on="mid", how="left")
        head()
        for tag in LAYER2 + LAYER1:
            has = T[T.layer2.eq(tag) | T.layer1.eq(tag)].groupby("mid", observed=True).size()
            Q = P[["mid", "pair", "arm"]].copy()
            Q["hit"] = Q.mid.isin(has.index)
            rows = []
            for _p, g in Q.groupby("pair", observed=True):
                b, a = g[g.arm == "base"], g[g.arm == "aligned"]
                if len(b) < MIN_N or len(a) < MIN_N:
                    continue
                rows.append((100 * a.hit.mean() - 100 * b.hit.mean(),
                             100 * b.hit.mean(), 100 * a.hit.mean()))
            report(rows, "presence: <%s>" % tag, boot_ci, wilcoxon)
        print("\n  ONSET (median first token index) and LENGTH (median tokens), by arm")
        print("  %-9s %12s %12s %12s %12s" % ("tag", "onset base", "onset algn", "len base", "len algn"))
        for tag in LAYER2:
            r = []
            for arm in ("base", "aligned"):
                s2 = T[(T.arm == arm) & (T.layer2 == tag)]
                if not len(s2):
                    r += [float("nan")] * 2
                    continue
                g = s2.groupby("l2_span_id", observed=True)
                r += [g.token_num.min().median(), g.size().median()]
            print("  %-9s %12.0f %12.0f %12.0f %12.0f" % (tag, r[0], r[2], r[1], r[3]))

    if want("story"):
        print("\n" + "=" * 100)
        print("IS A TAG HARDER THAN PLAIN STORY, SAME PASSAGE?")
        print("  Per passage: mean(surprisal in tag) - mean(surprisal in plain story).")
        print("  MEANS are length-invariant, so no window matching -- unlike a max, which is")
        print("  the high-variance statistic an earlier version used and which hid this.")
        print("=" * 100)
        D = pd.read_parquet(TOKDIR, columns=["mid", "layer1", "layer2", "base_surprisal", "aligned_surprisal"])
        D = D.merge(P[["mid", "arm", "pair"]], on="mid", how="left")
        plain = (D.layer1 == "story") & (D.layer2.isna())
        pm = P.set_index("mid").pair
        for scorer in ("base_surprisal", "aligned_surprisal"):
            print("\n  scorer: %s" % scorer)
            print("  %-9s %-8s %6s %9s %20s %10s %6s"
                  % ("tag", "written", "pairs", "delta", "boot 95% CI (mean)", "wilcoxon", "sign"))
            ps = []
            for tag in LAYER2:
                for arm in ("base", "aligned"):
                    sub = D[D.arm == arm]
                    ins = sub[sub.layer2 == tag].groupby("mid", observed=True)[scorer].mean()
                    out = sub[plain & (sub.arm == arm)].groupby("mid", observed=True)[scorer].mean()
                    j = pd.concat([ins.rename("i"), out.rename("o")], axis=1, join="inner")
                    if len(j) < 200:
                        continue
                    j["d"] = j.i - j.o
                    j["pair"] = pm.reindex(j.index).values
                    per = [g.d.mean() for _k, g in j.groupby("pair", observed=True) if len(g) >= 5]
                    if len(per) < 10:
                        continue
                    lo, hi = boot_mean(per)
                    w, _ = wilcoxon(per)
                    m = statistics.mean(per)
                    ps.append((w, tag, arm))
                    print("  %-9s %-8s %6d %+9.3f  [%+7.3f,%+7.3f] %10.1e %3d/%-2d%s"
                          % (tag, arm, len(per), m, lo, hi, w,
                             sum(1 for x in per if (x > 0) == (m > 0)), len(per),
                             "  <=" if (lo > 0 or hi < 0) else ""))
            ps.sort()
            m_ = len(ps)
            print("  Benjamini-Hochberg at q=0.05 over %d cells (NOT Bonferroni -- see docstring):" % m_)
            kmax = 0
            for i, (pv, tag, arm) in enumerate(ps, 1):
                if pv <= i / m_ * 0.05:
                    kmax = i
            print("     rejects %d: %s" % (kmax, ", ".join("%s/%s" % (t, a) for _p, t, a in ps[:kmax]) or "none"))

    if want("reproduce"):
        print("\n" + "=" * 100)
        print("REPRODUCTION CHECK against the published finding documents")
        print("=" * 100)
        exp = [("SUPEREGO_IN_SCENE", True, None, 8.58, 11.18, "Y_diegetic 2"),
               ("CLEAN_SCENE", True, None, 45.22, 38.26, "Y_diegetic 2"),
               ("EXIT", True, None, 26.48, 27.80, "Y_diegetic 1"),
               ("sexual_scene", False, None, 53.85, 50.01, "Y_diegetic 1"),
               ("assistant_refusal", False, None, 0.10, 1.14, "Y_diegetic 1"),
               ("guilt_or_shame", False, "sex", 3.55, 5.79, "Y_diegetic 3"),
               ("consent_hesitation", False, "sex", 11.00, 16.34, "Y_diegetic 3"),
               ("SUPEREGO_IN_SCENE", True, "sex", 15.18, 21.60, "Y_diegetic 3"),
               ("CLEAN_SCENE", True, "sex", 84.72, 76.68, "Y_diegetic 3")]
        sx = P[P.sexual_scene == "YES"]
        print("  %-22s %-10s %9s %9s %9s %9s  %s"
              % ("measure", "panel", "exp base", "got", "exp algn", "got", "ok"))
        bad = 0
        for k, comp, panel, eb, ea, src in exp:
            rows = paired(P, k, comp=comp, sub=(sx if panel else None))
            gb = statistics.mean(x[1] for x in rows)
            ga = statistics.mean(x[2] for x in rows)
            ok = abs(gb - eb) < 0.02 and abs(ga - ea) < 0.02
            bad += not ok
            print("  %-22s %-10s %9.2f %9.2f %9.2f %9.2f  %s"
                  % (k, panel or "all", eb, gb, ea, ga, "yes" if ok else "**NO**"))
        print("\n  %d of %d reproduce to 0.02pp  (%s)"
              % (len(exp) - bad, len(exp), "ALL MATCH" if not bad else "MISMATCH -- investigate"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
