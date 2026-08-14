"""Why is the ordering effect Chinese-only? Three explanations, all discarded.

    uv run python meta/M06_generation/scripts/m06_en_null_probes.py
    uv run python meta/M06_generation/scripts/m06_en_null_probes.py --splitter
    -> results/en_null_probes.json

`findings/zh_fluency_and_ordering.md` reports an ordering effect in Chinese
(18/25, median -0.0090, p=0.043) and nothing in English (14/25, -0.0003,
p=0.69). This file exists so the explanations that DID NOT work are on the
record with their numbers, rather than being re-derived by whoever asks next.

    1 HEADROOM      English base models are already coherent, so alignment has
                    nothing to tighten. Predicts the effect scales with how
                    badly the base writes.  -> DISCARDED
    2 LENGTH        English passages have ~2.5 fewer sentences (5.35 vs 7.79),
                    and order_ratio is a ratio of two within-passage averages,
                    so it is noisier.  -> DISCARDED
    3 SPLITTER      Chinese goes through stanza and English through NLTK, so
                    the two languages are not segmented into the same kind of
                    unit.  -> REAL, LARGE, AND ARM-NEUTRAL, so it does not
                    explain a contrast.  -> DISCARDED as the explanation

**A NEGATIVE RESULT NEEDS ITS NUMBER AS MUCH AS A POSITIVE ONE.** Each probe
prints the statistic that would have supported the explanation, so a reader
can see it was tested rather than dismissed.

THE ONE POSITIVE FINDING HERE IS INCIDENTAL and is reported with its own
caveat: aligned English models emit list/markup structure about 3.5x as often
as their base models. The detector is a crude token list and the base rate is
small; this is a lead, not a measurement.
"""
import argparse
import collections
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
OUTD = os.path.join(ROOT, "meta/M06_generation/results")
OUT = os.path.join(OUTD, "en_null_probes.json")

#: crude on purpose, and the docstring says so. A better-specified counter is
#: the obvious follow-up if the markup lead is ever taken up.
MARKUP = ("<li>", "</li>", "<p>", "\n- ", "\n* ", "\n1.", "\n2.")
ZH = "[一-鿿]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splitter", action="store_true",
                    help="run probe 3; needs stanza and is the slow one")
    ap.add_argument("--n", type=int, default=8, help="passages per model for probe 3")
    a = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "zo", os.path.join(HERE, "m06_zh_ordering.py"))
    zo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(zo)
    from scipy import stats

    d, use = zo.load("")
    sc = zo.fluency_scores()
    out = {"_about": __doc__.split("\n\n")[0], "probes": {}}

    #: ---- 1. HEADROOM ----
    print("1. HEADROOM: does the effect scale with how badly the BASE writes?")
    print("   (headroom predicts POSITIVE rho: worse base -> more negative delta)")
    print("   base fluency is a JUDGED measurement, independent of the embeddings,")
    print("   so this is not a delta correlated against its own baseline.")
    p1 = {}
    for lang in ("zh", "en"):
        for metric in ("order_ratio", "total_drift"):
            pm = zo.per_pair(d, metric, lang)
            xs, ys = [], []
            for p, v in pm.items():
                b = p.split(">")[0]
                if b in sc and v == v:
                    xs.append(sc[b])
                    ys.append(float(v))
            rho, pv = stats.spearmanr(xs, ys)
            p1["%s:%s" % (lang, metric)] = {"n": len(xs), "rho": float(rho),
                                            "p": float(pv)}
            print("   %-5s %-12s n=%d  rho %+.3f  p=%.3g" % (lang, metric, len(xs), rho, pv))
    out["probes"]["headroom"] = p1
    print("   -> DISCARDED: zh order_ratio is FLAT against base fluency.")
    print("      total_drift DOES trend the headroom way, which is one more")
    print("      way the two metrics behave differently.\n")

    #: ---- 2. LENGTH ----
    print("2. LENGTH: is English null because its passages have fewer sentences?")
    print("   mean n_sents: %s"
          % {l: round(float(d[d.lang == l].n_sents.mean()), 2) for l in ("zh", "en")})
    p2 = {}
    for lang in ("en", "zh"):
        for mn in (0, 6, 8):
            s = d[(d.lang == lang) & (d.n_sents >= mn)]
            g = (s.groupby(["pair", "prompt", "role"])["order_ratio"].mean()
                 .unstack("role").dropna(subset=["aligned", "base"]))
            pm = (g["aligned"] - g["base"]).groupby(level="pair").median()
            if len(pm) < 8:
                continue
            neg = int((pm < 0).sum())
            pv = stats.binomtest(neg, len(pm), 0.5).pvalue
            p2["%s:>=%d" % (lang, mn)] = {"pairs": len(pm), "neg": neg,
                                          "median": float(pm.median()), "p": float(pv)}
            print("   %-5s n_sents>=%-2d  %2d/%-2d  median %+.4f  p=%.3g"
                  % (lang, mn, neg, len(pm), pm.median(), pv))
    out["probes"]["length"] = p2
    print("   -> DISCARDED: English is still null at the Chinese sentence count.\n")

    #: ---- 3. SPLITTER ----
    if not a.splitter:
        print("3. SPLITTER: skipped (pass --splitter; needs stanza)")
    else:
        print("3. SPLITTER: nltk (English) against stanza (Chinese) on the SAME text")
        import nltk
        import stanza
        from malign_logits import ch
        pairs = [p for p in json.load(
            open(os.path.join(ROOT, "data/base_aligned_pairs.json")))
            if not p.get("ambiguous")]
        role = {}
        for p in pairs:
            role[p["base"]] = "base"
            role[p["aligned"]] = "aligned"
        rows = ch.query(
            "SELECT model, text FROM {db}.gen_sequences WHERE corpus='f11_l2' "
            "AND NOT match(prompt,'%s') ORDER BY cityHash64(model, prompt, "
            "sample_idx) LIMIT %d BY model" % (ZH, a.n))
        rows = [r for r in rows if r["model"] in role]
        en = stanza.Pipeline("en", processors="tokenize", verbose=False,
                             use_gpu=False)
        ident = same = 0
        jac, per, mk = [], collections.defaultdict(list), collections.defaultdict(list)
        for r in rows:
            t = r["text"]
            x = [s.strip() for s in nltk.sent_tokenize(t) if s.strip()]
            y = [s.text.strip() for s in en(t).sentences if s.text.strip()]
            ident += x == y
            same += len(x) == len(y)
            jac.append(len(set(x) & set(y)) / max(1, len(set(x) | set(y))))
            per[role[r["model"]]].append(len(y) - len(x))
            mk[role[r["model"]]].append(sum(t.count(m) for m in MARKUP))
        n = len(rows)
        u = stats.mannwhitneyu(per["base"], per["aligned"])
        p3 = {"n": n, "identical": ident / n, "same_count": same / n,
              "jaccard": float(st.mean(jac)),
              "delta_base": float(st.mean(per["base"])),
              "delta_aligned": float(st.mean(per["aligned"])),
              "arm_p": float(u.pvalue)}
        print("   passages %d | identical splits %.1f%% | same count %.1f%% | "
              "Jaccard %.3f" % (n, 100 * p3["identical"], 100 * p3["same_count"],
                                p3["jaccard"]))
        print("   stanza-minus-nltk sentences: base %+.3f | aligned %+.3f | "
              "Mann-Whitney p=%.3g"
              % (p3["delta_base"], p3["delta_aligned"], p3["arm_p"]))
        print("   -> the splitters disagree on MOST passages and disagree")
        print("      ARM-NEUTRALLY, so they cannot explain an arm contrast.")
        print("      DISCARDED as the explanation. What survives is that the")
        print("      English LEVEL is splitter-dependent, so the cross-language")
        print("      comparison of levels is not a comparison of one quantity.")
        out["probes"]["splitter"] = p3

        um = stats.mannwhitneyu(mk["base"], mk["aligned"])
        p4 = {"base_mean": float(st.mean(mk["base"])),
              "aligned_mean": float(st.mean(mk["aligned"])),
              "base_share_any": sum(1 for x in mk["base"] if x) / len(mk["base"]),
              "aligned_share_any": sum(1 for x in mk["aligned"] if x) / len(mk["aligned"]),
              "p": float(um.pvalue), "detector": list(MARKUP),
              "caveat": "crude token list; small base rate; a lead, not a measurement"}
        print("\n   INCIDENTAL: markup by arm (crude detector, see caveat)")
        print("      base    %.2f tokens/passage | %.1f%% of passages carry any"
              % (p4["base_mean"], 100 * p4["base_share_any"]))
        print("      aligned %.2f tokens/passage | %.1f%% of passages carry any"
              % (p4["aligned_mean"], 100 * p4["aligned_share_any"]))
        print("      Mann-Whitney p=%.4g" % p4["p"])
        out["probes"]["markup_by_arm"] = p4

    json.dump(out, open(OUT, "w"), indent=1)
    print("\n-> %s" % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
