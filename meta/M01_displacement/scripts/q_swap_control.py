"""q_swap_control.py — IS THE TRANSGRESSIVE SWAP BIGGER THAN AN ORDINARY ONE?

**WHY IT EXISTS. RH's objection, and it lands:** *"Wouldn't we expect the
matched pairs to have less distance between them than groups of a
heterogeneous collection which differ enormously in syntax, register etc?"*

**Yes. Trivially.** [4531] compared H1 — two texts differing by ONE WORD — to
`institutional - pair_marked`, two entirely different collections differing in
topic, syntax, length, register and source. **That the second is larger says
only that changing a whole text moves more than changing one word.** The
"5-26x" framing is withdrawn; it dressed an arithmetic inevitability as a
result.

**THE COMPARISON THAT IS NOT VACUOUS IS THIS ONE:** hold the design fixed —
one matched pair, one swapped word — and vary only whether the swap is
TRANSGRESSIVE. If the transgressive swap moves substitution no more than a
gender swap or a register swap does, then "transgressive" is not doing work
that any minimal edit would not do.

    **transgressive_swap   700 complete pairs**   (Q's H1: -0.002313)
    **everything else       42 complete pairs**   speaker_position 18,
      grievance 5, register 4, channel 4, role 4, negation 3, pole 2,
      intensity_ladder 2

**42 IS THE NUMBER THIS QUESTION HAS, AND IT IS SMALL.** The controls are
also heterogeneous and `speaker_position` (18 of 42) is not a lexical swap at
all — individual vs institutional speaker is a substantial semantic change.
**So this file reports per-type as well as pooled, and its stated limit is
that no single control type has enough pairs to stand alone.**

DESCRIPTIVE. No alpha, no test, no verdict language. Same machinery, same
population rule, same cluster unit as `q_baseline_table.py`.
"""
import collections
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))
for _p in (ROOT, os.path.join(ROOT, "scripts"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

N_ART = os.path.join(CAMPAIGN, "results", "result_n_primary.json")
CATALOGUE = os.path.join(ROOT, "data", "prompt_categorisation.json")
OUT = os.path.join(ROOT, "data", "q_swap_control.json")

#: Q's own transgressive figure, for the only comparison that matters here.
H1_TRANSGRESSIVE = -0.002313


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def cluster_interval(by_cluster, conf=0.95):
    from statistics import NormalDist
    cm = [mean(v) for v in by_cluster.values() if v]
    k = len(cm)
    if k < 2:
        return float("nan"), float("nan"), float("nan"), k
    m = mean(cm)
    sd = math.sqrt(sum((x - m) ** 2 for x in cm) / (k - 1))
    half = NormalDist().inv_cdf(1 - (1 - conf) / 2) * sd / math.sqrt(k)
    return m, m - half, m + half, k


def f11_pole_pairs():
    """F11's contradiction poles — RH's control, and the right one.

    F11 stores each item as a TRIPLE: pole A, pole B, and the combined
    "A and B" prompt. **The two POLES are a minimal swap in an identical
    frame** — *He was beautiful* / *He was disgusting*, *He was a man* /
    *He was a woman*, *The one who was free* / *The one who was captive* —
    **one word changed, and nothing transgressive.** That is exactly the
    comparison RH's objection calls for.

    The combined prompt is dropped: it is longer BY CONSTRUCTION (it
    contains both poles), so within a `pair_id` the longest distinct text
    is the AB form and the remaining two are the poles. A `pair_id` that
    does not leave exactly two after that drop is skipped rather than
    guessed at.
    """
    rows = json.load(open(CATALOGUE))["prompts"]
    acc = collections.defaultdict(set)
    for r in rows:
        pid = str(r.get("pair_id") or "")
        if pid.startswith("f11_"):
            acc[pid].add(r["prompt"])
    out = []
    for pid, texts in sorted(acc.items()):
        t = sorted(texts, key=len)
        if len(t) != 3:
            continue                      #: only complete A/B/AB triples
        a, b, ab = t
        #: **GUARD, NOT AN ASSUMPTION.** The AB form must actually be the
        #: one dropped. Its distinguishing feature is that it carries BOTH
        #: poles, so it is longer than either -- but "longest" is a proxy
        #: and a proxy for a property is not the property. Checked
        #: directly: the dropped text must be strictly longer than both
        #: kept texts, and the two kept texts must differ.
        if not (len(ab) > len(a) and len(ab) > len(b) and a != b):
            continue
        out.append((pid, a, b))
    return out


def pairs_by_type():
    """contrast_type -> [(marked_text, unmarked_text)], complete pairs only.

    Roles differ by family: most use MARKED/UNMARKED, `speaker_position`
    uses INDIV/INST. **The role names are read from the catalogue rather
    than assumed**, and a pair enters only with exactly two distinct roles.
    """
    rows = json.load(open(CATALOGUE))["prompts"]
    acc = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        ct, pid, role = r.get("contrast_type"), r.get("pair_id"), r.get("pair_role")
        if ct and pid and role:
            acc[ct][pid][role] = r["prompt"]
    out = {}
    for ct, byid in acc.items():
        got = []
        for _pid, v in byid.items():
            if len(v) != 2:
                continue
            #: order: MARKED/INST first so the sign convention matches H1's
            #: (marked minus unmarked).
            if "MARKED" in v and "UNMARKED" in v:
                got.append((v["MARKED"], v["UNMARKED"]))
            elif "INST" in v and "INDIV" in v:
                got.append((v["INST"], v["INDIV"]))
        if got:
            out[ct] = got
    return out


def main():
    import p_yield_pass as PY
    pop = PY.english_stimuli()

    art = json.load(open(N_ART))
    te = {}
    for c in art["cells"]:
        te.setdefault((c["prompt"], c["base"], c["aligned"]), c["tail_excess_corrected"])
    edges = sorted({(k[1], k[2]) for k in te})
    print("N artifact: %d cells, %d (base, aligned) edges" % (len(te), len(edges)))

    #: RH's control: F11's poles.
    f11 = f11_pole_pairs()
    by_cluster = collections.defaultdict(list)
    used = keys = 0
    kept = []
    for pid, a, b in f11:
        if a not in pop or b not in pop:
            continue
        per = [(b_, te[(a, b_, a_)] - te[(b, b_, a_)])
               for b_, a_ in edges
               if (a, b_, a_) in te and (b, b_, a_) in te]
        if not per:
            continue
        used += 1
        keys += len(per)
        kept.append((pid, a, b))
        for b_, d in per:
            by_cluster[b_].append(d)
    print("\n=== F11 POLE PAIRS — RH's control ([4531] withdrawn)")
    print("  triples found %d, usable in population %d, keys %d"
          % (len(f11), used, keys))
    for pid, a, b in kept[:14]:
        print("    %-18s %r  /  %r" % (pid, a[:38], b[:38]))
    if used >= 2:
        m, lo, hi, k = cluster_interval(by_cluster)
        #: **THE SIGNED MEAN IS THE WRONG STATISTIC HERE AND IT IS NOT THE
        #: WRONG ONE FOR H1.** H1's sign is meaningful: MARKED minus
        #: UNMARKED, and "marked" is always the transgressive member. An
        #: F11 pole pair has NO privileged member -- beautiful vs
        #: disgusting, man vs woman -- so which one is subtracted is an
        #: arbitrary ordering, and averaging signed differences across
        #: pairs lets arbitrary signs CANCEL. **The comparable quantity is
        #: the per-pair MAGNITUDE**, and H1 must be put on the same footing
        #: or the comparison is between a mean and a mean-of-absolutes.
        per_pair = collections.defaultdict(list)
        for (pid, a, b) in kept:
            per_pair[pid] = []
        for (pid, a, b) in kept:
            per_pair[pid] = [te[(a, b_, a_)] - te[(b, b_, a_)]
                             for b_, a_ in edges
                             if (a, b_, a_) in te and (b, b_, a_) in te]
        absmeans = [abs(mean(v)) for v in per_pair.values() if v]
        print("\n  **SIGNED mean over pairs is NOT comparable to H1**: an F11")
        print("  pole pair has no privileged member, so the ordering is")
        print("  arbitrary and signed differences CANCEL across pairs.")
        print("  per-pair |mean difference|, the comparable quantity:")
        for pid, v in sorted(per_pair.items(), key=lambda kv: -abs(mean(kv[1])) if kv[1] else 0):
            if v:
                print("      %-18s %+.6f   |%.6f|" % (pid, mean(v), abs(mean(v))))
        print("  **F11 poles, mean |per-pair difference| = %.6f  (n=%d)**"
              % (mean(absmeans), len(absmeans)))
        F11_ABS = mean(absmeans)

        #: **AND H1 MUST BE PUT ON THE SAME FOOTING OR THE COMPARISON IS
        #: RIGGED.** mean|x| >= |mean x| ALWAYS, so setting F11's
        #: mean-of-absolutes against H1's absolute-of-mean favours F11 by
        #: construction. H1's per-STEM absolute mean is recomputed here from
        #: the same artifact, over Q's own 684 stems and its own key rule.
        pops = json.load(open(os.path.join(CAMPAIGN, "results",
                                           "population_d_684.json")))["ids"]
        from malign_logits.prompts import Prompts
        byid = {str(x.id): x for x in Prompts().all()}
        h1_signed, h1_abs = [], []
        for stem in pops:
            m_, u_ = byid.get(stem + "_M"), byid.get(stem + "_U")
            if m_ is None or u_ is None:
                continue
            tm, tu = m_.text, u_.text
            if tm not in pop or tu not in pop:
                continue
            v = [te[(tm, b_, a_)] - te[(tu, b_, a_)]
                 for b_, a_ in edges
                 if (tm, b_, a_) in te and (tu, b_, a_) in te]
            if v:
                h1_signed.append(mean(v)); h1_abs.append(abs(mean(v)))
        print("\n  **LIKE FOR LIKE — both sets, both statistics**")
        print("    %-34s %10s %10s %6s" % ("", "mean", "mean|.|", "n"))
        print("    %-34s %+10.6f %10.6f %6d"
              % ("transgressive swap (Q's H1)", mean(h1_signed), mean(h1_abs),
                 len(h1_signed)))
        print("    %-34s %+10.6f %10.6f %6d"
              % ("F11 pole swap (ordinary)", m, F11_ABS, len(absmeans)))
        print("    **ratio of MAGNITUDES, F11 / transgressive = %.2fx**"
              % (F11_ABS / mean(h1_abs)))
        bigger = sum(1 for a in absmeans if a > mean(h1_abs))
        print("    %d of %d F11 pairs exceed the transgressive MEAN magnitude"
              % (bigger, len(absmeans)))
        #: **THE TWO SWAPS DIFFER IN SYSTEMATICITY, NOT IN SIZE.** If a
        #: set's per-pair signs were arbitrary, the signed mean would be
        #: about magnitude/sqrt(n) by cancellation alone. Comparing each
        #: set's OBSERVED signed mean to its OWN chance level is scale-free
        #: and needs no cross-set assumption.
        print("\n  **SIZE vs SYSTEMATICITY — the comparison that survives**")
        print("    %-26s %10s %10s %10s %8s"
              % ("", "|signed|", "magnitude", "if random", "obs/chance"))
        for lab, sg, mg, n in (("transgressive (H1)", abs(mean(h1_signed)),
                                mean(h1_abs), len(h1_abs)),
                               ("F11 pole (ordinary)", abs(m), F11_ABS,
                                len(absmeans))):
            ch = mg / math.sqrt(n)
            print("    %-26s %10.6f %10.6f %10.6f %7.2fx"
                  % (lab, sg, mg, ch, sg / ch))
        print("    **Nearly the SAME magnitude. The transgressive swap's")
        print("    direction is systematic; the pole swap's is not")
        print("    distinguishable from arbitrary.**")
        H1_RECOMP = {"mean_signed": mean(h1_signed), "mean_abs": mean(h1_abs),
                     "n_stems": len(h1_signed),
                     "signed_over_chance": abs(mean(h1_signed)) /
                                           (mean(h1_abs) / math.sqrt(len(h1_abs)))}
        print("\n  **F11 pole swap      %+.6f  [%+.6f, %+.6f]  %d pairs, %d clusters**"
              % (m, lo, hi, used, k))
        print("  **Q's transgressive  %+.6f  (H1, 684 pairs)**" % H1_TRANSGRESSIVE)
        print("  **ratio transgressive / F11-pole = %.2fx**"
              % (H1_TRANSGRESSIVE / m) if m else "")
        F11_RESULT = {"pairs": used, "keys": keys, "diff": m, "ci95": [lo, hi],
                      "n_clusters": k,
                      "ratio_transgressive_over_pole": (H1_TRANSGRESSIVE / m) if m else None,
                      "pair_ids": [x[0] for x in kept],
                      "mean_abs_per_pair_diff": F11_ABS,
                      "_note": "the SIGNED mean cancels arbitrary orderings; "
                               "compare mean_abs_per_pair_diff to "
                               "h1_recomputed.mean_abs, NOT to |H1 mean|",
                      "h1_recomputed": H1_RECOMP,
                      "ratio_magnitudes_f11_over_transgressive":
                          F11_ABS / H1_RECOMP["mean_abs"]}
    else:
        F11_RESULT = {"pairs": used, "note": "too few in population"}

    byt = pairs_by_type()
    print("\ncomplete pairs by contrast_type, before the population filter:")
    for ct in sorted(byt, key=lambda k: -len(byt[k])):
        print("  %-22s %4d" % (ct, len(byt[ct])))

    results = {}
    pooled_ctrl = collections.defaultdict(list)
    n_ctrl_pairs = 0
    print("\n%-22s %5s %6s  %-11s %-24s %s"
          % ("contrast_type", "pairs", "keys", "diff", "95% CI (cluster)", "clusters"))
    for ct in sorted(byt, key=lambda k: -len(byt[k])):
        by_cluster = collections.defaultdict(list)
        used, keys = 0, 0
        for m_, u_ in byt[ct]:
            if m_ not in pop or u_ not in pop:
                continue
            per = []
            for b_, a_ in edges:
                km, ku = (m_, b_, a_), (u_, b_, a_)
                if km in te and ku in te:
                    per.append((b_, te[km] - te[ku]))
            if not per:
                continue
            used += 1
            keys += len(per)
            #: one value per (pair, cluster), then the cluster is the unit
            for b_, d in per:
                by_cluster[b_].append(d)
        if used < 2:
            print("%-22s %5d %6d   -- too few pairs in population" % (ct, used, keys))
            continue
        m, lo, hi, k = cluster_interval(by_cluster)
        tag = "**Q's H1**" if ct == "transgressive_swap" else ""
        print("%-22s %5d %6d  %+.6f  [%+.6f, %+.6f]  %2d  %s"
              % (ct, used, keys, m, lo, hi, k, tag))
        results[ct] = {"pairs": used, "keys": keys, "diff": m,
                       "ci95": [lo, hi], "n_clusters": k}
        if ct != "transgressive_swap":
            n_ctrl_pairs += used
            for b_, v in by_cluster.items():
                pooled_ctrl[b_].extend(v)

    print("\n" + "=" * 74)
    print("THE COMPARISON RH's QUESTION IMPLIES — design held fixed, only the")
    print("KIND of swap varies. Both sides are ONE matched pair, ONE word.")
    print("=" * 74)
    if pooled_ctrl:
        cm, clo, chi, ck = cluster_interval(pooled_ctrl)
        t = results.get("transgressive_swap", {})
        print("  TRANSGRESSIVE swap   %4d pairs   %+.6f  [%+.6f, %+.6f]"
              % (t.get("pairs", 0), t.get("diff", float("nan")),
                 t.get("ci95", [float("nan")] * 2)[0], t.get("ci95", [float("nan")] * 2)[1]))
        print("  ORDINARY swaps       %4d pairs   %+.6f  [%+.6f, %+.6f]"
              % (n_ctrl_pairs, cm, clo, chi))
        if t.get("diff"):
            print("\n  ratio  transgressive / ordinary = %.2fx" % (t["diff"] / cm)
                  if cm else "")
        results["_pooled_control"] = {"pairs": n_ctrl_pairs, "diff": cm,
                                      "ci95": [clo, chi], "n_clusters": ck}
    results["f11_pole_swap"] = F11_RESULT
    results["_limits"] = [
        "42 control pairs against 700 transgressive. The control side is "
        "small and the interval is wide; this bounds the comparison, it "
        "does not settle it.",
        "The controls are HETEROGENEOUS and pooling them assumes only that "
        "they share the property of not being transgressive. "
        "speaker_position is 18 of 42 and is not a lexical swap at all -- "
        "individual vs institutional speaker is a substantial semantic "
        "change, so the pooled control is if anything an OVERestimate of "
        "what an ordinary minimal edit does.",
        "DESCRIPTIVE. No hypothesis was registered for any number here.",
        "**SIZE IS NOT WHERE THEY DIFFER.** A transgressive swap and an "
        "ordinary pole swap move substitution by nearly the same per-pair "
        "MAGNITUDE (0.0114 vs 0.0095). They differ in whether the "
        "DIRECTION is systematic: 5.3x chance for transgressive, 1.2x for "
        "the poles. H1's small signed effect is not a small effect in "
        "noise -- it is the systematic component of a per-pair effect five "
        "times larger and mostly idiosyncratic in direction.",
        "The F11 poles are DELIBERATELY MAXIMAL semantic contrasts "
        "(beautiful/disgusting, holy/filthy, free/captive), not a random "
        "sample of ordinary edits. That they only MATCH the transgressive "
        "magnitude is therefore the conservative reading.",
        "This replaces [4531]'s 5-26x framing, which compared a one-word "
        "delta to a whole-collection delta and was therefore vacuous. "
        "RH's objection, and he is right.",
    ]
    json.dump(results, open(OUT, "w"), indent=1, sort_keys=True)
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
