#!/usr/bin/env python
"""fc_analyse.py — the forced-continuation and cross-forcing analysis.

    scripts/fc_analyse.py                       every pair in the stash
    scripts/fc_analyse.py --pair Qwen2.5-7B     one pair, by substring
    scripts/fc_analyse.py --site-level          per-site rows, not just pairs

WHAT IS MEASURED, AND WHY IT NEEDS FOUR ARMS.

M04's cross-forcing varies the SCORER and holds the word path fixed: it asks
how hard the aligned model resists a continuation the base produced. That is a
repression-with-depth measure and it cannot separate *dislikes this word* from
*dislikes what follows this word*.

Damage is a claim about WORDS, so it needs the opposite contrast -- vary the
word, hold the model fixed. Forcing both the faller and the riser under ONE
model is that contrast. Running it on the base as well as the aligned model is
the control: without it, an aligned-arm number has nothing to sit against and
cannot distinguish alignment-specific damage from what happens when any model
is pushed off its preference.

REPORTING RULES, adopted 2026-08-06 and applied here throughout:

  * every null states its MDE, or prints UNTESTED in that word
  * no borrowed statistical vocabulary for an untested judgement -- "noise",
    "flat", "no effect" are barred unless the test is shown
  * ALL tests computed are printed, including the ones that came out dull;
    nothing is selected after looking
  * the unit is the PAIR, because sites within a pair share a base model and
    are not independent replicates of anything

NOTHING HERE IS PRE-REGISTERED and nothing needs to be. The control is that
105 of 684 stems were drawn against a recorded seed and hash, and the other
579 have never been read.
"""
import argparse
import collections
import json
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

STASH = "beam_fc"
#: Closed-class list, fixed in advance and NOT tuned to any result. Standard
#: English function words: determiners, pronouns, auxiliaries, prepositions,
#: conjunctions, particles. Chosen because lacan [4749].2 finds riser mass at
#: pure-riser sites landing on exactly this class (`and` 6.3x, `be` 5.7x,
#: `that` 4.0x, `would` 3.7x) while lexical vocabulary is depleted.
FUNCTION_WORDS = set("""
a an the this that these those his her its their my your our
i he she it they we you me him them us
be is are was were been being am
have has had having do does did doing
will would shall should can could may might must
and or but nor so yet if then than as because while when where
of in on at to for with from by about into over after before
up down out off through under again further once here there
not no nor very just only also too then still even own same
""".split())
TWP_KEY = dict(dict_sha="b16011275c42955c", mode="raw", rule_version=3, theta=0.001)
#: **MINIMUM SITES FOR A PAIR TO ENTER AN ACROSS-PAIR MEAN.** A pair with one
#: site has no within-pair mean worth the name: its "mean over sites" is a
#: single draw, and it enters the across-pair statistic carrying the same
#: weight as a pair averaged over eighty. pythia-2.8b (1 site) and pythia-6.9b
#: (2) are the whole of the excluded set at present. Applied UNIFORMLY to every
#: across-pair measure rather than to the one where it was noticed, and both
#: the filtered and unfiltered tables are printed so the choice is visible.
MIN_SITES = 5
SEED = 20260806
DRAWS = 20000
POS0_NOTE = ("position 0 carries the first-token spike F28 documents and the "
             "M04 charter excludes it; reported separately, never pooled in")


def mean_lp(rec, scorer, lo=None, hi=None):
    """Mean per-token logprob of a unit's 100 continuations under `scorer`.

    `lo`/`hi` slice the CONTINUATION POSITIONS, which matters when comparing a
    forced unit against an unforced one: **a forced beam's position i sits one
    token LATER in the sentence than an unforced beam's position i**, because
    the pinned word consumes a slot. Comparing index-to-index would compare
    different sentence positions. The caller offsets: unforced 1..10 against
    forced 0..9.
    """
    rows = rec.get("scored_by_" + scorer)
    if not rows:
        return None
    vals = [x for r in rows for i, x in enumerate(r)
            if (lo is None or i >= lo) and (hi is None or i < hi)]
    return statistics.mean(vals) if vals else None


def site_meta():
    """prompt -> {stem, member, domain, stratum}, from the frozen sample.

    **THE JOIN KEY IS THE PROMPT STRING, AND A JOIN KEY IS A CLAIM.** Asserted
    unique here: two stems sharing a prompt would merge silently and the
    markedness split would compare a stem against itself. 105 stems x 2
    members = 210 rows, and the assertion says 210 distinct prompts.
    """
    import csv
    path = os.path.join(ROOT, "data", "beam_sample_105.csv")
    rows = list(csv.DictReader(open(path)))
    out = {r["prompt"]: r for r in rows}
    assert len(out) == len(rows), (
        "%d rows collapse to %d distinct prompts -- the join key is not unique"
        % (len(rows), len(out)))
    return out


def permutation_p(xs, draws=DRAWS, seed=SEED):
    """Sign-flip permutation on the paired differences. Returns (p, n)."""
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n < 2:
        return None, n
    obs = abs(statistics.mean(xs))
    rng = random.Random(seed)
    hits = sum(1 for _ in range(draws)
               if abs(sum(x if rng.random() < .5 else -x for x in xs) / n) >= obs)
    return (hits + 1) / (draws + 1), n


def mde(xs, power=0.80, alpha=0.05):
    """Minimum detectable effect for a one-sample paired test at this n and sd.
    **PRINTED BESIDE EVERY NULL.** A null without one is not a finding, it is
    an absence of measurement, and the two are not the same claim."""
    xs = [x for x in xs if x is not None]
    if len(xs) < 3:
        return None
    sd = statistics.pstdev(xs)
    return 2.802 * sd / math.sqrt(len(xs))     # z(.975)+z(.80) = 1.960+0.842


def verdict(m, p, xs):
    """One of the four terminal states. NOT DETECTED carries its MDE."""
    if p is None:
        return "UNTESTED (n<2)"
    if p < 0.05:
        return "detected  p=%.4f" % p
    d = mde(xs)
    return ("not detected at n=%d, MDE %s"
            % (len([x for x in xs if x is not None]),
               "%.4f" % d if d else "unavailable"))


def word_probs(cm, model, prompt):
    """The word distribution behind a site, from the twp stash."""
    k = dict(TWP_KEY); k["model"] = model; k["prompt"] = prompt
    st = cm._stash("true_word_probs")
    try:
        v = st[k]
    except Exception:
        return None
    rows = v.get("rows") if isinstance(v, dict) else None
    if not rows:
        return None
    by = collections.defaultdict(float)
    for r in rows:
        by[r["word"]] += r["p"]
    return by


def confound_sign(cm, pid, sites):
    """**DOES THE PROBABILITY CONFOUND EVEN POINT THE RIGHT WAY?**

    The faller and the riser are not matched on prior probability -- measured
    on Qwen2.5-7B the riser is the base's argmax in 25 of 41 sites (median
    rank 0, p 0.16) while the faller never is (median rank 4, p 0.02). So a
    cost difference between them is partly just the probability gap.

    But the gap WIDENS under alignment by construction, since "fell" means
    exactly that. A purely probability-driven world therefore predicts the
    ALIGNED model pays MORE extra for the faller -- a POSITIVE interaction.
    If the observed interaction is NEGATIVE, the confound predicts the
    opposite sign to the observation and cannot be producing it.

    This costs two numbers and settles direction, though never magnitude.
    """
    b, a = pid.split(">")
    gb, ga, fr, rr = [], [], [], []
    for s in sites:
        pb, pa = word_probs(cm, b, s["prompt"]), word_probs(cm, a, s["prompt"])
        if not pb or not pa:
            continue
        f, r = s["faller"], s["riser"]
        if min(pb.get(f, 0), pb.get(r, 0), pa.get(f, 0), pa.get(r, 0)) <= 0:
            continue
        gb.append(math.log(pb[r]) - math.log(pb[f]))
        ga.append(math.log(pa[r]) - math.log(pa[f]))
        order = sorted(pb, key=lambda w: -pb[w])
        fr.append(order.index(f)); rr.append(order.index(r))
    if not gb:
        return None
    return {"n": len(gb), "gap_base": statistics.mean(gb),
            "gap_aligned": statistics.mean(ga),
            "widening": statistics.mean(ga) - statistics.mean(gb),
            "faller_rank": statistics.median(fr),
            "riser_rank": statistics.median(rr),
            "riser_is_argmax": sum(1 for x in rr if x == 0)}


def _js(p, q):
    ks = set(p) | set(q)
    out = 0.0
    for k in ks:
        a, b = p.get(k, 0.0), q.get(k, 0.0)
        m = (a + b) / 2
        if a > 0:
            out += 0.5 * a * math.log(a / m)
        if b > 0:
            out += 0.5 * b * math.log(b / m)
    return out / math.log(2)


def _first_token_marginal(rec):
    """p(next | prompt + forced word), approximated from the beams.

    **THIS IS A TOP-100-PATH TRUNCATION, NOT A DISTRIBUTION.** Beam search
    concentrates: 100 beams on Qwen2.5-7B explored continuations of only SIX
    distinct first tokens. F14's syntagmatic_js was computed over the full
    vocabulary; this is the same shape of quantity on a far coarser support
    and the two numbers are not interchangeable.
    """
    d = collections.defaultdict(float)
    for b in rec.get("beams", []):
        if b.get("tokens"):
            d[b["tokens"][0]] += math.exp(b["log_prob"])
    tot = sum(d.values()) or 1.0
    return {k: v / tot for k, v in d.items()}


LEGACY_DESIGN = "legacy-pass1"     #: records written before `design` existed


def load(cm, pair_filter=None, design=None, quiet=False):
    """**THE DESIGN GUARD.** One stash holds every design ever run, and an
    analysis that filters on `arm` alone gets all of them silently pooled. That
    is not hypothetical: the OLMo-2 SFT probe joined a roster check four posts
    after it had been declared out of that population, because the declaration
    lived in a docket post and a manifest note and `load()` reads neither.

    `design=None` returns everything AND PRINTS A CENSUS, so pooling is visible
    at the point it happens rather than inferable afterwards. Passing a string
    filters to it. Records written before the field existed are reported as
    LEGACY_DESIGN rather than dropped — a guard that silently excluded the
    entire existing roster would be far worse than the pooling it prevents.
    """
    st = cm._stash(STASH)
    by = collections.defaultdict(dict)
    seen = collections.Counter()
    for k in st.keys():
        if not isinstance(k, dict) or k.get("type") != "fc_v1":
            continue
        if pair_filter and pair_filter not in k["pair"]:
            continue
        v = st[k]
        d = (v.get("design") if isinstance(v, dict) else None) or LEGACY_DESIGN
        seen[d] += 1
        if design is not None and d != design:
            continue
        by[k["pair"]][(k["role"], k["arm"], k["word"] or "", k["prompt"])] = v
    if not quiet and len(seen) > 1:
        if design is None:
            print("  ** %d DESIGNS POOLED by load(): %s"
                  % (len(seen), ", ".join("%s=%d" % x for x in seen.most_common())))
            print("  ** pass design=... to select one. Pooling may be correct —")
            print("  ** it must not be accidental.")
        else:
            print("  design=%s: %d of %d records (%s)"
                  % (design, seen.get(design, 0), sum(seen.values()),
                     ", ".join("%s=%d" % x for x in seen.most_common())))
    return by


def analyse_pair(pid, cells):
    """Returns a dict of per-pair quantities plus the per-site vectors."""
    base_name = pid.split(">")[0].split("/")[-1]
    #: ---- DAMAGE: vary the word, hold the model fixed ----
    prompts = sorted({k[3] for k in cells if k[1] == "force_faller"})
    swap_base, swap_algn, dd, own = [], [], [], []
    #: prompt label kept ALONGSIDE dd, index-aligned, so the markedness split
    #: below can subset without recomputing. Appended in the same branch as dd
    #: -- if the two ever drift apart the assertion at the call site fires.
    dd_prompt = []
    #: **THE CONSTRAINT COST, against each model's own unforced beam.** Forcing
    #: ANY first token costs the model something -- measured at 0.03-0.08
    #: nats/token on Qwen2.5-7B, which is the same size as the whole
    #: riser-minus-faller effect. Without this reference that cost is invisible
    #: and a relative difference gets read as an absolute one.
    cost_f = {"base": [], "aligned": []}
    cost_r = {"base": [], "aligned": []}
    #: prompt labels index-aligned with cost_f/cost_r, per role, so the
    #: markedness section can split the FALLER and RISER arms SEPARATELY.
    #: Needed because lacan [4735].4 finds withdrawal is transgression-specific
    #: while substitution is universal -- if so, a difference statistic like
    #: `dd` is precisely where that structure cancels and disappears.
    cost_prompt = {"base": [], "aligned": []}
    for p in prompts:
        g = {(r, a): v for (r, a, w, pp), v in cells.items() if pp == p}
        need = [("base", "force_faller"), ("base", "force_riser"),
                ("aligned", "force_faller"), ("aligned", "force_riser")]
        if not all(x in g for x in need):
            continue
        bf, br = mean_lp(g[need[0]], "base"), mean_lp(g[need[1]], "base")
        af, ar = mean_lp(g[need[2]], "aligned"), mean_lp(g[need[3]], "aligned")
        if None in (bf, br, af, ar):
            continue
        #: **ONE SIGN CONVENTION: every column is RISER MINUS FALLER.** The
        #: first version had swap_algn as faller-minus-riser, so two columns
        #: with parallel names ran in opposite directions and the table read
        #: as disagreement where there was none.
        #:
        #: And neither word is "the model's own". A faller is a word that LOST
        #: probability under alignment and a riser is one that gained; neither
        #: is necessarily any model's argmax. Calling the faller the base's
        #: own word imports an assumption the population does not carry.
        swap_base.append(br - bf)          # base:    riser minus faller
        swap_algn.append(ar - af)          # aligned: riser minus faller
        dd.append((ar - af) - (br - bf))   # the interaction
        dd_prompt.append(p)                # index-aligned with dd
        own.append(ar - bf)                # aligned-after-riser vs base-after-faller
        #: cost of the constraint: unforced minus forced, positions offset by
        #: one so both sides cover the same sentence positions.
        for role in ("base", "aligned"):
            un = g.get((role, "undisturbed"))
            ff, rr = g.get((role, "force_faller")), g.get((role, "force_riser"))
            if not (un and ff and rr):
                continue
            #: **OFFSET BY THE FORCED WORD'S ACTUAL TOKEN COUNT, not by 1.**
            #: The pinned word consumes as many sentence positions as it has
            #: tokens, so a two-token faller puts forced position 0 at sentence
            #: position 2. 98.6% of forced words are single-token and the
            #: constant was right for them; it was wrong for the 1.4% that are
            #: not, comparing different sentence positions. `n_forced_tokens`
            #: was recorded from the first run and simply never read.
            nf = ff.get("n_forced_tokens") or 1
            nr = rr.get("n_forced_tokens") or 1
            u_f = mean_lp(un, role, nf, nf + 9)
            u_r = mean_lp(un, role, nr, nr + 9)
            a_ = mean_lp(ff, role, 0, 9)
            b_ = mean_lp(rr, role, 0, 9)
            if None in (u_f, u_r):
                continue
            if None in (a_, b_):
                continue
            cost_f[role].append(u_f - a_)
            cost_r[role].append(u_r - b_)
            cost_prompt[role].append(p)
    #: ---- RESIST: vary the scorer, hold the word path fixed ----
    #:
    #: **SELF-BASELINING IS BIASED AND BOTH DIRECTIONS ARE NEEDED.** Beam
    #: search MAXIMISES the generator's own score, so self-minus-judge is
    #: inflated toward positive by construction: the self term is a max over
    #: candidates, the judge term is merely an evaluation. Measured on
    #: Qwen2.5-7B, direction A was +0.002 and direction B +0.333 -- BOTH
    #: positive, which is the selection cost showing up twice rather than two
    #: findings.
    #:
    #:     A = on the BASE's beams:    logP_base    - logP_aligned
    #:     B = on the ALIGNED's beams: logP_aligned - logP_base
    #:     (A+B)/2  the COMMON SELECTION COST -- an artefact of beam search,
    #:              and exactly what a one-direction measure hides
    #:     (A-B)/2  the ASYMMETRY -- does the aligned model resist the base's
    #:              continuations more than the base resists the aligned's?
    #:              This is the quantity a one-direction measure cannot see.
    pos = {"A": collections.defaultdict(list), "B": collections.defaultdict(list)}
    for (role, arm, w, p), rec in cells.items():
        if arm != "undisturbed":
            continue
        sb, sa = rec.get("scored_by_base"), rec.get("scored_by_aligned")
        if not sb or not sa:
            continue
        #: on the base's beams the self term is base; on the aligned's it is
        #: aligned. Same sequence, same scorers -- only which one generated.
        first, second = (sb, sa) if role == "base" else (sa, sb)
        key = "A" if role == "base" else "B"
        for r1, r2 in zip(first, second):
            for i, (x, y) in enumerate(zip(r1, r2)):
                pos[key][i].append(x - y)
    #: **THE ASYMMETRY'S OWN DENOMINATOR.** `n_sites` counts FORCED sites
    #: (len(dd)); the asymmetry above is built from the UNDISTURBED arm and has
    #: a different, larger n. Printing the asymmetry beside n_sites pairs a
    #: value with a count from another arm -- 46 against 210 for deepseek. Same
    #: family as tonight's other unit errors, so the count travels with the
    #: quantity rather than being fetched from whatever is nearest.
    _und = {}
    for (role, arm, w, p_) in cells:
        if arm == "undisturbed":
            _und.setdefault(p_, set()).add(role)
    n_asym = sum(1 for v in _und.values() if len(v) == 2)
    assert len(dd_prompt) == len(dd), "dd_prompt drifted out of alignment with dd"
    return {"pair": pid, "base": base_name, "n_sites": len(dd),
            "n_asym": n_asym,
            "swap_base": swap_base, "swap_algn": swap_algn, "dd": dd,
            "dd_prompt": dd_prompt,
            "own": own, "pos": pos, "cost_f": cost_f, "cost_r": cost_r,
            "cost_prompt": cost_prompt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", help="substring filter on the pair id")
    ap.add_argument("--site-level", action="store_true")
    ap.add_argument("--no-movement", action="store_true",
                    help="skip the true_word_probs section")
    a = ap.parse_args()
    from malign_logits.cache import get_cache
    by = load(get_cache(), a.pair)
    if not by:
        sys.exit("no fc_v1 records match")
    rows = [analyse_pair(p, c) for p, c in sorted(by.items())]

    print("PAIRS %d | sites with all four arms: %d\n"
          % (len(rows), sum(r["n_sites"] for r in rows)))

    #: ---------- per pair ----------
    print("PER PAIR — damage contrasts, mean over sites (nats/token)")
    print("%-26s %6s %10s %10s %10s %10s"
          % ("pair", "sites", "swap_base", "swap_algn", "diff-in-d", "own-word"))
    for r in rows:
        f = lambda v: ("%+.4f" % statistics.mean(v)) if v else "-"
        print("%-26s %6d %10s %10s %10s %10s"
              % (r["base"][:24], r["n_sites"], f(r["swap_base"]),
                 f(r["swap_algn"]), f(r["dd"]), f(r["own"])))
    print("""
  ALL THREE SWAP COLUMNS ARE RISER MINUS FALLER, same sign convention.

  swap_base  how much more fluently the BASE continues after the promoted
             word than after the demoted one. Positive = the riser is the
             easier continuation point even for the model that did not
             promote it.
  swap_algn  the same for the ALIGNED model.
  diff-in-d  swap_algn minus swap_base -- the alignment-SPECIFIC part.
             Positive = the aligned model gains more from the riser than the
             base does. This is the only column the base arm makes readable;
             swap_algn alone cannot distinguish alignment from the word.
  own-word   aligned-after-riser minus base-after-faller. Not a swap: it asks
             whether the two models are equally fluent after the word each
             actually ended up on.""")

    #: ---------- across pairs, unit = pair ----------
    def across(rs, label):
        print("\nACROSS PAIRS%s — unit is the PAIR (sites within a pair share a base)"
              % label)
        print("measure               n       mean         sd  verdict")
        for key in ("swap_base", "swap_algn", "dd", "own"):
            xs = [statistics.mean(r[key]) for r in rs if r[key]]
            if len(xs) < 2:
                print("%-14s UNTESTED (n<2)" % key); continue
            pp, nn = permutation_p(xs)
            print("%-14s %6d %+10.5f %10.5f  %s"
                  % (key, nn, statistics.mean(xs), statistics.pstdev(xs),
                     verdict(statistics.mean(xs), pp, xs)))
    thin = [r for r in rows if r["n_sites"] < MIN_SITES]
    across([r for r in rows if r["n_sites"] >= MIN_SITES],
           "  (pairs with >= %d sites; %d excluded: %s)"
           % (MIN_SITES, len(thin),
              ", ".join("%s n=%d" % (r["base"][:14], r["n_sites"]) for r in thin) or "none"))
    across(rows, "  (ALL pairs, unfiltered — printed so the filter is a choice"
                 " you can see, not one you must trust)")
    print("\n#ORIGINAL#")
    print("%-14s %8s %10s %10s  %s" % ("measure", "n", "mean", "sd", "verdict"))
    for name in ("swap_base", "swap_algn", "dd", "own"):
        vals = [statistics.mean(r[name]) for r in rows if r[name]]
        if not vals:
            print("%-14s %8s %10s %10s  UNTESTED (no pair has all four arms)"
                  % (name, 0, "-", "-"))
            continue
        p, n = permutation_p(vals)
        print("%-14s %8d %+10.5f %10.5f  %s"
              % (name, n, statistics.mean(vals), statistics.pstdev(vals),
                 verdict(statistics.mean(vals), p, vals)))

    #: ---------- resist by position ----------
    print("\nRESIST BY POSITION — SYMMETRIC. Both generation directions.")
    print("  %s" % POS0_NOTE)
    AA = collections.defaultdict(list); BB = collections.defaultdict(list)
    for r in rows:
        for i, v in r["pos"]["A"].items():
            AA[i].extend(v)
        for i, v in r["pos"]["B"].items():
            BB[i].extend(v)
    if AA and BB:
        ks = sorted(set(AA) & set(BB))
        fmt = lambda d: "  ".join("p%d %+.3f" % (i, statistics.mean(d[i])) for i in ks)
        print("  A  base's beams,    self-judge : %s" % fmt(AA))
        print("  B  aligned's beams, self-judge : %s" % fmt(BB))
        print("  SELECTION (A+B)/2              : %s"
              % "  ".join("p%d %+.3f" % (i, (statistics.mean(AA[i]) + statistics.mean(BB[i])) / 2) for i in ks))
        print("  ASYMMETRY (A-B)/2              : %s"
              % "  ".join("p%d %+.3f" % (i, (statistics.mean(AA[i]) - statistics.mean(BB[i])) / 2) for i in ks))
        pa = [x for i, v in AA.items() if i > 0 for x in v]
        pb = [x for i, v in BB.items() if i > 0 for x in v]
        print("\n  pooled 1..n   A %+.4f (n %d) | B %+.4f (n %d)"
              % (statistics.mean(pa), len(pa), statistics.mean(pb), len(pb)))
        print("  selection cost %+.4f  <- BOTH directions positive means beam"
              % ((statistics.mean(pa) + statistics.mean(pb)) / 2))
        print("     search choosing its own maximum, not resistance")
        print("  asymmetry      %+.4f  <- the quantity a one-direction measure"
              % ((statistics.mean(pa) - statistics.mean(pb)) / 2))
        print("     cannot see: negative means the BASE finds the aligned")
        print("     model's continuations stranger than the reverse")
        print("  A median %+.4f, %%>0 %.1f%%  |  B median %+.4f, %%>0 %.1f%%"
              % (statistics.median(pa), 100 * sum(1 for x in pa if x > 0) / len(pa),
                 statistics.median(pb), 100 * sum(1 for x in pb if x > 0) / len(pb)))
        #: **THE POOLED NUMBER ABOVE IS NOT A TEST AND MUST NEVER BE QUOTED AS
        #: ONE.** Its n is token positions -- over two million -- and positions
        #: within a pair share a base model, a tokenizer and a prompt set, so
        #: pooling them is pseudo-replication: the n is enormous and the
        #: independent units are eleven. Every other section of this file uses
        #: the pair; this one printed a pooled mean for a year without one.
        #: **THE ASYMMETRY NEVER TRAVELS WITHOUT ITS LOCALITY CHECK.**
        #: Registered as standing at [4784]. lacan's [4780].2: a scalar cancels
        #: under pooling whenever its SIGN varies across sites -- one dimension
        #: instead of a thousand, rarer and not impossible. A PER-PAIR count of
        #: 18-of-19 cannot speak to that, being already pooled over the pair's
        #: sites. So the per-site sign share is computed here and printed
        #: beside the pooled number, always.
        site_share = []
        for r in rows:
            per_site = {}
            for (role, arm, w, prompt), rec in by[r["pair"]].items():
                if arm != "undisturbed":
                    continue
                sb, sa = rec.get("scored_by_base"), rec.get("scored_by_aligned")
                if not sb or not sa:
                    continue
                first, second = (sb, sa) if role == "base" else (sa, sb)
                vals = [x - y for r1, r2 in zip(first, second)
                        for i, (x, y) in enumerate(zip(r1, r2)) if i > 0]
                if vals:
                    per_site.setdefault(prompt, {})[role] = statistics.mean(vals)
            a_ = [(d["base"] - d["aligned"]) / 2
                  for d in per_site.values() if len(d) == 2]
            if len(a_) >= 5:
                pooled_ = statistics.mean(a_)
                agree = sum(1 for x in a_ if (x < 0) == (pooled_ < 0))
                site_share.append((abs(pooled_), 100.0 * agree / len(a_)))
        per_pair, per_pair_named = [], []
        for r in rows:
            #: NOT `a`/`b` -- `a` is the argparse namespace in this scope and
            #: shadowing it made `a.site_level` fail 80 lines later.
            av = [x for i, v in r["pos"]["A"].items() if i > 0 for x in v]
            bv = [x for i, v in r["pos"]["B"].items() if i > 0 for x in v]
            if av and bv:
                _v = (statistics.mean(av) - statistics.mean(bv)) / 2
                per_pair.append(_v)
                per_pair_named.append((_v, r["base"], r["n_asym"]))
        if len(per_pair) >= 2:
            pp, nn = permutation_p(per_pair)
            neg = sum(1 for x in per_pair if x < 0)
            print("\n  PER-PAIR ASYMMETRY — unit = pair, the honest test")
            print("    mean %+.4f  n=%d  %d of %d negative  %s"
                  % (statistics.mean(per_pair), nn, neg, nn,
                     verdict(statistics.mean(per_pair), pp, per_pair)))
            print("    range %+.4f .. %+.4f"
                  % (min(per_pair), max(per_pair)))
            #: **NAME THE NON-NEGATIVE PAIRS.** "deepseek is the single
            #: reversal" was carried for a night as the one-number answer to
            #: the concentration account, while the printed summary said 29 of
            #: 32 -- i.e. THREE are not negative. A count cannot be checked
            #: against a claim about WHICH, and the claim was the load-bearing
            #: one. One anomaly is an existence proof; three is a pattern that
            #: needs its own account. Printed always, not on a flag, because
            #: the discrepancy was invisible precisely while nobody asked.
            _pos = sorted((x for x in per_pair_named if x[0] >= 0), reverse=True)
            print("    NOT NEGATIVE: %d of %d" % (len(_pos), nn))
            for v, nm, ns in _pos:
                print("       %-30s %+.4f   %d undisturbed prompts" % (nm[:30], v, ns))
            _neg = sorted(x for x in per_pair_named if x[0] < 0)
            print("    most negative, for scale:")
            for v, nm, ns in _neg[:3]:
                print("       %-30s %+.4f   %d undisturbed prompts" % (nm[:30], v, ns))
            #: Bonferroni across the four across-pair families tested in this
            #: file (swap_base, swap_algn, dd, asymmetry): alpha 0.0125.
            print("    Bonferroni across the 4 across-pair tests (alpha 0.0125): %s"
                  % ("SURVIVES" if pp < 0.0125 else "does NOT survive"))
            if len(site_share) >= 4:
                sh = [x for _, x in site_share]
                big = [x for m, x in site_share if m >= 0.10]
                small = [x for m, x in site_share if m < 0.10]
                xs = [m for m, _ in site_share]
                mx, my = statistics.mean(xs), statistics.mean(sh)
                den = (math.sqrt(sum((a - mx) ** 2 for a in xs)) *
                       math.sqrt(sum((b - my) ** 2 for b in sh)))
                rr = (sum((a - mx) * (b - my) for a, b in site_share) / den
                      if den else float("nan"))
                print("    LOCALITY — share of SITES agreeing with their pair's "
                      "pooled sign")
                print("      median %.0f%%  mean %.0f%%  min %.0f%%  over %d pairs"
                      % (statistics.median(sh), statistics.mean(sh),
                         min(sh), len(sh)))
                if big and small:
                    print("      |asym|>=0.10 (n=%d) %.0f%%   |asym|<0.10 (n=%d) %.0f%%"
                          % (len(big), statistics.mean(big),
                             len(small), statistics.mean(small)))
                print("      corr(|pooled|, share) = %+.3f" % rr)
                #: A SUBSET-CARRIED effect shows LOW agreement at LARGE pooled
                #: values. Agreement RISING with effect size is a per-site
                #: constant plus noise -- sites split evenly exactly where the
                #: true effect is ~0. The sign of this correlation is the read.
                print("      positive => per-site constant + noise; negative or "
                      "flat => a subset carries it")

    #: ---------- F14's UNDIRECTED measure, on this data ----------
    #:
    #: F14 asked how much the SWAP CHANGES the continuation distribution, and
    #: reported the aligned model's synt_js exceeding the base's in every
    #: category (+0.032 to +0.106, OLMo, 23k pairs). Its causal framing --
    #: "alignment damages combination" -- is RETRACTED in the finding itself;
    #: the deltas stand as amplification of a pre-existing structure.
    #:
    #: **JS IS SYMMETRIC AND THEREFORE UNDIRECTED.** It cannot distinguish
    #: "the substitution made the continuation WORSE" from "made it
    #: DIFFERENT". That is exactly what the constraint-cost measure above adds,
    #: and why both belong here: one says how far the swap moves the model,
    #: the other says whether it costs it anything.
    print("\nDIVERGENCE (F14-analogue) — JS between the two forced arms")
    print("  undirected: distance, not degradation. Read beside the cost table.")
    jsv = {"base": [], "aligned": []}
    #: kept for the label-permutation control below: (pair, role) -> list of
    #: (faller_marginal, riser_marginal), one entry per prompt.
    marg = collections.defaultdict(list)
    for r in rows:
        cells = by[r["pair"]]
        for pr in sorted({k[3] for k in cells if k[1] == "force_faller"}):
            for role in ("base", "aligned"):
                f = [v for (ro, ar, w, pp), v in cells.items()
                     if ro == role and ar == "force_faller" and pp == pr]
                g = [v for (ro, ar, w, pp), v in cells.items()
                     if ro == role and ar == "force_riser" and pp == pr]
                if f and g:
                    mf, mg = _first_token_marginal(f[0]), _first_token_marginal(g[0])
                    jsv[role].append(_js(mf, mg))
                    marg[(r["pair"], role)].append((mf, mg))
    if jsv["base"] and jsv["aligned"]:
        for role in ("base", "aligned"):
            v = jsv[role]
            print("  %-9s n %d  mean %.4f  sd %.4f" % (role, len(v), statistics.mean(v), statistics.pstdev(v)))
        d = [x - y for x, y in zip(jsv["aligned"], jsv["base"])]
        pp, nn = permutation_p(d)
        print("  DELTA aligned-base %+.4f  %s" % (statistics.mean(d), verdict(statistics.mean(d), pp, d)))
        #: **THE CEILING TRAVELS WITH THE MEAN, ALWAYS.** Two thirds of site
        #: values sat above 0.95 bits on the first pair measured. A mean over
        #: mostly-saturated values cannot move much, and quoting it without
        #: this count invites a reader to treat a floor effect as a null.
        allv = jsv["base"] + jsv["aligned"]
        hi = sum(1 for x in allv if x > 0.95)
        print("  CEILING: %d of %d site-values above 0.95 bits (%.0f%%) -- the"
              % (hi, len(allv), 100 * hi / len(allv)))
        print("  arms explore near-disjoint continuations, so this metric is")
        print("  compressed at the top and has little room to differ. Never")
        print("  quote the mean without this line.")
        print("  support: %.1f distinct first tokens per unit (100 beams)"
              % statistics.mean([len(_first_token_marginal(v))
                                 for r in rows for v in by[r["pair"]].values()
                                 if v.get("beams")][:200]))
        #: **LABEL-PERMUTATION CONTROL: COULD THIS NUMBER HAVE BEEN ANYTHING
        #: ELSE?** Re-pair each faller marginal with a riser marginal from a
        #: DIFFERENT prompt in the same cell and recompute. A quantity that
        #: measures the ARM CONTRAST falls when the arms are randomised; one
        #: that measures the instrument does not move.
        #:
        #: This metric is a prime candidate for not moving, and the reason is
        #: visible two lines up: `_first_token_marginal` RESTRICTS to the top
        #: 100 beams and THEN normalises, so each arm's mass is renormalised
        #: over its own truncated support. Two units that explored disjoint
        #: continuations score ~1 bit whether or not they are the same prompt.
        #: The ceiling count says the same thing from the other side.
        #:
        #: Owed to lacan [4731].2, who proposed permuting the labels and
        #: recomputing as the general form of "an identity does not move, a
        #: finding does" -- strictly better than the roundness tell that
        #: prompted it, since a statistic pinned to a NON-round constant looks
        #: like a modest real effect and passes every roundness check.
        #: **PER PAIR, THEN ACROSS PAIRS.** The first version pooled every
        #: re-pairing from every model into one mean -- the same
        #: pseudo-replication the asymmetry section carried: re-pairings inside
        #: a pair share a base model and a prompt set, so their n is not the
        #: number of independent units. Computed per pair here; the across-pairs
        #: test below uses the pair.
        rng = random.Random(SEED)
        ctrl, per_pair_gap = [], []
        bypair = collections.defaultdict(lambda: {"obs": [], "ctrl": []})
        for (pr, role), pairs in marg.items():
            if len(pairs) < 2:
                continue
            for i, (mf, mg) in enumerate(pairs):
                j = rng.randrange(len(pairs) - 1)
                j = j + 1 if j >= i else j          #: any index but i
                c = _js(mf, pairs[j][1])
                ctrl.append(c)
                bypair[pr]["ctrl"].append(c)
                bypair[pr]["obs"].append(_js(mf, mg))
        for pr, d in sorted(bypair.items()):
            if d["obs"] and d["ctrl"]:
                per_pair_gap.append(statistics.mean(d["obs"]) -
                                    statistics.mean(d["ctrl"]))
        if ctrl:
            obs, per = statistics.mean(allv), statistics.mean(ctrl)
            print("  PERMUTED LABELS: %.4f over %d re-pairings vs %.4f observed"
                  % (per, len(ctrl), obs))
            #: **REPORT THE BACKGROUND AS A FRACTION, NOT THE GAP AS A VERDICT.**
            #: The gap alone invites "non-zero, therefore it measures the arms".
            #: What a reader needs is how much of the QUOTED MEAN any two units
            #: would produce: the cross-prompt value is the floor this metric
            #: cannot go below, so obs-minus-floor is the whole of the arm
            #: signal and everything under it is the instrument.
            print("  cross-prompt reference %.4f = what ANY two units score;"
                  % per)
            print("  same-prompt sits %+.4f from it, %.1f%% of the reference."
                  % (obs - per, 100 * abs(obs - per) / per))
            print("  So the quoted mean is %.0f%% reference and the arm contrast"
                  % (100 * min(obs, per) / max(obs, per)))
            print("  is the REMAINDER. Never quote the mean as arm divergence.")
            if len(per_pair_gap) >= 2:
                pg, ng = permutation_p(per_pair_gap)
                neg = sum(1 for x in per_pair_gap if x < 0)
                print("  PER-PAIR GAP — unit = pair: mean %+.4f  n=%d  %d of %d"
                      % (statistics.mean(per_pair_gap), ng, neg, ng))
                print("    negative  %s"
                      % verdict(statistics.mean(per_pair_gap), pg, per_pair_gap))
                print("    range %+.4f .. %+.4f  — a pair whose gap is ~0 has a"
                      % (min(per_pair_gap), max(per_pair_gap)))
                print("    metric measuring only its own beam truncation.")
        else:
            print("  PERMUTED LABELS: UNTESTED -- no cell has 2+ prompts.")

    #: ---------- constraint cost: the PRIMARY damage measure ----------
    print("\nCOST OF THE CONSTRAINT — each model's unforced beam as reference")
    print("  Forcing any word costs something; that price is the baseline, and")
    print("  it is the same size as the riser-faller effect it was hiding.")
    print("  %-10s %12s %12s %12s" % ("model", "cost FALLER", "cost RISER", "fall-rise"))
    #: **UNIT = PAIR HERE TOO.** The first version pooled all 339 sites, which
    #: contradicted the rule stated at the top of this file and gave the
    #: interaction an implicit n of 339 when sites within a pair share a base
    #: model and are not independent. Per-pair means first, then across pairs.
    for role in ("base", "aligned"):
        cf = [statistics.mean(r["cost_f"][role]) for r in rows if r["cost_f"][role]]
        cr = [statistics.mean(r["cost_r"][role]) for r in rows if r["cost_r"][role]]
        if cf and cr:
            print("  %-10s %12.4f %12.4f %12.4f  (n=%d PAIRS)"
                  % (role, statistics.mean(cf), statistics.mean(cr),
                     statistics.mean(cf) - statistics.mean(cr), len(cf)))
    inter_by_pair = []
    for r in rows:
        if not (r["cost_f"]["base"] and r["cost_f"]["aligned"]):
            continue
        inter_by_pair.append(
            (statistics.mean(r["cost_f"]["aligned"]) - statistics.mean(r["cost_r"]["aligned"]))
            - (statistics.mean(r["cost_f"]["base"]) - statistics.mean(r["cost_r"]["base"])))
    if inter_by_pair:
        pp, nn = permutation_p(inter_by_pair)
        print("  ALIGNMENT-SPECIFIC interaction: %+.4f  n=%d pairs  %s"
              % (statistics.mean(inter_by_pair), nn,
                 verdict(statistics.mean(inter_by_pair), pp, inter_by_pair)))
        print("     positive = the aligned model pays MORE than the base to say")
        print("     the word it demoted, which is the damage direction;")
        print("     negative = it pays less, which is not.")

    #: ---------- the confound's own direction ----------
    print("\nPROBABILITY CONFOUND — does it predict the sign we observe?")
    print("  The faller and riser are NOT matched on prior probability. But the")
    print("  gap widens under alignment BY CONSTRUCTION, so a purely")
    print("  probability-driven world predicts a POSITIVE interaction.")
    print("  %-22s %8s %9s %9s %9s %8s %8s"
          % ("pair", "sites", "gap_base", "gap_algn", "widening", "f_rank", "r_rank"))
    import json as _json
    man = {}
    for mf in ("data/fc_manifest_mps.json", "data/fc_manifest_vast.json"):
        fp = os.path.join(ROOT, mf)
        if os.path.exists(fp):
            for pr in _json.load(open(fp))["pairs"]:
                man["%s>%s" % (pr["base"], pr["aligned"])] = pr["sites"]
    from malign_logits.cache import get_cache as _gc
    cm2 = _gc()
    for r in rows:
        sites = man.get(r["pair"])
        if not sites:
            continue
        c = confound_sign(cm2, r["pair"], sites)
        if c:
            print("  %-22s %8d %+9.3f %+9.3f %+9.3f %8.0f %8.0f"
                  % (r["base"][:20], c["n"], c["gap_base"], c["gap_aligned"],
                     c["widening"], c["faller_rank"], c["riser_rank"]))
    print("  widening > 0 means the confound predicts a POSITIVE interaction.")
    print("  Read it against the ALIGNMENT-SPECIFIC number above: opposite")
    print("  signs mean the confound cannot be producing the observation.")
    print("  It never licenses the MAGNITUDE, only the direction.")

    #: ---------- within-word control ----------
    print("\nWITHIN-WORD CONTROL — words appearing as BOTH faller and riser")
    print("  Same word, opposite roles at different sites: lexical identity is")
    print("  held fixed by construction, so no probability matching is needed.")
    fal = collections.defaultdict(list); ris = collections.defaultdict(list)
    for r in rows:
        cells = by[r["pair"]]
        for (role, arm, w, p), rec in cells.items():
            if role != "aligned" or not w:
                continue
            v = mean_lp(rec, "aligned", 0, 9)
            if v is None:
                continue
            (fal if arm == "force_faller" else ris)[w].append(v)
    both = sorted(set(fal) & set(ris))
    if both:
        deltas = [statistics.mean(fal[w]) - statistics.mean(ris[w]) for w in both]
        pp, nn = permutation_p(deltas)
        print("  %d words in both roles: %s" % (len(both), ", ".join(both[:10])))
        print("  cost as faller minus cost as riser: mean %+.4f  %s"
              % (statistics.mean(deltas), verdict(statistics.mean(deltas), pp, deltas)))
        print("  negative = the SAME word continues better when it is the")
        print("  promoted one, which no probability difference can explain.")
    else:
        print("  none yet -- needs more pairs. UNTESTED.")

    #: ---------- the full damage grid: 2 generators x 2 scorers ----------
    print("\nDAMAGE GRID — riser minus faller, every generator x scorer cell")
    print("  A scorer artefact would show as the two SCORER rows disagreeing;")
    print("  a generator artefact as the two GENERATOR rows disagreeing.")
    print("  %-28s %10s %10s %12s" % ("generator / scorer", "faller", "riser", "riser-faller"))
    for gen in ("base", "aligned"):
        for sc in ("base", "aligned"):
            fs, rs = [], []
            for r in rows:
                cells = by[r["pair"]]
                fs += [mean_lp(v, sc) for (ro, ar, w, p), v in cells.items()
                       if ro == gen and ar == "force_faller" and mean_lp(v, sc) is not None]
                rs += [mean_lp(v, sc) for (ro, ar, w, p), v in cells.items()
                       if ro == gen and ar == "force_riser" and mean_lp(v, sc) is not None]
            if fs and rs:
                print("  %-28s %10.4f %10.4f %+12.4f"
                      % ("gen=%s scored by %s" % (gen, sc), statistics.mean(fs),
                         statistics.mean(rs), statistics.mean(rs) - statistics.mean(fs)))

    if a.site_level:
        print("\nSITE-LEVEL rows")
        for r in rows:
            for i, d in enumerate(r["dd"]):
                print("  %-24s %3d %+0.5f" % (r["base"][:22], i, d))

    #: ---------- markedness: the design's own within-stem control ----------
    print("\nMARKED vs UNMARKED — the contrast the sample was built to carry")
    print("  Each stem yields two prompts differing in the marked word:")
    print("  'She SQUEEZED the rabbit in her grip and' / 'She CRADLED ...'.")
    meta = site_meta()
    #: **A JOIN THAT MISSES EVERYTHING LOOKS LIKE A POPULATION THAT HAS
    #: NOTHING.** Every markedness result below joins the beam records to the
    #: frozen sample on the PROMPT STRING. If that key ever drifts -- a
    #: whitespace change, a re-encoded CSV, a manifest built from a different
    #: sample -- `meta.get(p)` returns None for every row, each section prints
    #: a shorter table or UNTESTED, and nothing raises. lacan hit this shape
    #: today from the other side ([4769].3): a `gi:` prefix against a
    #: `gi_primary` key made 112 clusters report zero significant components,
    #: which read as the substantive finding "these carry nothing".
    #:
    #: So the join RATE is asserted, not assumed. Not >0 -- a single match
    #: would pass that -- but a floor that only a working key can clear.
    _keys = {p for r in rows for p in r["dd_prompt"]}
    _hit = sum(1 for p in _keys if p in meta)
    _rate = _hit / len(_keys) if _keys else 0.0
    print("  join to the frozen sample: %d of %d prompt keys matched (%.0f%%)"
          % (_hit, len(_keys), 100 * _rate))
    assert _rate >= 0.80, (
        "prompt-key join matched only %.0f%% (%d of %d) -- the markedness "
        "results below would be computed on whatever happened to match. Check "
        "that data/beam_sample_105.csv is the sample these pairs were built "
        "from." % (100 * _rate, _hit, len(_keys)))
    #: **THE UNPAIRED SPLIT IS REPORTED BUT IT IS NOT THE CONTROL.** Site
    #: selection keeps a prompt only if it has both a faller and a riser under
    #: CANONICAL, and that filter is not markedness-neutral: MARKED prompts
    #: survive at 19.0% against UNMARKED 17.6% locally (29.6% / 27.5% remote).
    #: A small gap, consistent in direction, and enough that an unpaired
    #: comparison is confounded with which stems survived. The WITHIN-STEM
    #: form below is immune to it -- both members come from one stem or the
    #: stem contributes nothing.
    #: **KEYED BY PAIR, NOT TWO PARALLEL LISTS.** The first version appended to
    #: separate MARKED/UNMARKED lists and zipped them: with 10 pairs having
    #: marked sites and 11 having unmarked, zip silently truncated to 10 and
    #: differenced MISMATCHED MODELS against each other. Two lists that must
    #: stay index-aligned are a defect waiting for the first asymmetric row.
    unp = collections.defaultdict(dict)
    paired = []
    for r in rows:
        vals = collections.defaultdict(dict)
        per = {"MARKED": [], "UNMARKED": []}
        for v, p in zip(r["dd"], r["dd_prompt"]):
            m = meta.get(p)
            if not m:
                continue
            per[m["member"]].append(v)
            vals[m["stem"]][m["member"]] = v
        for k in ("MARKED", "UNMARKED"):
            if per[k]:
                unp[r["pair"]][k] = statistics.mean(per[k])
        both = [(d["MARKED"], d["UNMARKED"]) for d in vals.values()
                if len(d) == 2]
        if both:
            paired.append((statistics.mean([x - y for x, y in both]), len(both)))
    for k in ("MARKED", "UNMARKED"):
        vs = [v[k] for v in unp.values() if k in v]
        if vs:
            print("  %-9s dd mean %+.4f over %d pairs (unpaired, confounded"
                  % (k, statistics.mean(vs), len(vs)))
            print("            with differential site survival -- see note)")
    #: only pairs holding BOTH cells enter the difference
    d = [v["MARKED"] - v["UNMARKED"] for v in unp.values() if len(v) == 2]
    if len(d) >= 2:
        pp, nn = permutation_p(d)
        print("  UNPAIRED  marked-unmarked %+.4f  %s"
              % (statistics.mean(d), verdict(statistics.mean(d), pp, d)))
    if paired:
        xs = [x for x, _ in paired]
        stems = sum(n for _, n in paired)
        pp, nn = permutation_p(xs)
        print("  WITHIN-STEM  marked-unmarked %+.4f  %s"
              % (statistics.mean(xs), verdict(statistics.mean(xs), pp, xs)))
        #: DERIVED, NEVER TYPED. An earlier draft hardcoded "23%" from a
        #: one-off count; the fraction depends on which pairs are in the run.
        seen = set()
        for r in rows:
            for p in r["dd_prompt"]:
                if p in meta:
                    seen.add((r["pair"], meta[p]["stem"], meta[p]["member"]))
        allst = {(a, b) for a, b, _ in seen}
        frac = 100.0 * stems / len(allst) if allst else 0.0
        print("  %d pairs contributing, %d of %d pair-stems keep BOTH members"
              % (len(xs), stems, len(allst)))
        print("  (%.0f%%); the rest lost one to site selection. This is the" % frac)
        print("  clean contrast and it is thin; quote its n beside it, never")
        print("  the unpaired number alone.")
    else:
        print("  WITHIN-STEM: UNTESTED — no stem retains both members here.")

    #: **THE ARMS SPLIT SEPARATELY, WHICH IS WHERE lacan [4735].4 SAYS TO LOOK.**
    #: On six lexicons he finds WITHDRAWAL is transgression-specific (fallers
    #: larger in the marked twin) while SUBSTITUTION is universal (risers larger
    #: in the NEUTRAL twin). If that holds here, the two arms carry markedness
    #: in OPPOSITE directions and `dd` -- which differences them -- is the one
    #: statistic guaranteed to show nothing. A null in a difference is not a
    #: null in its components, and this is the cheap way to tell them apart.
    print("\n  ARMS SPLIT SEPARATELY (the dd null may be a cancellation)")
    for arm, key in (("FALLER cost", "cost_f"), ("RISER cost", "cost_r")):
        for role in ("base", "aligned"):
            diffs = []
            for r in rows:
                per = {"MARKED": [], "UNMARKED": []}
                for v, p in zip(r[key][role], r["cost_prompt"][role]):
                    m = meta.get(p)
                    if m:
                        per[m["member"]].append(v)
                if per["MARKED"] and per["UNMARKED"]:
                    diffs.append(statistics.mean(per["MARKED"]) -
                                 statistics.mean(per["UNMARKED"]))
            if len(diffs) >= 2:
                pp, nn = permutation_p(diffs)
                print("    %-11s %-7s marked-unmarked %+.4f  n=%d  %s"
                      % (arm, role, statistics.mean(diffs), nn,
                         verdict(statistics.mean(diffs), pp, diffs)))
            else:
                print("    %-11s %-7s UNTESTED (n<2)" % (arm, role))
    print("    lacan's prediction: FALLER cost MORE marked (positive),")
    print("    RISER cost LESS marked (negative). Opposite signs would mean")
    print("    the dd null above is a cancellation, not an absence.")

    #: ---- lacan [4735].4 on ITS OWN quantity, not on continuation cost ----
    #: The arms-split above tests markedness on CONTINUATION LOGPROB COST and
    #: is underpowered against the ~0.037 gap he reports. His quantity is the
    #: MOVEMENT ITSELF -- how much probability the word lost -- which is in the
    #: true_word_probs stash and needs no beam generation. Same population,
    #: same pairs, his measurement. This is the test that can actually speak
    #: to his claim; the cost version cannot.
    if not a.no_movement:
        print("\n  MOVEMENT MAGNITUDE by markedness — lacan [4735].4's own quantity")
        try:
            from malign_logits.movement import movement, CANONICAL, RESIDUAL_KEY
            sys.path.insert(0, os.path.join(ROOT, "meta", "M01_displacement",
                                            "scripts"))
            from m05_sites import prepare
            TWP = dict(dict_sha="b16011275c42955c", mode="raw",
                       rule_version=3, theta=0.001)
            st = get_cache()._stash("true_word_probs")

            def mv_for(model, prompt):
                k = dict(TWP); k["model"] = model; k["prompt"] = prompt
                try:
                    v = st[k]
                except Exception:
                    return None
                rows_ = v.get("rows") if isinstance(v, dict) else None
                if not rows_:
                    return None
                o, pr = prepare(rows_)
                return {w: pr[w] for w in o}

            fall, rise, fall_s, rise_s = [], [], [], []
            nfall, nrise, szfall, szrise = [], [], [], []
            cv = {'f': [], 'r': []}
            cvpair = []
            link = []
            fw_top, fw_tail, fwf_top, fwf_tail = [], [], [], []
            splitacc = {'FN': [], 'LEX': []}
            entacc = []
            argmax_fn, mass_fn, cand_fn = [], [], []
            mpt = []
            cnt = {'MARKED': {'f': [], 'r': []},
                   'UNMARKED': {'f': [], 'r': []}}
            for r in rows:
                bm, am = r["pair"].split(">")[0], r["pair"].split(">")[1]
                percv = {'f': [], 'r': []}
                ent = {'MARKED': [], 'UNMARKED': []}
                split = {'FN': {'MARKED': [], 'UNMARKED': []},
                         'LEX': {'MARKED': [], 'UNMARKED': []}}
                per = {"MARKED": {"f": [], "r": [], "fs": [], "rs": []},
                       "UNMARKED": {"f": [], "r": [], "fs": [], "rs": []}}
                for p in set(r["dd_prompt"]):
                    m = meta.get(p)
                    if not m:
                        continue
                    P, Q = mv_for(bm, p), mv_for(am, p)
                    if not P or not Q:
                        continue
                    mvo = movement(P, Q, CANONICAL)
                    F = [w for w in mvo.fallers if w != RESIDUAL_KEY]
                    R = [w for w in mvo.risers if w != RESIDUAL_KEY]
                    key = mvo.excess if mvo.rule.null_test else mvo.delta
                    if F:
                        topf = min(F, key=lambda w: mvo.delta.get(w, 0.0))
                        mag = abs(mvo.delta.get(topf, 0.0))
                        per[m["member"]]["f"].append(mag)
                        #: **SPLIT BY WHETHER THE TOP FALLER IS LEXICAL.** 55.4%
                        #: of top fallers are closed-class, so the marked-minus-
                        #: unmarked magnitude result I reported at [4737] -- and
                        #: which lacan recorded as confirmation of his
                        #: withdrawal claim -- could be a fact about `the` and
                        #: `was` rather than about transgressive content. If it
                        #: survives only on the closed-class half it is not the
                        #: result either of us thought.
                        cls = ("FN" if topf.lower() in FUNCTION_WORDS
                               else "LEX")
                        split[cls][m["member"]].append(mag)
                        #: SUM over all fallers, not just the largest
                        per[m["member"]]["fs"].append(
                            sum(abs(mvo.delta.get(w, 0.0)) for w in F))
                    if R:
                        per[m["member"]]["r"].append(
                            max(key.get(w, 0.0) for w in R))
                        #: **SUM OVER ALL RISERS — lacan [4738].6.** His finding
                        #: 14 has fallers FEW AND LARGE and risers MANY AND
                        #: SMALL (206 against 36, 3.8x per category). If riser
                        #: mass is that diffuse, a top-riser statistic measures
                        #: the head of a distribution whose action is the tail,
                        #: and my riser null would be an artefact of the
                        #: summary rather than a fact about markedness.
                        per[m["member"]]["rs"].append(
                            sum(key.get(w, 0.0) for w in R))
                    #: **MOVERS PER SITE — lacan finding 14 tested on THIS
                    #: population (his [4740].4: a test of his finding on my
                    #: data, so it is mine to run).** He has fallers FEW AND
                    #: LARGE, risers MANY AND SMALL. That prediction already
                    #: earned its keep: it named which of my four summary
                    #: statistics would be unstable BEFORE the instability was
                    #: observed. Counted here directly.
                    nfall.append(len(F)); nrise.append(len(R))
                    #: **THE SHARPNESS CONFOUND (lacan [4753].5).** His two DEPTH
                    #: detections are closed-class in BOTH directions at near-
                    #: identical magnitude (+5.2% down, +5.3% up). An effect
                    #: moving function words equally far each way is not
                    #: withdrawal -- it is the marked prompt having a sharper
                    #: distribution, so every delta scales. Same family as the
                    #: entropy confound this campaign already booked twice.
                    #: Measured on the BASE distribution, pre-alignment.
                    #: **WHY IS MY MOVEMENT VOCABULARY 55.4% CLOSED AGAINST
                    #: HIS 36.2%? (lacan [4755].2 names this as where to look.)**
                    #: My stems are all mid-clause narrative continuations --
                    #: "...in her grip and ___" -- where the next word is
                    #: structurally likely to be closed-class. If the BASE
                    #: argmax is usually a function word here, the population
                    #: difference is a property of the prompt design and not of
                    #: alignment, and the class disagreement has a home.
                    if P:
                        #: **THE THIRD COMPARATOR, AND THE ONLY ONE WITH
                        #: GROUNDS (lacan [4757].2 says he has none; he names
                        #: them himself and does not follow them).** Argmax is
                        #: ONE word and cannot be the base rate for a set of
                        #: ~15 movers. Total mass includes the whole tail,
                        #: most of which can never be a faller because a faller
                        #: must clear CANONICAL's floor. The denominator a
                        #: mover is actually drawn from is the ELIGIBLE POOL:
                        #: the words present in the prepared base distribution,
                        #: counted as TYPES because a faller is a type and the
                        #: question is what share of types are closed-class.
                        #: **I ARGUED THE HUMP WAS ALGEBRAIC AND THE DATA SAYS
                        #: NO.** The argument: mass-per-type has the closed-type
                        #: COUNT in its denominator, so raising it should starve
                        #: the pool of candidates, and a hump follows from the
                        #: opposition. Measured, the two are POSITIVELY related
                        #: (r=+0.353, 793 sites) -- sites with more closed types
                        #: also carry more closed mass, more than
                        #: proportionally, and `len(P)` and `cmass` vary too, so
                        #: nothing was forced by the algebra. The reversal lacan
                        #: found above 1.4x is UNEXPLAINED, and his candidate
                        #: (few types holding everything) is not supported here.
                        #: Kept because a refuted mechanism that was nearly
                        #: asserted from a formula is worth the two lines.
                        nclosed = sum(1 for w in P if w.lower() in FUNCTION_WORDS)
                        cmass = sum(v for w, v in P.items()
                                    if w.lower() in FUNCTION_WORDS)
                        tot = sum(P.values()) or 1.0
                        if nclosed and len(P):
                            mpt.append(((cmass / tot) / (nclosed / len(P)),
                                        nclosed))
                        cand_fn.append(
                            sum(1.0 for w in P if w.lower() in FUNCTION_WORDS)
                            / len(P))
                        top1 = max(P, key=lambda w: P[w])
                        argmax_fn.append(1.0 if top1.lower() in FUNCTION_WORDS
                                         else 0.0)
                        mass_fn.append(sum(v for w, v in P.items()
                                           if w.lower() in FUNCTION_WORDS)
                                       / (sum(P.values()) or 1.0))
                    ent[m["member"]].append(
                        -sum(x * math.log(x) for x in P.values() if x > 0))
                    #: **IS THE RISER TAIL FUNCTION WORDS? (lacan [4749].2)**
                    #: He finds pure-riser mass landing on `and`/`be`/`that`/
                    #: `would` -- renormalisation absorbed by high-frequency
                    #: closed class. If that also describes the TAIL at
                    #: both-arms sites, it is a better mechanism for my
                    #: top-vs-sum instability than dispersion ever was: the top
                    #: riser would be lexical and the sum would drag in
                    #: function words. Top against tail, counted here.
                    if len(R) >= 2:
                        ordR = sorted(R, key=lambda w: -key.get(w, 0.0))
                        fw_top.append(1.0 if ordR[0].lower() in FUNCTION_WORDS
                                      else 0.0)
                        tail = ordR[1:]
                        fw_tail.append(sum(1.0 for w in tail
                                           if w.lower() in FUNCTION_WORDS)
                                       / len(tail))
                    if len(F) >= 2:
                        ordF = sorted(F, key=lambda w: mvo.delta.get(w, 0.0))
                        fwf_top.append(1.0 if ordF[0].lower() in FUNCTION_WORDS
                                       else 0.0)
                        tf = ordF[1:]
                        fwf_tail.append(sum(1.0 for w in tf
                                            if w.lower() in FUNCTION_WORDS)
                                        / len(tf))
                    #: counts split by markedness -- my population is stratified
                    #: for transgression and lacan's walk is not, which is a
                    #: candidate explanation for my 0.62 against his 0.93/1.11.
                    cnt[m["member"]]["f"].append(len(F))
                    cnt[m["member"]]["r"].append(len(R))
                    #: **SPREAD, the [4743] pre-registered arm.** CV not sd:
                    #: the arms have different means and a raw sd would track
                    #: the mean rather than the dispersion.
                    for tg, WS in (("f", F), ("r", R)):
                        if len(WS) >= 3:
                            ds = [abs(mvo.delta.get(w, 0.0)) for w in WS]
                            mu = statistics.mean(ds)
                            if mu > 0:
                                cv[tg].append(statistics.pstdev(ds) / mu)
                                percv[tg].append(statistics.pstdev(ds) / mu)
                    if F:
                        szfall.append(statistics.mean(
                            [abs(mvo.delta.get(w, 0.0)) for w in F]))
                    if R:
                        #: **abs(delta) FOR BOTH ARMS, not excess for risers.**
                        #: `excess` is delta minus the renormalisation the null
                        #: expects, so comparing mean-excess-per-riser against
                        #: mean-|delta|-per-faller compares two scales and the
                        #: ratio means nothing. The counts above are unaffected
                        #: -- a count does not depend on which quantity ranks it.
                        szrise.append(statistics.mean(
                            [abs(mvo.delta.get(w, 0.0)) for w in R]))
                if ent["MARKED"] and ent["UNMARKED"]:
                    entacc.append(statistics.mean(ent["MARKED"]) -
                                  statistics.mean(ent["UNMARKED"]))
                for cls in ("FN", "LEX"):
                    if split[cls]["MARKED"] and split[cls]["UNMARKED"]:
                        splitacc[cls].append(
                            statistics.mean(split[cls]["MARKED"]) -
                            statistics.mean(split[cls]["UNMARKED"]))
                if percv["f"] and percv["r"]:
                    cvpair.append(statistics.mean(percv["r"]) -
                                  statistics.mean(percv["f"]))
                    #: **THE DISCRIMINATING TEST (lacan [4745].2).** A spread
                    #: result that merely lands is a correlate. What makes it a
                    #: MECHANISM is that the pairs with the most riser
                    #: dispersion are the ones where top-vs-sum actually
                    #: diverges in the markedness estimate it is invoked to
                    #: explain. Paired here per pair: mean riser CV against the
                    #: top-minus-sum discrepancy in the riser arm.
                    if (per["MARKED"]["r"] and per["UNMARKED"]["r"]
                            and per["MARKED"]["rs"] and per["UNMARKED"]["rs"]):
                        top = (statistics.mean(per["MARKED"]["r"]) -
                               statistics.mean(per["UNMARKED"]["r"]))
                        sm = (statistics.mean(per["MARKED"]["rs"]) -
                              statistics.mean(per["UNMARKED"]["rs"]))
                        link.append((statistics.mean(percv["r"]),
                                     abs(top - sm)))
                for tag, acc in (("f", fall), ("r", rise),
                                 ("fs", fall_s), ("rs", rise_s)):
                    if per["MARKED"][tag] and per["UNMARKED"][tag]:
                        acc.append(statistics.mean(per["MARKED"][tag]) -
                                   statistics.mean(per["UNMARKED"][tag]))
            for lbl, acc, pred in (
                    ("FALLER top |d|", fall, "POSITIVE"),
                    ("FALLER sum |d|", fall_s, "POSITIVE"),
                    ("RISER  top exc", rise, "NEGATIVE"),
                    ("RISER  sum exc", rise_s, "NEGATIVE")):
                if len(acc) >= 2:
                    pp, nn = permutation_p(acc)
                    print("    %-14s marked-unmarked %+.5f  n=%d  %s"
                          % (lbl, statistics.mean(acc), nn,
                             verdict(statistics.mean(acc), pp, acc)))
                    print("    %-14s   he predicts %s" % ("", pred))
                else:
                    print("    %-14s UNTESTED (n<2)" % lbl)
            #: lacan finding 14 on this population, site-level counts
            if nfall and nrise:
                print("\n    MOVERS PER SITE — lacan finding 14 tested here")
                print("      fallers %.2f per site (median %.0f) | risers %.2f "
                      "(median %.0f) over %d sites"
                      % (statistics.mean(nfall), statistics.median(nfall),
                         statistics.mean(nrise), statistics.median(nrise),
                         len(nfall)))
                print("      ratio risers:fallers %.2f  — he predicts >1"
                      % (statistics.mean(nrise) / statistics.mean(nfall)
                         if statistics.mean(nfall) else float("nan")))
            if szfall and szrise:
                print("      mean |delta| per FALLER %.5f | per RISER %.5f  "
                      "ratio %.2fx"
                      % (statistics.mean(szfall), statistics.mean(szrise),
                         statistics.mean(szfall) / statistics.mean(szrise)
                         if statistics.mean(szrise) else float("nan")))
                print("      he predicts fallers LARGER per mover (his 3.8x)")
            #: lacan [4742].1 is right that one number will not do
            if nfall and nrise:
                sh = sum(1 for x, y in zip(nrise, nfall) if x > y)
                print("      risers OUTNUMBER fallers at %.1f%% of sites; "
                      "fallers at %.1f%%"
                      % (100.0 * sh / len(nfall),
                         100.0 * sum(1 for x, y in zip(nrise, nfall) if y > x)
                         / len(nfall)))
            for k in ("MARKED", "UNMARKED"):
                if cnt[k]["f"]:
                    print("      %-9s fallers %.2f | risers %.2f per site (n=%d)"
                          % (k, statistics.mean(cnt[k]["f"]),
                             statistics.mean(cnt[k]["r"]), len(cnt[k]["f"])))
            if cv["f"] and cv["r"]:
                print("\n    SPREAD (CV of |delta| within a site) — [4743] arm")
                print("      faller CV %.4f (n=%d) | riser CV %.4f (n=%d)"
                      % (statistics.mean(cv["f"]), len(cv["f"]),
                         statistics.mean(cv["r"]), len(cv["r"])))
                print("      predicted BEFORE computing: riser CV the LARGER")
                if len(cvpair) >= 2:
                    pp, nn = permutation_p(cvpair)
                    pos = sum(1 for x in cvpair if x > 0)
                    print("      PER-PAIR riser-minus-faller CV %+.4f  n=%d  "
                          "%d of %d positive  %s"
                          % (statistics.mean(cvpair), nn, pos, nn,
                             verdict(statistics.mean(cvpair), pp, cvpair)))
                    print("      (the pooled line above is 700/664 SITES and is"
                          " not a test -- sites within a pair share a base)")
                if argmax_fn:
                    print("\n    WHY IS MY MOVEMENT VOCABULARY CLOSED-HEAVY? "
                          "(lacan [4755].2)")
                    print("      BASE argmax is closed-class at %.1f%% of sites"
                          "  (n=%d)"
                          % (100 * statistics.mean(argmax_fn), len(argmax_fn)))
                    print("      BASE probability mass on closed class: %.1f%%"
                          % (100 * statistics.mean(mass_fn)))
                    if cand_fn:
                        print("      ELIGIBLE POOL (types in the prepared base "
                              "distribution): %.1f%%"
                              % (100 * statistics.mean(cand_fn)))
                        print("      -> top fallers %.1f%% against a candidate "
                              "pool of %.1f%%"
                              % (100 * statistics.mean(fwf_top)
                                 if fwf_top else float("nan"),
                                 100 * statistics.mean(cand_fn)))
                    if len(mpt) >= 4:
                        xs = [a for a, _ in mpt]; ys = [float(b) for _, b in mpt]
                        mx, my = statistics.mean(xs), statistics.mean(ys)
                        den = (math.sqrt(sum((a - mx) ** 2 for a in xs)) *
                               math.sqrt(sum((b - my) ** 2 for b in ys)))
                        r = (sum((a - mx) * (b - my) for a, b in mpt) / den
                             if den else float("nan"))
                        print("      mass-per-type vs closed-TYPE COUNT: "
                              "r=%+.3f over %d sites" % (r, len(mpt)))
                        print("      POSITIVE, so they do NOT oppose: the "
                              "count-starvation account of")
                        print("      lacan's reversal above 1.4x is not "
                              "supported here. Reversal UNEXPLAINED.")
                    print("      argmax is ONE word; mass includes a tail that "
                          "cannot clear the floor;")
                    print("      the pool is what a faller is actually drawn "
                          "from -- that is the denominator")
                if len(entacc) >= 2:
                    pp, nn = permutation_p(entacc)
                    print("\n    SHARPNESS CONFOUND (lacan [4753].5) — BASE "
                          "entropy, pre-alignment")
                    print("      marked-unmarked %+.5f nats  n=%d  %s"
                          % (statistics.mean(entacc), nn,
                             verdict(statistics.mean(entacc), pp, entacc)))
                    print("      NEGATIVE = marked prompts SHARPER, which would"
                          " inflate every delta")
                    print("      and is his candidate explanation for the "
                          "closed-class detections")
                print("\n    IS THE [4737] WITHDRAWAL RESULT LEXICAL? "
                      "top faller split by word class")
                for cls, lbl in (("LEX", "top faller OPEN-class"),
                                 ("FN", "top faller CLOSED-class")):
                    xs = splitacc[cls]
                    if len(xs) >= 2:
                        pp, nn = permutation_p(xs)
                        print("      %-26s marked-unmarked %+.5f  n=%d  %s"
                              % (lbl, statistics.mean(xs), nn,
                                 verdict(statistics.mean(xs), pp, xs)))
                    else:
                        print("      %-26s UNTESTED (n<2)" % lbl)
                print("      the pooled result was +0.00294 detected p=0.0279")
                if fw_top and fw_tail:
                    print("\n    IS THE RISER TAIL FUNCTION WORDS? "
                          "(lacan [4749].2, on BOTH-ARMS sites)")
                    print("      RISER  top is closed-class %.1f%% of sites | "
                          "tail %.1f%% of words  (n=%d sites)"
                          % (100 * statistics.mean(fw_top),
                             100 * statistics.mean(fw_tail), len(fw_top)))
                    if fwf_top and fwf_tail:
                        print("      FALLER top is closed-class %.1f%% | "
                              "tail %.1f%%  (n=%d)"
                              % (100 * statistics.mean(fwf_top),
                                 100 * statistics.mean(fwf_tail), len(fwf_top)))
                    print("      a riser tail much heavier in closed class than"
                          " its top would")
                    print("      explain top-vs-sum directly, and better than "
                          "dispersion does")
                if len(link) >= 4:
                    xs = [a for a, _ in link]; ys = [b for _, b in link]
                    mx, my = statistics.mean(xs), statistics.mean(ys)
                    num = sum((a - mx) * (b - my) for a, b in link)
                    den = (math.sqrt(sum((a - mx) ** 2 for a in xs)) *
                           math.sqrt(sum((b - my) ** 2 for b in ys)))
                    r = num / den if den else float("nan")
                    #: permutation on the PAIRING, not sign-flip: shuffle y
                    rng2 = random.Random(SEED)
                    hits = 0
                    for _ in range(DRAWS):
                        sh = ys[:]; rng2.shuffle(sh)
                        n2 = sum((a - mx) * (b - my) for a, b in zip(xs, sh))
                        if abs(n2 / den if den else 0) >= abs(r):
                            hits += 1
                    print("\n    DOES SPREAD EXPLAIN THE INSTABILITY? "
                          "(lacan [4745].2)")
                    print("      riser CV vs |top-minus-sum| in the riser "
                          "markedness estimate")
                    print("      r=%+.3f  n=%d pairs  permutation p=%.4f"
                          % (r, len(link), (hits + 1) / (DRAWS + 1)))
                    print("      a correlate becomes a mechanism only if this "
                          "is positive AND detected")
        except Exception as e:                       #: never kill the run
            print("    UNAVAILABLE: %s: %s" % (type(e).__name__, e))
            print("    (the beam results above are unaffected)")

    print("\nEVERY test computed is printed above; none was selected after "
          "looking. Unit = pair. Seed %d, %d permutation draws." % (SEED, DRAWS))


if __name__ == "__main__":
    main()
