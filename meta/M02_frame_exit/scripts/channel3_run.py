#!/usr/bin/env python
"""channel3_run.py — the FROZEN channel-3 analysis. Spec: 85fd7d10.

    channel3_run.py --selftest    known-answer column, runs first
    channel3_run.py               the frozen analysis

WHAT IS FROZEN AND WHAT IS NOT. The primary, its population, its unit, its
test and its decision rule are fixed by
`meta/M02_frame_exit/registrations/spec_channel3_renewed_displacement.md` at
commit 85fd7d10 and are not parameters of this file. Anything this script
prints beyond them is EXPLORATORY and labelled so, per RH's ruling booked at
spec section 10: **a freeze binds the pre-registered claim, it does not close
the question.**

THE PRIMARY

    D(s,a) = mean_lp(aligned's beams | under aligned)
           - mean_lp(base's beams    | under aligned)
    delta(s) = D(s, force_faller) - D(s, force_riser)      delta > 0 = renewed displacement

REQUIRED CO-REPORT (registrar [5015]): the four scoring terms, because on Y the
composite concealed a one-term effect inside a two-composite symmetry.

    A|A aligned text, aligned scorer      A|B aligned text, base scorer
    B|A base text,    aligned scorer      B|B base text,    base scorer

ORDER MATTERS AND IS FIXED: positive control FIRST, then primary, then the four
terms, then secondaries. The control BOUNDS what the instrument detects; it
does NOT void the primary ([5013], accepted [5015]) -- control and primary are
not nested, because D carries a model-level constant that delta differences out
and the control does not.

SOURCES. wave3-lexical from the `beam_fc` stash (design lives in the VALUE, not
the key). newlin-lexical from the rsynced jsonl, which is not yet ingested --
read directly rather than blocking this on an ingest that should not run
unattended.
"""
import argparse
import collections
import glob
import json
import math
import os
import statistics
import sys

ROOT = os.path.expanduser("~/github/malign-logits")
sys.path.insert(0, ROOT)

SEED = 20260808
NBOOT = 10000
NEWLIN_GLOB = os.path.join(ROOT, "data/raw/fc_newlin_out/*.jsonl")
#: the 12 checkpoints whose generation_config carries do_sample=True ([4996]).
SAMPLERS = {
    "HuggingFaceTB/SmolLM3-3B", "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B", "openbmb/MiniCPM5-1B", "openbmb/MiniCPM5-1B-SFT",
    "Qwen/Qwen2.5-0.5B-Instruct", "deepseek-ai/deepseek-llm-7b-chat",
    "allenai/Olmo-3.1-32B-Instruct", "microsoft/phi-4-reasoning",
    "allenai/Llama-3.1-Tulu-3-8B-DPO",
}


def mean_lp(arrays):
    """Mean per-token logprob over all beams x all positions."""
    tot = n = 0.0
    for row in arrays:
        for x in row:
            tot += x
            n += 1
    return (tot / n) if n else None


def mean_at(arrays, i):
    """Mean logprob at ONE position across beams. For the position profile."""
    tot = n = 0.0
    for row in arrays:
        if i < len(row):
            tot += row[i]
            n += 1
    return (tot / n) if n else float("nan")


def collect():
    """(pair, prompt, word, arm, role) -> {'sb': [[...]], 'sa': [[...]]}."""
    cells = {}

    def take(pair, prompt, word, arm, role, sb, sa, beams=None):
        if not sb or not sa:
            return
        #: TOP BEAM ONLY for secondary #3. Beam search returns sequences sorted
        #: by sequence score, so beams[0] is the argmax path. Storing all 100
        #: would be 24M tokens for a comparison that is defined on the top path.
        top = None
        if beams:
            top = tuple((beams[0].get("tokens") or [])[:10])
        cells[(pair, prompt, word or "", arm, role)] = {"sb": sb, "sa": sa,
                                                        "top": top}

    from malign_logits.cache import get_cache
    st = get_cache()._stash("beam_fc")
    for k in st.keys():
        if not isinstance(k, dict) or k.get("type") != "fc_v1":
            continue
        v = st[k]
        #: **BOTH DESIGNS FROM THE STASH NOW.** The frozen run read the
        #: new-lineage pairs from raw shards because they were not yet
        #: ingested, which made the headline non-re-derivable by any other
        #: seat -- registrar's pen reached 28 pairs where this reached 33
        #: ([5017]). Merged 2026-08-08; the shard path below is retained only
        #: as a fallback and reports if it ever fires.
        if not isinstance(v, dict) or v.get("design") not in ("wave3-lexical",
                                                              "newlin-lexical"):
            continue
        take(k["pair"], k["prompt"], k.get("word"), k["arm"], k["role"],
             v.get("scored_by_base"), v.get("scored_by_aligned"), v.get("beams"))

    n_stash = len(cells)
    for f in glob.glob(NEWLIN_GLOB):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("design") != "newlin-lexical":
                continue
            w = r.get("word")
            w = "" if w in (None, "None") else w
            key = (r["pair"], r["prompt"], w, r["arm"], r["role"])
            if key in cells:      #: already in the stash -- the merge landed
                continue
            take(r["pair"], r["prompt"], w, r["arm"], r["role"],
                 r.get("scored_by_base"), r.get("scored_by_aligned"), r.get("beams"))
    if len(cells) != n_stash:
        print("  ** %d cell(s) came from RAW SHARDS, not the stash -- the merge "
              "is incomplete and this result is not re-derivable"
              % (len(cells) - n_stash))
    else:
        print("  all cells sourced from the beam_fc stash (re-derivable)")
    return cells


def sites(cells):
    """Sites with ALL FOUR cells present. Half-present sites are COUNTED."""
    want = collections.defaultdict(dict)
    for (pair, prompt, word, arm, role), v in cells.items():
        if arm not in ("force_faller", "force_riser"):
            continue
        want[(pair, prompt)][(arm, role)] = v
    full, partial = [], 0
    need = [("force_faller", "base"), ("force_faller", "aligned"),
            ("force_riser", "base"), ("force_riser", "aligned")]
    for key, d in want.items():
        if all(n in d for n in need):
            full.append((key, d))
        else:
            partial += 1
    return full, partial


def terms(d, arm):
    """The four scoring terms for one arm at one site."""
    b, a = d[(arm, "base")], d[(arm, "aligned")]
    return {"AA": mean_lp(a["sa"]), "AB": mean_lp(a["sb"]),
            "BA": mean_lp(b["sa"]), "BB": mean_lp(b["sb"])}


def boot_ci(vals, seed=SEED, n=NBOOT):
    """Percentile bootstrap CI on the median. Deterministic: seeded."""
    import random
    rnd = random.Random(seed)
    k = len(vals)
    if k < 2:
        return (float("nan"), float("nan"))
    meds = []
    for _ in range(n):
        meds.append(statistics.median(vals[rnd.randrange(k)] for _ in range(k)))
    meds.sort()
    return meds[int(0.025 * n)], meds[int(0.975 * n)]


def wilcoxon(vals):
    """Two-sided Wilcoxon signed-rank p. Ties dropped and COUNTED by caller."""
    nz = [v for v in vals if v != 0]
    n = len(nz)
    if n < 6:
        return float("nan"), n
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        r = (i + j) / 2.0 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = r
        i = j + 1
    wp = sum(ranks[i] for i in range(n) if nz[i] > 0)
    wm = sum(ranks[i] for i in range(n) if nz[i] < 0)
    w = min(wp, wm)
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w - mu + 0.5) / sd
    p = 2 * 0.5 * math.erfc(abs(z) / math.sqrt(2))
    return min(1.0, p), n


def per_pair(full, arm_fn, clean_only=False):
    """pair -> median over its sites of arm_fn(site_dict)."""
    by = collections.defaultdict(list)
    for (pair, _prompt), d in full:
        if clean_only:
            b, a = pair.split(">", 1)
            if b in SAMPLERS or a in SAMPLERS:
                continue
        v = arm_fn(d)
        if v is not None and not math.isnan(v):
            by[pair].append(v)
    return {p: statistics.median(vs) for p, vs in by.items() if vs}


def report(name, vals, note=""):
    if not vals:
        print("%-34s NO DATA" % name)
        return None
    xs = list(vals.values())
    med = statistics.median(xs)
    p, nz = wilcoxon(xs)
    lo, hi = boot_ci(xs)
    star = "CI excl 0" if (lo > 0 or hi < 0) else "CI incl 0"
    print("%-34s median %+9.5f  p %6.4f  n %2d (nz %2d)  CI [%+.5f,%+.5f] %s %s"
          % (name, med, p, len(xs), nz, lo, hi, star, note))
    return {"median": med, "p": p, "n": len(xs), "nz": nz, "ci": [lo, hi]}


def selftest():
    ok = []
    def case(n, c):
        ok.append((n, bool(c)))
    case("mean_lp averages over beams AND positions",
         abs(mean_lp([[-1.0, -3.0], [-2.0, -2.0]]) - (-2.0)) < 1e-12)
    case("mean_lp of empty is None", mean_lp([]) is None)
    #: a site missing one of four cells must NOT enter
    d = {("force_faller", "base"): 1, ("force_faller", "aligned"): 1,
         ("force_riser", "base"): 1}
    cells = {}
    for (arm, role) in d:
        cells[("P", "pr", "w", arm, role)] = {"sb": [[-1.0]], "sa": [[-1.0]]}
    full, partial = sites(cells)
    case("three-of-four cells is EXCLUDED and counted",
         len(full) == 0 and partial == 1)
    cells[("P", "pr", "w", "force_riser", "aligned")] = {"sb": [[-1.0]], "sa": [[-1.0]]}
    full, partial = sites(cells)
    case("four-of-four cells is INCLUDED", len(full) == 1 and partial == 0)
    #: delta must be zero when both arms are identical
    dd = full[0][1]
    dl = ((terms(dd, "force_faller")["AA"] - terms(dd, "force_faller")["BA"])
          - (terms(dd, "force_riser")["AA"] - terms(dd, "force_riser")["BA"]))
    case("identical arms give delta == 0", abs(dl) < 1e-12)
    case("bootstrap is deterministic under the frozen seed",
         boot_ci([1.0, 2.0, 3.0, 4.0, 5.0]) == boot_ci([1.0, 2.0, 3.0, 4.0, 5.0]))
    bad = [n for n, r in ok if not r]
    for n in bad:
        print("  [FAIL] %s" % n)
    print("channel3 self-test: %d of %d" % (len(ok) - len(bad), len(ok)))
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not selftest():
        sys.exit("self-test failed; refusing to run the frozen analysis")

    print("\nloading corpus ...")
    cells = collect()
    full, partial = sites(cells)
    pairs = {p for (p, _), _ in full}
    print("cells %d | sites with all four cells %d | half-present EXCLUDED %d | pairs %d"
          % (len(cells), len(full), partial, len(pairs)))

    D = lambda d, arm: (lambda t: t["AA"] - t["BA"])(terms(d, arm))
    Dp = lambda d, arm: (lambda t: t["BB"] - t["AB"])(terms(d, arm))

    print("\n=== POSITIVE CONTROL (runs FIRST; BOUNDS the primary, does not void it) ===")
    und = {}
    for (pair, prompt), d in full:
        u = cells.get((pair, prompt, "", "undisturbed", "base")), \
            cells.get((pair, prompt, "", "undisturbed", "aligned"))
        if all(u):
            dd = {("undisturbed", "base"): u[0], ("undisturbed", "aligned"): u[1]}
            und.setdefault(pair, []).append(D(dd, "undisturbed"))
    forced = per_pair(full, lambda d: (D(d, "force_faller") + D(d, "force_riser")) / 2)
    if und:
        undm = {p: statistics.median(v) for p, v in und.items()}
        both = sorted(set(forced) & set(undm))
        report("D(forced) - D(undisturbed)", {p: forced[p] - undm[p] for p in both})
    else:
        print("  undisturbed cells not matched at these sites — control NOT RUN")

    print("\n=== PRIMARY (frozen) ===")
    prim = per_pair(full, lambda d: D(d, "force_faller") - D(d, "force_riser"))
    r_all = report("delta  ALL", prim)
    prim_c = per_pair(full, lambda d: D(d, "force_faller") - D(d, "force_riser"),
                      clean_only=True)
    r_cln = report("delta  CLEAN (arbiter)", prim_c)

    print("\n=== REQUIRED CO-REPORT: four terms ===")
    for nm, fn in (("A|A aligned text, aligned scorer", lambda d: terms(d, "force_faller")["AA"] - terms(d, "force_riser")["AA"]),
                   ("A|B aligned text, base scorer", lambda d: terms(d, "force_faller")["AB"] - terms(d, "force_riser")["AB"]),
                   ("B|A base text, aligned scorer", lambda d: terms(d, "force_faller")["BA"] - terms(d, "force_riser")["BA"]),
                   ("B|B base text, base scorer", lambda d: terms(d, "force_faller")["BB"] - terms(d, "force_riser")["BB"])):
        report(nm, per_pair(full, fn))

    print("\n=== SECONDARY: mirror (one contrast among the four, not reported alone) ===")
    report("delta'", per_pair(full, lambda d: Dp(d, "force_faller") - Dp(d, "force_riser")))

    # ── DECLARED SECONDARIES (spec section 5). Declared, therefore NOT exploratory. ──
    print("\n=== SECONDARY 1: position profile ===")
    print("  one-shot cost spikes at +1 and decays; ongoing defense persists")
    for i in range(10):
        f = (lambda i: lambda d: (mean_at(d[("force_faller", "aligned")]["sa"], i)
                                  - mean_at(d[("force_faller", "base")]["sa"], i))
                                 - (mean_at(d[("force_riser", "aligned")]["sa"], i)
                                    - mean_at(d[("force_riser", "base")]["sa"], i)))(i)
        report("  position +%d" % (i + 1), per_pair(full, f))

    print("\n=== SECONDARY 2: twin moderator (MARKED vs UNMARKED) ===")
    member = {}
    try:
        import csv
        for r in csv.DictReader(open(os.path.join(ROOT, "data/beam_sample_105.csv"))):
            member[r["prompt"]] = r["member"]
    except Exception as e:
        print("  beam_sample_105.csv unreadable (%s) — NOT RUN" % type(e).__name__)
    if member:
        for want in ("MARKED", "UNMARKED"):
            sub = [(k, d) for k, d in full if member.get(k[1]) == want]
            print("  %-10s sites %d" % (want, len(sub)))
            report("  delta %s" % want,
                   per_pair(sub, lambda d: D(d, "force_faller") - D(d, "force_riser")))
        unk = sum(1 for k, _ in full if k[1] not in member)
        print("  sites with no stem metadata (new-lineage prompts share the "
              "sample, so 0 expected): %d" % unk)

    print("\n=== SECONDARY 3: own-beam divergence (BEHAVIOURAL) ===")
    print("  licenses which sentence section 7 permits: flat -> field with extent;")
    print("  moves -> charged signifier")
    def disagree(d, arm):
        b, a = d[(arm, "base")].get("top"), d[(arm, "aligned")].get("top")
        if not b or not a:
            return None
        n = min(len(b), len(a))
        if not n:
            return None
        return sum(1 for i in range(n) if b[i] != a[i]) / float(n)
    def ownbeam(d):
        x, y = disagree(d, "force_faller"), disagree(d, "force_riser")
        return None if (x is None or y is None) else x - y
    r_own = report("  own-beam disagreement delta", per_pair(full, ownbeam))
    lv = per_pair(full, lambda d: disagree(d, "force_faller"))
    if lv:
        print("  (level, forced-faller: median %.3f of 10 positions differ)"
              % statistics.median(lv.values()))

    print("\n=== SECONDARY 4: per-family, DESCRIPTIVE ONLY, no test ===")
    byfam = collections.defaultdict(list)
    for (pair, _p), d in full:
        fam = pair.split(">")[0].split("/")[0]
        v = D(d, "force_faller") - D(d, "force_riser")
        if v is not None and not math.isnan(v):
            byfam[fam].append(v)
    for fam, vs in sorted(byfam.items(), key=lambda kv: statistics.median(kv[1])):
        print("  %-24s median %+8.5f  sites %5d" % (fam, statistics.median(vs), len(vs)))

    if r_all:
        xs = list(prim.values())
        sd = statistics.pstdev(xs) if len(xs) > 1 else float("nan")
        mde = 2.8 * sd / math.sqrt(len(xs)) if len(xs) > 1 else float("nan")
        print("\nMDE at 80%% power, two-sided a=0.05, observed SD %.5f: %.5f nats/token"
              % (sd, mde))
        print("  = %.2fx the channel-1 effect (+0.0144) it would need to constrain"
              % (mde / 0.0144))
    print("\nEXPLORATORY analyses may follow and are labelled as such (spec section 10).")


if __name__ == "__main__":
    main()
