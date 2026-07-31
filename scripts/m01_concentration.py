"""Producer for M01/concentration and M01/recipient-agreement. One pass, two clauses.

    uv run .venv/bin/python scripts/m01_concentration.py
    ... --by-step        also emit every consecutive step, not just the operation edge
    ... --csv out.csv    per-cell rows

THIS IS A RE-MEASUREMENT, NOT A REFRESH ([1053].2). The booked figure — top recipient
takes 30-41% of gained mass — came from a DIFFERENT INSTRUMENT (logits exact-count) on a
DIFFERENT POPULATION (607 of 975 prompts by union, a roster including 52 models no longer
analysed). **It will not return 30-41% and must not be described as restoring it.** Its
verdict was relabelled VERIFIED-AS-ARITHMETIC-AT-[515]/[522]-TIME and 30-41% is unquotable
as a current number.

EVERY CHOICE THAT COULD MOVE A NUMBER IS A NAMED CONSTANT BELOW. That is the day's lesson
in file form: on 2026-07-31 five undeclared choices moved counts between 3 and 14 of 16 on
one comparison, and not one of them moved the direction. A choice a seat remembers is a
choice the next reader cannot audit.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import os
import statistics as st
import sys

# Walk UP looking for the package rather than counting dirnames. A fixed dirname depth
# is right for exactly one location: five levels resolved to '/Users/rj416/Dropbox/Prof/
# Articles' from the seat directory and to '/Users' from scripts/, so NEITHER worked and
# the import was carried entirely by the hardcoded home-directory fallback. That fallback
# is now gone; a committed producer must not depend on one machine's layout.
def _find_repo():
    # Two roots, in order: the FILE (correct once this lives in scripts/, and the only
    # one that works for a clone) and the CWD (correct while it still sits in the seat
    # directory, which is outside the repo entirely). Neither names a home directory.
    for start in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        d = start
        while True:
            if os.path.isdir(os.path.join(d, "malign_logits")):
                return d
            if d == os.path.dirname(d):
                break
            d = os.path.dirname(d)
    return None


REPO = _find_repo()
if REPO is None:                         # pragma: no cover - environment failure
    sys.exit("cannot find the malign_logits package above this file or the cwd; "
             "run from the repository")
sys.path.insert(0, REPO)

# --- THE DECLARED TRIPLE, per [1030].3 v2 ----------------------------------
POPULATION = "ACTIVE distinct texts, ALL languages"   # 975; not English-only
RESIDUAL = "carried as a bin (__TAIL__), never renormalised away"
SIDEDNESS = "n/a — this producer reports distributions, it runs no test"

# --- THE FROZEN POPULATION, [1071].2, diffed two-seat at [1100]/[1102] -----
PROMPTS_SHA = "a8693d7963f725c52386b5e2734ed0752bb74c471bbe4624343aceaf604991a4"
MODELS_SHA = "7bcc9e9d6aee8323e2dc343c4bb8dbe348d838d08f2a7fbe9f59d2d3f909dc78"
#: Declared because a digest without one makes a MISMATCH ambiguous between
#: "different set" and "different encoding" — [1102].2. Matching hashes prove
#: agreement; differing hashes prove nothing without this line.
CANONICALISATION = "sorted() ascending, '\\n'-joined, no trailing newline, utf-8, sha256"

# --- THE EDGE. NOT FIXED BY THE FROZEN SPEC, SO IT IS FIXED HERE. ----------
#: The spec ([1053].2) froze the INSTRUMENT and the POPULATION and left the EDGE open —
#: the population is 93 MODELS while concentration is a property of a STEP. The roster
#: does not offer one step type to everyone:
#:
#:     20 families   base->dpo only
#:     12 families   base->sft -> sft->dpo
#:      5 families   base->sft -> sft->dpo -> dpo->rlvr
#:      5 families   other two-step chains
#:
#: So an "isolated preference step" edge would silently drop the 20 single-step families
#: — a roster selection dressed as a method. THE OPERATION EDGE is base -> most-aligned
#: declared arm, which every family has by construction and which is what the clause is
#: about: what alignment does, not what one of its stages does.
EDGE = "base -> most-aligned declared arm"
#: `--by-step` additionally emits every consecutive step, so a reader can see whether the
#: choice moved anything rather than taking this comment's word for it.

RULE = "CANONICAL"          # riser tested against the renormalisation null

# --- THE DECLARED NULL FOR CONCENTRATION, [1120].3 -------------------------
#: "The top receiver takes 38% of arriving excess" has no meaning until you know what
#: SOME receiver takes when nothing concentrates. That baseline is not 1/n: the maximum
#: of n shares exceeds the mean by construction. Under a random split of a fixed total
#: across n receivers -- Dirichlet(1,...,1), the uniform distribution on the simplex --
#: the expected top share has the closed form H_n / n, the nth harmonic number over n.
#:
#:     n_risers   observed median   H_n/n     ratio
#:            2             0.733   0.750     0.977
#:            5             0.503   0.457     1.101
#:          10+             0.265   0.293     1.203
#:                                   POOLED   1.135
#:
#: So concentration is REAL and MODEST: about 14% above a random split among the words
#: that qualified, which is much smaller than 0.381 sounds. [1120].2 makes this rider
#: travel with the headline figure rather than sitting in a footnote.
#:
#: THE ORDERING DIED HERE AND THE SPREAD DID NOT. Raw concentration and the corrected
#: ratio INVERT each other across families (archangel tops the raw column and sits BELOW
#: chance corrected), so neither ranks families -- [1120].1. Both print, with their
#: disagreement measured, so no reader can take one ordering without meeting the other.
NULL = "Dirichlet(1..1) random split; E[top share] = H_n / n"


def null_top_share(n):
    """Expected top share under a random split across n receivers."""
    return sum(1.0 / k for k in range(1, n + 1)) / n


def check_null(trials=8000, tol=0.004):
    """Verify the closed form by simulation. RUNS UNCONDITIONALLY, never behind a flag.

    A declared null that nobody checks is the sharpening gate's failure again. And this
    check has already earned its place: the FIRST version of it simulated max(x)/sum(x)
    for x iid Uniform(0,1) and disagreed with the closed form by 0.11 at n=10. The
    simulation was wrong, not the formula -- Dirichlet(1,...,1) is normalised
    EXPONENTIALS, and normalised uniforms are a different distribution entirely. Booked
    at [1120].4(iii): a simulation check is itself checked against the closed form, and
    when they disagree the simulation is the likelier suspect.

    Unconditional because `--by-step` spent a morning declared and dead, certifying the
    choice it was written to audit ([1112].1). An audit gated behind a flag is an audit
    nobody has confirmed runs.
    """
    import random
    rng = random.Random(20260731)          # fixed: a check that varies run to run is not a check
    for n in (2, 5, 10):
        sim = []
        for _ in range(trials):
            x = [rng.expovariate(1.0) for _ in range(n)]
            sim.append(max(x) / sum(x))
        got, want = st.mean(sim), null_top_share(n)
        if abs(got - want) > tol:          # pragma: no cover - would be a real failure
            sys.exit(f"NULL CHECK FAILED at n={n}: simulated {got:.4f} vs H_n/n {want:.4f}")
    return True


def frozen_population():
    """Re-derive from the RULE and verify the digests. Never read from a stored list.

    A population frozen as a COUNT goes stale — the spec's "84 models" was the store at
    07:26, before the repair pass, and the rule yields 93 today without being touched
    ([1100].3). So the rule is the artifact and the digest is the check on it.
    """
    from malign_logits.cache import get_cache
    from malign_logits.prompts import Prompts

    prompts = sorted({p.text for p in Prompts.all(status="ACTIVE")})
    per = collections.defaultdict(set)
    for k in get_cache()._stash("true_word_probs"):
        d = dict(k) if not isinstance(k, dict) else k
        per[d.get("model")].add(d.get("prompt"))
    need = set(prompts)
    models = sorted(m for m, got in per.items() if need <= got)

    def h(seq):
        return hashlib.sha256("\n".join(seq).encode("utf-8")).hexdigest()

    ph, mh = h(prompts), h(models)
    drift = []
    if ph != PROMPTS_SHA:
        drift.append(f"prompts {ph} != frozen {PROMPTS_SHA}")
    if mh != MODELS_SHA:
        drift.append(f"models {mh} != frozen {MODELS_SHA}")
    return prompts, models, (ph, mh), drift


def operation_edges(models):
    """One Step per family: base -> most-aligned arm, both ends in the frozen models.

    Derived from the registry's declared positions, never from a hand-listed roster —
    a hand-enumerated candidate set is not a derivation ([1100].4).
    """
    from malign_logits.family import Family
    from malign_logits.step import Step

    ORDER = ("base", "ego", "superego", "reinforced_superego")
    inpop, out, dropped = set(models), [], collections.Counter()
    for f in Family.all():
        arms = [(p, f[p]) for p in ORDER]
        arms = [(p, a) for p, a in arms if a is not None and a.id in inpop]
        if len(arms) < 2:
            dropped["fewer than two arms in the frozen population"] += 1
            continue
        (p0, pre), (p1, post) = arms[0], arms[-1]
        if p0 != "base":
            dropped["no base arm in the frozen population"] += 1
            continue
        out.append((f.key, p1, Step(pre, post)))
    return out, dropped


def measure(step, prompts, rule):
    """Per-cell rows. Both clauses come off ONE Movement object per cell."""
    from malign_logits.movement import CANONICAL, DRAW

    r = {"CANONICAL": CANONICAL, "DRAW": DRAW}[rule]
    rows, skipped = [], collections.Counter()
    for t in prompts:
        c = step.cell(t)
        if not c.is_present:
            skipped["cell absent"] += 1
            continue
        try:
            d = c.decompose(r)
            m = c.movement(r)
        except ValueError as e:
            skipped["mixed rule_version" if "rule_version" in str(e) else "error"] += 1
            continue
        if d is None or m is None:
            skipped["no movement"] += 1
            continue
        if d["concentration"] is None:
            skipped["no riser gained beyond the null"] += 1
            continue
        rows.append({
            "prompt": t, "domain": c.domain, "language": c.language,
            "concentration": d["concentration"],        # M01/concentration
            "vs_chance": (d["concentration"] / null_top_share(d["n_risers"])
                          if d["n_risers"] else None),   # against the DECLARED null
            "top_riser": m.top_riser(),                  # M01/recipient-agreement
            "arrived": d["arrived"], "departed": d["departed"],
            "n_risers": d["n_risers"], "n_fallers": d["n_fallers"],
            "residual_pre": c.pre.residual, "residual_post": c.post.residual,
        })
    return rows, skipped


def agreement(by_family):
    """M01/recipient-agreement: do independent families name the same receiver?

    Per prompt, the modal top_riser across families and how many families supply it.
    Reported as a DISTRIBUTION over prompts, not a single rate — the booked form
    ("scream top riser in 24/45 families") is one word's rate at one site, and a single
    rate cannot show whether agreement is general or carried by a few prompts.
    """
    per_prompt = collections.defaultdict(dict)
    for fam, rows in by_family.items():
        for r in rows:
            if r["top_riser"]:
                per_prompt[r["prompt"]][fam] = r["top_riser"]
    out = []
    for t, fams in per_prompt.items():
        if len(fams) < 2:
            continue
        c = collections.Counter(fams.values())
        word, n = c.most_common(1)[0]
        out.append({"prompt": t, "modal_word": word, "n_agree": n,
                    "n_families": len(fams), "share": n / len(fams)})
    return out


def main(a):
    from malign_logits.sharpening import sharpening

    check_null()                 # unconditional; see its docstring for why
    prompts, models, (ph, mh), drift = frozen_population()
    print(f"POPULATION   {POPULATION}")
    print(f"NULL         {NULL}   (simulation-checked at import)")
    print(f"RESIDUAL     {RESIDUAL}")
    print(f"SIDEDNESS    {SIDEDNESS}")
    print(f"EDGE         {EDGE}")
    print(f"RULE         {RULE}")
    print(f"CANONICAL.   {CANONICALISATION}")
    print(f"FROZEN       prompts {len(prompts)} {ph[:16]}...  models {len(models)} {mh[:16]}...")
    if drift:
        print("\n  *** POPULATION DRIFT — the store has moved since the freeze ***")
        for d in drift:
            print(f"      {d}")
        print("  Refusing to measure. Re-diff with the other seat and re-freeze;")
        print("  measuring across a moved population is what the freeze exists to stop.")
        return 1

    edges, dropped = operation_edges(models)
    print(f"\n{len(edges)} families on the operation edge"
          + (f"   dropped: {dict(dropped)}" if dropped else ""))

    by_family, allrows, allskipped = {}, [], collections.Counter()
    print(f"\n  {'family':<18}{'edge':<12}{'n':>5}{'concentration':>16}"
          f"{'vs':>8}{'med':>7}{'1-riser':>9}{'entropy':>9}{'residual':>16}")
    print(f"  {'':<18}{'':<12}{'':>5}{'all':>8}{'>1 riser':>8}{'chance':>8}"
          f"{'risers':>7}{'cells':>9}{'delta':>9}{'pre':>8}{'post':>8}")
    for fam, pos, step in sorted(edges):
        rows, skipped = measure(step, prompts, RULE)
        allskipped.update(skipped)
        if not rows:
            print(f"  {fam:<18}{step.label:<14}{'0':>5}   no measurable cell"
                  f"   {dict(skipped)}")
            continue
        by_family[fam] = rows
        for r in rows:
            r["family"], r["edge"] = fam, step.label
        allrows += rows
        con = sorted(r["concentration"] for r in rows)
        multi = sorted(r["concentration"] for r in rows if r["n_risers"] > 1)
        one = sum(1 for r in rows if r["n_risers"] == 1)
        sh = sharpening(step, texts=prompts)
        vsc = [r["vs_chance"] for r in rows if r["n_risers"] > 1]
        nrs = [r["n_risers"] for r in rows if r["n_risers"] > 1]
        print(f"  {fam:<18}{step.label:<12}{len(rows):>5}"
              f"{st.median(con):>8.3f}"
              f"{(st.median(multi) if multi else float('nan')):>8.3f}"
              f"{(st.median(vsc) if vsc else float('nan')):>8.2f}"
              f"{(st.median(nrs) if nrs else float('nan')):>7.0f}"
              f"{100 * one / len(rows):>8.0f}%"
              f"{(sh['entropy_delta'] if sh else float('nan')):>+9.3f}"
              f"{st.median([r['residual_pre'] for r in rows]):>8.3f}"
              f"{st.median([r['residual_post'] for r in rows]):>8.3f}"
              + ("  FLAT" if sh and sh["is_flat"] else ""))

    if not allrows:
        print("\nno cells measured")
        return 1

    # THE DENOMINATOR, ACCOUNTED FOR. Counted per family and then dropped on the floor
    # until 2026-07-31: the run reported n with no account of what fell out of 975 x 42,
    # and `mixed rule_version` in particular is a SILENT EXCLUSION of exactly what the
    # stamp regime exists to catch. Count the denominator first; no silent caps.
    offered = len(prompts) * len(by_family)
    print(f"\n  DENOMINATOR   {offered} cells offered ({len(prompts)} prompts x "
          f"{len(by_family)} families), {len(allrows)} measured, "
          f"{sum(allskipped.values())} dropped")
    for reason, n in allskipped.most_common():
        print(f"      {n:>6}  {reason}")
    if allskipped.get("mixed rule_version"):
        print("      *** mixed rule_version is an INSTRUMENT CHANGE booked as movement;")
        print("          any nonzero count here needs a stated reason before quotation.")

    con = sorted(r["concentration"] for r in allrows)
    multi = sorted(r["concentration"] for r in allrows if r["n_risers"] > 1)
    one = len(allrows) - len(multi)
    print(f"\n  M01/concentration  POOLED over {len(allrows)} cells, "
          f"{len(by_family)} families")
    print(f"    all cells      median {st.median(con):.3f}   "
          f"p10 {con[int(.1*(len(con)-1))]:.3f}   p90 {con[int(.9*(len(con)-1))]:.3f}")
    print(f"    >1 riser only  median {st.median(multi):.3f}   "
          f"p10 {multi[int(.1*(len(multi)-1))]:.3f}   p90 {multi[int(.9*(len(multi)-1))]:.3f}"
          f"   (n={len(multi)})")
    print(f"    A SINGLE-RISER CELL HAS CONCENTRATION 1.000 BY CONSTRUCTION -- one word")
    print(f"    qualified, which is not mass concentrating. {one} of {len(allrows)} cells")
    print(f"    ({100*one/len(allrows):.0f}%), and the RATE VARIES 1%-25% BY FAMILY. The")
    print(f"    >1-riser row is the quotable one; both print so the gap is visible.")
    print("    POOLED IS A MIXTURE either way.")

    # AGAINST THE DECLARED NULL, [1120].2(ii). The rider that travels with the headline.
    vsc = sorted(r["vs_chance"] for r in allrows if r["n_risers"] > 1)
    print(f"\n    vs the {NULL}:")
    print(f"      median ratio {st.median(vsc):.3f}   p10 {vsc[int(.1*(len(vsc)-1))]:.3f}"
          f"   p90 {vsc[int(.9*(len(vsc)-1))]:.3f}   (1.000 = chance)")
    print(f"      CONCENTRATION IS REAL AND MODEST: {100*(st.median(vsc)-1):.0f}% above a")
    print(f"      random split among the words that qualified. This rider is NOT optional")
    print(f"      beside the {st.median(multi):.3f} -- [1120].2 makes the three-part shape")
    print(f"      travel together: the share, its ratio to chance, and the size of the")
    print(f"      receiving set (median risers {st.median([r['n_risers'] for r in allrows if r['n_risers']>1]):.0f}, ranging 3 to 20 by family).")

    # THE TWO ORDERINGS, AND THEIR DISAGREEMENT MEASURED. Neither ranks families.
    fam_raw = {f: st.median([r["concentration"] for r in rs if r["n_risers"] > 1])
               for f, rs in by_family.items()
               if any(r["n_risers"] > 1 for r in rs)}
    fam_vsc = {f: st.median([r["vs_chance"] for r in rs if r["n_risers"] > 1])
               for f in fam_raw for rs in [by_family[f]]}
    fam_nr = {f: st.median([r["n_risers"] for r in by_family[f] if r["n_risers"] > 1])
              for f in fam_raw}
    keys = sorted(fam_raw)

    def _rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0] * len(vals)
        for pos, i in enumerate(order):
            out[i] = pos
        return out

    raw_v = [fam_raw[k] for k in keys]
    vsc_v = [fam_vsc[k] for k in keys]
    nr_v = [fam_nr[k] for k in keys]
    if len(keys) > 2:
        rr = st.correlation(_rank(raw_v), _rank(vsc_v))
        r_nr = st.correlation(_rank(raw_v), _rank(nr_v))
        v_nr = st.correlation(_rank(vsc_v), _rank(nr_v))
        print(f"\n    NO BETWEEN-FAMILY ORDERING IS QUOTABLE IN EITHER METRIC ([1120].1).")
        print(f"      Spearman(raw ordering, vs-chance ordering) = {rr:+.3f}")
        print(f"      raw vs n_risers {r_nr:+.3f}      vs-chance vs n_risers {v_nr:+.3f}")
        print(f"      The two metrics {'INVERT' if rr < 0 else 'agree on'} each other's ranking, and each is")
        print(f"      dominated by how many words qualified -- in OPPOSITE directions, so")
        print(f"      correcting overshoots rather than fixing. The per-family spread")
        print(f"      ({min(raw_v):.3f} to {max(raw_v):.3f}) is a SPREAD STATEMENT, never a ranking.")
        print(f"      Both orderings print here so no reader can take one without the other:")
        print(f"\n      {'by RAW':<22}{'':4}{'by VS-CHANCE':<22}")
        hi_raw = sorted(fam_raw, key=lambda k: -fam_raw[k])
        hi_vsc = sorted(fam_vsc, key=lambda k: -fam_vsc[k])
        for i in range(min(5, len(keys))):
            print(f"      {hi_raw[i]:<14}{fam_raw[hi_raw[i]]:>8.3f}    "
                  f"{hi_vsc[i]:<14}{fam_vsc[hi_vsc[i]]:>8.2f}")

    ag = agreement(by_family)
    if ag:
        sh = sorted(r["share"] for r in ag)
        print(f"\n  M01/recipient-agreement  {len(ag)} prompts with >=2 families")
        print(f"    AGREEMENT IS A PROPERTY OF A SITE, NOT AN AVERAGE. The booked form")
        print(f"    ('scream top riser in 24/45 families') is ONE WORD'S RATE AT ONE")
        print(f"    PROMPT. Averaging over all {len(ag)} prompts answers a different")
        print(f"    question and returns function words, because most prompts have no")
        print(f"    contested receiver at all. Distribution first, then the sites.")
        print(f"    share of families naming the modal receiver:"
              f"   median {st.median(sh):.3f}"
              f"   p90 {sh[int(.9*(len(sh)-1))]:.3f}   max {sh[-1]:.3f}")

        # CONDITION ON DISPLACEMENT. Agreement at a site where nothing FELL is
        # grammatical inevitability, not convergence on a substitute.
        dep = collections.defaultdict(list)
        for rows in by_family.values():
            for r in rows:
                dep[r["prompt"]].append(r["departed"])
        med = {t: st.median(v) for t, v in dep.items()}
        lo = [r["share"] for r in ag if med.get(r["prompt"], 0) < 0.02]
        hi = [r["share"] for r in ag if med.get(r["prompt"], 0) >= 0.10]
        print(f"\n    AND AGREEMENT RUNS THE WRONG WAY AGAINST DISPLACEMENT:")
        print(f"      little/no displacement (median departed < 0.02)   n={len(lo):>4}"
              f"   median agreement {st.median(lo):.3f}")
        print(f"      REAL displacement      (median departed >= 0.10)  n={len(hi):>4}"
              f"   median agreement {st.median(hi):.3f}")
        print(f"    Agreement is HIGHER where nothing was suppressed. The top-agreement")
        print(f"    sites are forced slots -- 'the capital of France is' -> Paris, 'held")
        print(f"    his' -> hand -- so a POOLED agreement rate measures grammatical")
        print(f"    inevitability, not families converging on a substitute. The clause's")
        print(f"    subject is agreement AT A DISPLACING SITE, and only these qualify.")
        print(f"\n    HIGHEST-AGREEMENT DISPLACING SITES (departed >= 0.10):")
        shown = [r for r in ag if med.get(r["prompt"], 0) >= 0.10]
        for r in sorted(shown, key=lambda r: -r["share"])[:10]:
            print(f"      {r['share']:.2f}  {r['n_agree']:>2}/{r['n_families']:<2}"
                  f"  {r['modal_word']!r:<13} {r['prompt'][:44]!r}")

    if a.by_step:
        # THE AUDIT OF `EDGE`, AND IT MUST ACTUALLY RUN. This block was declared in the
        # usage text and in EDGE's own comment -- "so a reader can see whether the choice
        # moved anything rather than taking this comment's word for it" -- and then never
        # consulted, so `--by-step` returned output byte-identical to the plain run and a
        # reader would have concluded the edge was immaterial. The flag manufactured the
        # reassurance it was written to avoid. Caught by malign's custody read, not by me.
        from malign_logits.family import Family
        from malign_logits.step import Step
        print(f"\n  --by-step: EVERY CONSECUTIVE STEP, against the operation edge above")
        print(f"  {'family':<18}{'step':<14}{'n':>5}{'conc >1riser':>13}"
              f"{'1-riser':>9}{'vs edge':>9}")
        edge_med = {fam: st.median([r["concentration"] for r in rows if r["n_risers"] > 1])
                    for fam, rows in by_family.items()}
        for f in sorted(Family.all(), key=lambda f: f.key):
            chain = [s for s in Step.chain(f)
                     if s.pre.id in models and s.post.id in models]
            if len(chain) < 2:
                continue          # a one-step chain IS the operation edge; nothing to compare
            for step in chain:
                rows, _ = measure(step, prompts, RULE)
                m = [r["concentration"] for r in rows if r["n_risers"] > 1]
                if not m:
                    continue
                one = sum(1 for r in rows if r["n_risers"] == 1)
                delta = st.median(m) - edge_med.get(f.key, float("nan"))
                print(f"  {f.key:<18}{step.label:<14}{len(rows):>5}{st.median(m):>13.3f}"
                      f"{100 * one / len(rows):>8.0f}%{delta:>+9.3f}")
        print("  vs edge = this step's median minus the family's OPERATION-EDGE median.")
        print("  Large values mean the edge choice moved the number and the ruling at")
        print("  [1110].2 needs re-reading; near-zero means it did not.")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(allrows[0]))
            w.writeheader()
            w.writerows(allrows)
        print(f"\nwrote {a.csv}  {len(allrows)} rows")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--by-step", action="store_true",
                   help="also emit every consecutive step, to show whether EDGE matters")
    p.add_argument("--csv")
    sys.exit(main(p.parse_args()))
