"""Plan B analysis: the arm effect on twp, with the speaker factorial joined on.

    uv run python b_analysis.py                # primary + all secondaries
    uv run python b_analysis.py --primary      # the arm contrast alone
    uv run python b_analysis.py --csv          # also write the tidy tables

Reads `results/b_pair_prompt.jsonl` (the producer, `b_twp_institutional.py`) and
joins the speaker-kernel design labels from `data/prompt_categorisation.json`.

WHY THE JOIN EXISTS AT ALL. The producer stored `stratum` -- which FILE a prompt
came from, `f21_institutional` or `m03_speaker_kernel` -- and not the design. A
comparison between those two strata is a comparison between two prompt SOURCES
and answers nothing F21 asked. The arm (indiv / inst) lives in `group_role`, and
without it the whole reason for taking the kernel whole is unreachable. Fixed
here rather than in the producer because the producer's output is correct as far
as it goes and re-running it would change nothing but the columns.

JOINED ON THE PROMPT TEXT, NEVER ON THE ID. Standing rule, RH: prompt ids are not
trustworthy across the campaign's stores; the text is the key. Verified rather
than assumed -- the kernel's 252 rows carry 252 distinct texts, so the join is
one-to-one, and the script REFUSES if any kernel row fails to join.

THE UNIT IS THE LINEAGE, n = 46, and the primary never reports a pooled number
without the per-lineage direction beside it. Pooled numbers died four times in
one day (CAMPAIGN.md); a magnitude carried by a nameable subset travels with its
owner named. Group sizes print before any grouped n is quoted.

JS IS IN BITS, and the producer measures it through `Cell.js()` rather than
computing its own. An earlier version read a hand-rolled `js` column in NATS,
and the same producer summed `twp_words` across SOURCES -- 708 cells of 13,340
carried a `mass_base` above 1.05, which is not a probability. Both are fixed at
the producer; this file reads `js_bits`.

WHAT THIS MEASURES, AND WHAT IT DOES NOT. `js_bits` is HOW FAR the next-word
distribution moved base->aligned. F21 coded WHAT the response said. These are
different quantities and a large move is not a procedural move: a model can
shift a great deal of mass and become no more deferential. So a reversal here
is a reversal of F21's DIRECTION ON A DIFFERENT INSTRUMENT, not a refutation of
its coded finding, and must be reported as such.

THE HEADROOM ALTERNATIVE IS COMPUTED, NOT WAVED AT. If the institutional arm's
base distribution is flatter -- more words above theta, less mass resolved --
then it has more room to move and a larger JS is partly an artifact of the
support. `--primary` prints n_words_base, mass_base and residual_share per arm
for exactly this reason, and the arm contrast is re-run on the resolved-mass
subset as a secondary.
"""
import argparse
import collections
import csv
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))

ROWS = os.path.join(CAMP, "results", "b_pair_prompt.jsonl")
CAT = os.path.join(ROOT, "data", "prompt_categorisation.json")
OUT_CELL = os.path.join(CAMP, "results", "b_arm_cells.csv")
OUT_LIN = os.path.join(CAMP, "results", "b_arm_by_lineage.csv")

KERNEL = "m03_speaker_kernel"
F21 = "f21_institutional"

#: the factorial, as it is spelled in `group_role`. ARM is the first token and
#: everything after the first underscore is the condition. Enumerated rather
#: than parsed further, so a new role added upstream is a REFUSAL and not a
#: silent recode into one of these buckets.
CONDITIONS = ("I_absent", "I_medial", "I_final", "I_final_ought",
              "we_absent", "we_medial", "we_final")
ARMS = ("indiv", "inst")


def load_design():
    """group_role and group_id per kernel prompt TEXT. Refuses on a duplicate."""
    cat = json.load(open(CAT))["prompts"]
    k = [r for r in cat if r.get("source") == "M03_SPEAKER_KERNEL"]
    texts = collections.Counter(r["prompt"] for r in k)
    dup = [t for t, n in texts.items() if n > 1]
    if dup:
        raise SystemExit("kernel texts are not unique; %d duplicated, e.g. %r"
                         % (len(dup), dup[0][:60]))
    design = {}
    for r in k:
        arm, _, cond = r["group_role"].partition("_")
        if arm not in ARMS or cond not in CONDITIONS:
            raise SystemExit("unenumerated group_role %r -- add it to CONDITIONS/"
                             "ARMS deliberately, do not let it fall through"
                             % r["group_role"])
        design[r["prompt"]] = (arm, cond, r["group_id"])
    return design


def load_rows():
    if not os.path.exists(ROWS):
        raise SystemExit("no producer output at %s -- run b_twp_institutional.py "
                         "--run first" % os.path.relpath(ROWS, ROOT))
    return [json.loads(l) for l in open(ROWS)]


def joined(rows, design):
    """Kernel rows with the design attached. REFUSES on any unjoined row."""
    kr = [r for r in rows if r["stratum"] == KERNEL]
    miss = [r for r in kr if r["prompt"] not in design]
    if miss:
        raise SystemExit("%d of %d kernel rows did not join on text; first: %r"
                         % (len(miss), len(kr), miss[0]["prompt"][:70]))
    out = []
    for r in kr:
        arm, cond, gid = design[r["prompt"]]
        d = dict(r)
        d["arm"], d["condition"], d["scenario"] = arm, cond, gid
        out.append(d)
    return out


def sign_test(vals):
    """Two-sided exact sign test on the sign of a paired difference."""
    v = [x for x in vals if x != 0.0]
    n, k = len(v), sum(1 for x in v if x > 0)
    if n == 0:
        return n, k, float("nan")
    tail = min(k, n - k)
    p = 2 * sum(math.comb(n, i) for i in range(0, tail + 1)) / 2 ** n
    return n, k, min(1.0, p)


def cells(J):
    """One row per (lineage, scenario, condition) with BOTH arms present.

    This is the paired unit: person, modal position, modal type and the
    scenario itself are identical across the two arms, so the only thing that
    differs is the arm. A cell missing either arm is dropped and COUNTED --
    a silent drop here would be a population change wearing an analysis hat.
    """
    idx = {}
    for r in J:
        idx[(r["lineage"], r["scenario"], r["condition"], r["arm"])] = r
    out, dropped = [], 0
    keys = sorted({k[:3] for k in idx})
    for lin, scen, cond in keys:
        a, b = idx.get((lin, scen, cond, "indiv")), idx.get((lin, scen, cond, "inst"))
        if a is None or b is None:
            dropped += 1
            continue
        out.append({
            "lineage": lin, "scenario": scen, "condition": cond,
            "js_indiv": a["js_bits"], "js_inst": b["js_bits"],
            "d_js": b["js_bits"] - a["js_bits"],
            "resid_indiv": a["residual_share"], "resid_inst": b["residual_share"],
            "nw_indiv": a["n_words_base"], "nw_inst": b["n_words_base"],
            "mass_indiv": a["mass_base"], "mass_inst": b["mass_base"],
            })
    return out, dropped


def h_sizes(label, groups):
    """CAMPAIGN.md: print the group sizes BEFORE quoting a grouped n.

    A grouping with all sizes 1 is not a grouping; a grouping whose sizes are
    3, 2, 1, 1... is not the n about to be quoted.
    """
    sz = collections.Counter(len(v) for v in groups.values())
    print("  %-28s %d groups; sizes %s"
          % (label, len(groups), ", ".join("%dx%d" % (n, s) for s, n in sorted(sz.items()))))


def primary(C):
    print("=" * 74)
    print("PRIMARY -- the arm effect, person / modal / scenario held fixed")
    print("=" * 74)
    print("\nunit words, stated because a count is a fact about the unit counted in:")
    print("  cell     = (lineage, scenario, condition), both arms present")
    print("  lineage  = the 46 lineage representatives; the unit of the test")
    print("  d_js     = JS(inst) - JS(indiv); NEGATIVE is F21's stated direction")
    print("             (alignment proceduralises the individual, not the institution)")

    by_lin = collections.defaultdict(list)
    for c in C:
        by_lin[c["lineage"]].append(c["d_js"])
    print()
    h_sizes("cells per lineage", by_lin)

    meds = {l: st.median(v) for l, v in by_lin.items()}
    n, k, p = sign_test(list(meds.values()))
    print("\n  cells                       %d" % len(C))
    print("  lineages                    %d" % len(meds))
    print("  median of lineage medians   %+.5f" % st.median(list(meds.values())))
    print("  lineages with d_js > 0      %d of %d   (institution moves MORE)" % (k, n))
    print("  exact sign test, two-sided  p = %.3g" % p)

    srt = sorted((v, l) for l, v in meds.items())
    print("\n  the lineages in F21's direction (individual moves more):")
    neg = [(v, l) for v, l in srt if v < 0]
    for v, l in neg:
        print("    %+.5f  %s" % (v, l))
    if not neg:
        print("    none")
    print("  the five furthest the other way:")
    for v, l in srt[-5:]:
        print("    %+.5f  %s" % (v, l))

    print("\n  BY CONDITION -- this is RH's `should` confound, measured.")
    print("  If the arm gap were the modal, it would vanish at `absent`.")
    by_c = collections.defaultdict(list)
    for c in C:
        by_c[c["condition"]].append(c["d_js"])
    for cond in CONDITIONS:
        v = by_c[cond]
        lin = collections.defaultdict(list)
        for c in C:
            if c["condition"] == cond:
                lin[c["lineage"]].append(c["d_js"])
        lm = [st.median(x) for x in lin.values()]
        _, kk, pp = sign_test(lm)
        print("    %-16s cells %4d  median %+.5f   lineages>0 %2d/%2d  p=%.2g"
              % (cond, len(v), st.median(v), kk, len(lm), pp))

    print("\n  THE HEADROOM ALTERNATIVE, computed rather than waved at.")
    print("  A flatter base distribution has more room to move, so a larger JS")
    print("  can be a fact about the support rather than about alignment.")
    for arm, a, b, c in (("indiv", "nw_indiv", "mass_indiv", "resid_indiv"),
                         ("inst", "nw_inst", "mass_inst", "resid_inst")):
        print("    %-6s median n_words_base %6.0f   mass_base %.4f   residual_share %.4f"
              % (arm, st.median([x[a] for x in C]), st.median([x[b] for x in C]),
                 st.median([x[c] for x in C])))
    dn = st.median([x["nw_inst"] - x["nw_indiv"] for x in C])
    dr = st.median([x["resid_inst"] - x["resid_indiv"] for x in C])
    print("    paired difference (inst - indiv): n_words %+.0f   residual_share %+.4f"
          % (dn, dr))
    print("    -> the arms are NOT matched on support. Read the secondary below")
    print("       before quoting the primary as an alignment effect.")


def secondary_matched(C):
    """The arm contrast on the cells where the two arms' supports are close.

    Not a repair and not a better estimate -- a check on whether the primary
    survives when the headroom difference is small. The band is stated as a
    parameter of the check and both branches are reported, because a band
    chosen after seeing which one works is chosen on the outcome.
    """
    print("\n" + "=" * 74)
    print("SECONDARY 1 -- does the arm effect survive matching on support?")
    print("=" * 74)
    for band in (0.02, 0.05, 0.10):
        sub = [c for c in C if abs(c["resid_inst"] - c["resid_indiv"]) <= band]
        if not sub:
            print("  band %.2f: no cells" % band)
            continue
        lin = collections.defaultdict(list)
        for c in sub:
            lin[c["lineage"]].append(c["d_js"])
        lm = [st.median(v) for v in lin.values()]
        n, k, p = sign_test(lm)
        print("  |d residual_share| <= %.2f   cells %5d  lineages %2d  "
              "median %+.5f  >0: %2d/%2d  p=%.2g"
              % (band, len(sub), len(lin), st.median(lm), k, n, p))


#: F21's OWN arm axis, carried in `subdomain` on the 24 `institutional_*`
#: prompts: twelve scenarios written from both ends. This is the contrast F21
#: was originally made on, and the M03 kernel was built to replicate it.
F21_INDIV = ("worker", "tenant", "patient", "citizen")
F21_INST = ("mgmt", "landlord", "doctor", "agency", "officer", "party")


def f21_labels():
    """Two INDEPENDENT label maps over F21's texts, deliberately not merged.

    `subdomain` marks the arm on the 24 `institutional_*` prompts; `pair_role`
    marks MARKED/UNMARKED on the 14 SETE prompts. ONE TEXT CARRIES BOTH -- "My
    boss takes credit for my work in every meeting. I should" is `e1_credit_M`
    and `institutional_labor_worker_2`, two prompt_ids for identical text. It
    enters both analyses, which is correct, and merging the maps would have
    forced a choice between two labels that answer different questions.
    """
    cat = json.load(open(CAT))["prompts"]
    f = [r for r in cat if r.get("finding") == "F21"
         and r.get("status") == "ACTIVE" and r.get("language") == "en"]
    arm, pole = {}, {}
    for r in f:
        sd = r.get("subdomain")
        if sd in F21_INDIV:
            arm[r["prompt"]] = ("indiv", r.get("domain"), r["prompt_id"])
        elif sd in F21_INST:
            arm[r["prompt"]] = ("inst", r.get("domain"), r["prompt_id"])
        if r.get("pair_role") in ("MARKED", "UNMARKED"):
            pole[r["prompt"]] = (r["pair_role"], r.get("pair_id"), r["prompt_id"])
    return f, arm, pole


def secondary_f21(rows):
    """F21's own arm contrast, on F21's own prompts.

    THIS REPLACES A WRONG SECONDARY. The previous version printed "NO
    individual arm exists in this stratum, so no arm contrast is computable on
    it" -- exactly backwards. F21's individual/institution axis is in
    `subdomain`, and the finding was made on it. I read `pair_role` (null on
    these 24 rows), concluded there was no pairing, and reported the absence of
    the column as the absence of the design.

    THE PAIRING IS AT THE LINEAGE, NOT AT THE TEXT. The twelve scenarios are an
    authored mirror -- "I need to lay off a team to cut costs" against "laid
    off my whole team with no severance" -- not minimal pairs; they differ in
    more than the arm. So each lineage contributes one median per side and the
    test is over the 46 lineages. Pairing by index would assume a mirror I can
    only verify by reading, and one of the five labour pairs does not hold.
    """
    print("\n" + "=" * 74)
    print("SECONDARY 2 -- F21's OWN arm contrast, on F21's own 24 prompts")
    print("=" * 74)
    f, arm, pole = f21_labels()
    fr = [r for r in rows if r["stratum"] == F21]
    miss = [r for r in fr if r["prompt"] not in {x["prompt"] for x in f}]
    if miss:
        raise SystemExit("%d F21 rows did not join on text (TSV escaping?); "
                         "first: %r" % (len(miss), miss[0]["prompt"][:70]))

    side = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in fr:
        a = arm.get(r["prompt"])
        if a:
            side[r["lineage"]][a[0]].append(r["js_bits"])
    texts = {t for t in arm}
    print("  texts carrying an arm label: %d  (indiv %d / inst %d)"
          % (len(texts), sum(1 for v in arm.values() if v[0] == "indiv"),
             sum(1 for v in arm.values() if v[0] == "inst")))
    doms = collections.Counter((v[1], v[0]) for v in arm.values())
    print("  by domain: %s" % ", ".join("%s %s:%d" % (d, a, n)
                                        for (d, a), n in sorted(doms.items())))

    d = [st.median(v["inst"]) - st.median(v["indiv"])
         for v in side.values() if v.get("inst") and v.get("indiv")]
    n, k, p = sign_test(d)
    print("\n  d = median JS(inst) - median JS(indiv), per lineage")
    print("  lineages          %d" % len(d))
    print("  median            %+.5f" % st.median(d))
    print("  lineages > 0      %d of %d   (institution moves MORE)" % (k, n))
    print("  sign test         p = %.3g" % p)
    print("\n  SAME DIRECTION AS THE KERNEL PRIMARY OR NOT, this is a SECOND")
    print("  population: 12 authored mirror scenarios, no modal or person")
    print("  control, and 24 texts against the kernel's 252.")

    #: the SETE poles, which are a different manipulation and are not pooled
    print("\n  ---- the 7 SETE MARKED/UNMARKED pairs, reported apart ----")
    ps = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in fr:
        q = pole.get(r["prompt"])
        if q:
            ps[r["lineage"]][q[0]].append(r["js_bits"])
    dm = [st.median(v["MARKED"]) - st.median(v["UNMARKED"])
          for v in ps.values() if v.get("MARKED") and v.get("UNMARKED")]
    n2, k2, p2 = sign_test(dm)
    print("  d = median JS(MARKED) - median JS(UNMARKED), per lineage")
    print("  lineages %d   median %+.5f   >0: %d/%d   p=%.3g"
          % (len(dm), st.median(dm), k2, n2, p2))
    print("  NOT ONE MANIPULATION. Five pairs swap a grievance for its absence")
    print("  (`refuses to fix` / `offered to fix`); e5_agency and e5_deposit")
    print("  swap the INSTITUTION for a PERSON (`the agency` / `my cousin`,")
    print("  `the company` / `my flatmate`), which is the arm contrast again")
    print("  wearing a pole label. Pooling the seven averages two designs.")


def secondary_movement(J):
    """Risers and fallers by arm. Counts only -- the fields file is separate."""
    print("\n" + "=" * 74)
    print("SECONDARY 3 -- movement counts by arm")
    print("=" * 74)
    print("  ASYMMETRY, preserved from movement.CANONICAL: risers are tested")
    print("  against the renormalisation null, fallers are a bare ratio rule.")
    print("  Fallers may never be described as 'beyond renormalisation'.")
    for arm in ARMS:
        v = [r for r in J if r["arm"] == arm]
        print("    %-6s cells %5d  median risers %4.1f  median fallers %4.1f  "
              "inflation %.4f" % (arm, len(v),
                                  st.median([x["n_risers"] for x in v]),
                                  st.median([x["n_fallers"] for x in v]),
                                  st.median([x["inflation"] for x in v])))
    ex = sum(1 for r in J if r.get("exact_null"))
    print("  exact_null true on %d of %d cells -- twp is truncated at theta, so"
          % (ex, len(J)))
    print("  the null is computed over a truncated support by construction.")


def _flat_fields(f):
    """Yield (source, field, count) over both schemas the module emits.

    Flat sources give `{counts: {...}, coverage: x}`. `norms` is a level
    deeper -- one sub-dictionary per norm (valence, arousal, ...) each with
    its own counts and its OWN coverage, because Warriner and Brysbaert know
    different words. Flattening them into one namespace without the prefix
    would silently merge `neutral` across five different norms.
    """
    for src, v in f.items():
        if src == "norms":
            for norm, w in v.items():
                for k, n in w.get("counts", {}).items():
                    yield "norms:" + norm, k, n
        else:
            for k, n in v.get("counts", {}).items():
                yield src, k, n


def _coverage(f):
    for src, v in f.items():
        if src == "norms":
            for norm, w in v.items():
                yield "norms:" + norm, w.get("coverage")
        else:
            yield src, v.get("coverage")


def secondary_fields(design, side_want="risers", top=12):
    """Semantic fields on the risers/fallers, by arm.

    THE COMPOSITION IS THE OUTCOME, NOT THE COUNT. A cell with 25 risers and a
    cell with 6 contribute equally once each is turned into a share, which is
    what makes the arms comparable when the institutional arm simply has more
    risers (Secondary 3: median 13 vs 9). Comparing raw counts here would
    re-measure the movement-count difference and call it a field difference.

    COVERAGE IS PRINTED PER SOURCE AND PER ARM, and it is not decoration. The
    General Inquirer is a 1960s resource with no entry for `raped`,
    `desecrated` or `stomped`; a caller comparing two texts on GI counts
    without looking at coverage is comparing how much of each text GI happens
    to know. If the two arms differ in coverage on a source, that source's
    field differences are partly a difference in what the lexicon knows.
    """
    path = os.path.join(CAMP, "results", "b_fields.jsonl")
    if not os.path.exists(path):
        print("\n(no fields file at %s)" % os.path.relpath(path, ROOT))
        return
    print("\n" + "=" * 74)
    print("SECONDARY 4 -- semantic fields on the %s, by arm" % side_want)
    print("=" * 74)

    #: share[(lineage, scenario, condition, arm)][(source, field)] = share
    share = collections.defaultdict(dict)
    cov = collections.defaultdict(lambda: collections.defaultdict(list))
    seen = 0
    for line in open(path):
        r = json.loads(line)
        if r["stratum"] != KERNEL or r["side"] != side_want:
            continue
        d = design.get(r["prompt"])
        if d is None:
            raise SystemExit("fields row did not join on text: %r"
                             % r["prompt"][:70])
        arm, cond, gid = d
        seen += 1
        tot = collections.Counter()
        for src, field, n in _flat_fields(r["fields"]):
            tot[src] += n
        cell = {}
        for src, field, n in _flat_fields(r["fields"]):
            if tot[src]:
                cell[(src, field)] = n / tot[src]
        share[(r["lineage"], gid, cond, arm)] = cell
        for src, c in _coverage(r["fields"]):
            if c is not None:
                cov[src][arm].append(c)

    print("  fields rows read (kernel, side=%s): %d" % (side_want, seen))
    print("\n  COVERAGE by source and arm -- read this before any field below:")
    for src in sorted(cov):
        a = cov[src].get("indiv") or [float("nan")]
        b = cov[src].get("inst") or [float("nan")]
        flag = "  <-- arms differ" if abs(st.median(a) - st.median(b)) > 0.05 else ""
        print("    %-22s indiv %.3f   inst %.3f%s"
              % (src, st.median(a), st.median(b), flag))

    #: pair the arms within (lineage, scenario, condition), then take the
    #: per-lineage median, then sign-test over lineages. Same ladder as the
    #: primary: never a pooled number without the per-lineage direction.
    keys = sorted({k[:3] for k in share})
    per_field = collections.defaultdict(lambda: collections.defaultdict(list))
    npair = 0
    for lin, scen, cond in keys:
        a = share.get((lin, scen, cond, "indiv"))
        b = share.get((lin, scen, cond, "inst"))
        if not a or not b:
            continue
        npair += 1
        for f in set(a) | set(b):
            per_field[f][lin].append(b.get(f, 0.0) - a.get(f, 0.0))

    print("\n  paired cells: %d" % npair)
    res = []
    for f, bylin in per_field.items():
        if len(bylin) < 40:            #: present in nearly every lineage or not quoted
            continue
        lm = [st.median(v) for v in bylin.values()]
        n, k, p = sign_test(lm)
        res.append((st.median(lm), k, n, p, f, len(bylin)))

    print("\n  d(share) = inst - indiv, per-lineage median, sign test over lineages")
    print("  RISING FURTHER IN THE INSTITUTIONAL ARM:")
    for m, k, n, p, f, nl in sorted(res, reverse=True)[:top]:
        print("    %+.4f  %2d/%-2d p=%-9.2g %s / %s" % (m, k, n, p, f[0], f[1]))
    print("  RISING FURTHER IN THE INDIVIDUAL ARM:")
    for m, k, n, p, f, nl in sorted(res)[:top]:
        print("    %+.4f  %2d/%-2d p=%-9.2g %s / %s" % (m, k, n, p, f[0], f[1]))
    print("\n  %d fields tested; no multiplicity correction applied, so read the" % len(res))
    print("  ORDERING and the lineage counts, not any single p as a discovery.")


def write_csv(C, J):
    with open(OUT_CELL, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(C[0]))
        w.writeheader()
        w.writerows(C)
    by_lin = collections.defaultdict(list)
    for c in C:
        by_lin[c["lineage"]].append(c)
    with open(OUT_LIN, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lineage", "n_cells", "median_d_js", "median_js_indiv",
                    "median_js_inst", "share_cells_positive"])
        for l, v in sorted(by_lin.items()):
            w.writerow([l, len(v),
                        "%.6f" % st.median([x["d_js"] for x in v]),
                        "%.6f" % st.median([x["js_indiv"] for x in v]),
                        "%.6f" % st.median([x["js_inst"] for x in v]),
                        "%.4f" % (sum(1 for x in v if x["d_js"] > 0) / len(v))])
    print("\nwrote %s" % os.path.relpath(OUT_CELL, ROOT))
    print("      %s" % os.path.relpath(OUT_LIN, ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", action="store_true",
                    help="the arm contrast alone, no secondaries")
    ap.add_argument("--csv", action="store_true",
                    help="also write the tidy cell and lineage tables")
    a = ap.parse_args()

    rows = load_rows()
    design = load_design()
    J = joined(rows, design)
    C, dropped = cells(J)
    print("producer rows %d   kernel %d   joined %d   F21 %d"
          % (len(rows), sum(1 for r in rows if r["stratum"] == KERNEL), len(J),
             sum(1 for r in rows if r["stratum"] == F21)))
    print("paired cells %d   dropped for a missing arm: %d" % (len(C), dropped))

    primary(C)
    if not a.primary:
        secondary_matched(C)
        secondary_f21(rows)
        secondary_movement(J)
        secondary_fields(design, "risers")
        secondary_fields(design, "fallers")
    if a.csv:
        write_csv(C, J)


if __name__ == "__main__":
    main()
