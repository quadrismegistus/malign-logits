#!/usr/bin/env python
"""Z: where in the pipeline the register shift happens, and whether frontier
models show it.

    python z_ladders.py
    python z_ladders.py --only frontier

TWO QUESTIONS THAT A BASE/ALIGNED CONTRAST CANNOT ASK.

**Which step does it.** Four families in the stash carry base -> SFT -> DPO as
separate checkpoints, so the total movement can be decomposed:

    step 1 = SFT - base      socialisation
    step 2 = DPO - SFT       legislation
    total  = DPO - base

The project already has this division for entropy (F18: SFT does 66-81% in
three families) and for surprisal (F15). If the register shift lands mostly at
SFT it is a property of instruction data; if at DPO, of preference data.

**Whether it is an open-weight artefact.** Three frontier models sit in the
stash with `-raw` and non-raw variants on the same prompts. That is the closest
thing in this corpus to the object the argument is actually about, and it is a
different kind of contrast -- not two checkpoints but two access modes of one
deployed system -- so it is reported separately and never pooled with the
ladders.

UNIT IS THE PROMPT, WITHIN A CHAIN. Every chain has 47-73 prompts that all its
stages carry; a prompt only ever compares to itself. Wilcoxon over prompts gives
each chain its own p. Four ladders is too few to test ACROSS chains, so they are
reported individually and the consistency of sign is the only cross-chain claim
made.

WHAT `-raw` MEANS IS NOT DOCUMENTED HERE and I have not verified it. It is
presumably a base-like completion mode against the chat-tuned default, but the
stash does not say, and a `-raw` that merely omits a system prompt is a much
weaker contrast than a base checkpoint. Treated as a mode difference of unknown
depth, which is why no ladder language is used for it.
"""
import argparse
import collections
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")
sys.path.insert(0, HERE)

from malign_logits.cache import CacheManager   # noqa: E402
from malign_logits import fields                # noqa: E402
from y_paired_tests import wilcoxon, boot_ci    # noqa: E402

LADDERS = [
    ("OLMo-2-1B", ["allenai/OLMo-2-0425-1B", "allenai/OLMo-2-0425-1B-SFT",
                   "allenai/OLMo-2-0425-1B-DPO"]),
    ("Olmo-3-7B", ["allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-Instruct-SFT",
                   "allenai/Olmo-3-7B-Instruct-DPO"]),
    ("pythia-6.9B", ["EleutherAI/pythia-6.9b", "lomahony/eleuther-pythia6.9b-hh-sft",
                     "lomahony/eleuther-pythia6.9b-hh-dpo"]),
    #: Tulu's usable SFT rung is the no-safety-data ablation (5,300 passages);
    #: plain SFT has 231 and cannot carry a prompt-matched contrast. So this
    #: chain's step 1 is "SFT WITHOUT SAFETY DATA", which is a different and
    #: arguably more interesting rung, and it is labelled as such rather than
    #: silently standing in for SFT.
    ("Tulu-3-8B (SFT=no-safety)",
     ["meta-llama/Llama-3.1-8B", "allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data",
      "allenai/Llama-3.1-Tulu-3-8B-DPO"]),
]
#: ORDER IS (continuation-prompted, native chat) AND THE SUFFIX IS COUNTERINTUITIVE.
#: From api_generate.py:6-14 and cli.py:1319-1320:
#:     {model}       WITH system prompt "Continue the following text. Write only
#:                   the continuation, no commentary or explanation."
#:     {model}-raw   NO system prompt -- native chat behaviour
#: So `-raw` is the DEPLOYED ASSISTANT end and the bare name is the end talked
#: into behaving like a completion engine. Listed continuation-first so the
#: reported delta (chat - continuation) points the same way as base -> DPO.
FRONTIER = [
    ("claude-sonnet-4-6", ["anthropic/claude-sonnet-4-6", "anthropic/claude-sonnet-4-6-raw"]),
    ("claude-haiku-4-5", ["anthropic/claude-haiku-4-5", "anthropic/claude-haiku-4-5-raw"]),
    ("gpt-4o-mini", ["openai/gpt-4o-mini", "openai/gpt-4o-mini-raw"]),
]
#: The open rungs, for the regime comparison. Base and DPO only: SFT is a rung
#: not every family defines the same way (see the Tulu note above).
OPEN_BASE = [ms[0] for _, ms in LADDERS]
OPEN_DPO = [ms[-1] for _, ms in LADDERS]
WATCH = ["F:emotion_and_arousal", "F:sensory_perception",
         "F:physical_appearance_and_properties",
         "F:logical_modal_and_discourse_operators",
         "F:language_and_communication", "N:dominance=dominant",
         "N:concreteness=concrete", "R:sensation", "R:need"]
MIN_PER_CELL = 3
#: passages read per (model, prompt) cell. Cells hold 40-80; the cell mean
#: stabilises well before that and the cap is what keeps this a minutes-long
#: run rather than an hours-long one. Sampled by sorted `idx` so the same
#: passages are read on every invocation -- an unseeded sample would make two
#: runs of the same script disagree and neither would be checkable.
PER_CELL = 12


def cell(idx, m, p):
    ks = idx.get((m, p), [])
    return sorted(ks, key=lambda k: (k.get("idx") if isinstance(k.get("idx"), int) else 0,
                                     repr(sorted(k.items(), key=repr))))[:PER_CELL]


def profile(text):
    out = {}
    r = fields.count(text)
    n = r["n_counted"] or 0
    if n < 5:
        return None
    for g, c in r["counts"].items():
        out["F:" + g] = c / n
    for dim, x in fields.norms(text).items():
        t = sum(x["counts"].values())
        if t:
            for b, c in x["counts"].items():
                out["N:%s=%s" % (dim, b)] = c / t
    rd = fields.count(text, "rid")["counts"]
    rt = sum(rd.values())
    if rt >= 3:
        for g, c in rd.items():
            k = "R:" + g.split(":")[0]
            out[k] = out.get(k, 0) + c / rt
    return out


def stage_profiles(st, idx, models, prompts):
    """prompt -> stage index -> {measure: mean}. Drops a prompt unless every
    stage has MIN_PER_CELL usable passages for it."""
    out = {}
    for p in prompts:
        per = []
        ok = True
        for m in models:
            ks = idx.get((m, p), [])
            if len(ks) < MIN_PER_CELL:
                ok = False
                break
            acc = collections.defaultdict(list)
            for k in ks:
                try:
                    txt = st.get(k)
                except Exception:
                    continue
                if not isinstance(txt, str) or len(txt) < 200:
                    continue
                pr = profile(txt)
                if not pr:
                    continue
                for g, v in pr.items():
                    acc[g].append(v)
            if not acc:
                ok = False
                break
            per.append({g: statistics.mean(v) for g, v in acc.items()})
        if ok and len(per) == len(models):
            out[p] = per
    return out


#: CAPTURE-ONLY WRITES, ADDED 2026-08-12 (producer-debt class 1A).
#: `mark()` computed the median, the Wilcoxon p and the agreement marker and
#: DISCARDED all three into a format string; `report()` printed the table and
#: kept nothing. Z's section 1 (the SFT/DPO cancellation) and section 5 (the
#: closed-system comparison) therefore had no data behind them -- the finding
#: could be re-run and hoped to match, not re-queried or audited.
#:
#: **NOTHING BELOW CHANGES A COMPUTATION.** `mark()` records the values it
#: already returns and returns the same string; `report()` files them under the
#: keys it already prints. That is what makes the re-run a TEST of the document
#: rather than a revision of it -- if a write had required altering a
#: calculation, agreement would have proved nothing (the M04 A precedent,
#: [5439]).
CAPTURE = {}
_CAP_KEY = [None]
_CAP_STEP = [None]
_WRITE = [None]


def mark(d):
    """One cell: median, Wilcoxon p, and WHETHER THE TWO STATISTICS AGREE.

    They measure different things and can legitimately disagree. In two Olmo-3
    cells 63% of prompts moved up while the few that moved DOWN moved further
    (mean |neg| 2.14 vs |pos| 1.49): the median is displaced from zero, and the
    signed-rank, which ranks by magnitude, is not. Both correct.

        *   both agree the effect is there
        ~   median CI excludes zero but signed-rank does not (or vice versa)
            -- a displaced median with magnitude-asymmetric tails, NOT
               significance, and not quotable as significance
    """
    if len(d) < 12:
        return "%20s" % "-"
    wp, _ = wilcoxon(d)
    lo, hi = boot_ci(d)
    ci = lo > 0 or hi < 0
    p = wp == wp and wp < 0.05
    m = "*" if (ci and p) else ("~" if (ci or p) else " ")
    if _CAP_KEY[0] is not None:
        CAPTURE.setdefault(_CAP_KEY[0], {})[_CAP_STEP[0]] = {
            "median_pp": 100 * statistics.median(d), "wilcoxon_p": wp,
            "ci_lo_pp": 100 * lo, "ci_hi_pp": 100 * hi, "agree": m.strip() or None,
            "n": len(d)}
    return "%+9.2f p%-6.3f%s" % (100 * statistics.median(d), wp, m)


def report(name, models, prof, labels):
    print("  %s   %d prompts" % (name, len(prof)))
    if len(prof) < 12:
        print("     too few shared prompts with data at every stage\n")
        return
    print("     %-38s %s" % ("measure", " ".join("%20s" % l for l in labels)))
    print("     " + "-" * (39 + 21 * len(labels)))
    for g in WATCH:
        steps = [(i, i + 1) for i in range(len(models) - 1)]
        if len(models) > 2:
            steps.append((0, len(models) - 1))
        cells = []
        for i, j in steps:
            _CAP_KEY[0] = "%s|%s" % (name, g)
            _CAP_STEP[0] = labels[len(cells)] if len(cells) < len(labels) else "%d->%d" % (i, j)
            cells.append(mark([pr[j].get(g, 0) - pr[i].get(g, 0) for pr in prof.values()
                               if g in pr[i] and g in pr[j]]))
        _CAP_KEY[0] = None
        print("     %-38s %s" % (g, " ".join(cells)))
    print("     * both tests agree   ~ they disagree, not significance\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("ladders", "frontier", "safety"), default=None)
    ap.add_argument("--write", metavar="PATH", nargs="?", const=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "z_ladders.json"),
        help="capture-only: write every reported cell to JSON")
    a = ap.parse_args(argv)
    _WRITE[0] = a.write
    st = CacheManager()._stash("generations")
    #: TULU_SFT is in NEITHER list -- the ladder carries the no-safety ablation
    #: in its place -- so omitting it here emptied the index for it and the
    #: safety section printed "0 prompts" where an independent count of the
    #: stash found 35. A model missing from `want` is indistinguishable in the
    #: output from a model missing from the corpus.
    want = {m for _, ms in LADDERS + FRONTIER for m in ms} | {TULU_SFT}
    idx = collections.defaultdict(list)
    for k in st.keys():
        if k.get("temp") != 1.0:
            continue
        m = k.get("model") or ""
        if ":" in m or m not in want:
            continue
        idx[(m, k.get("prompt"))].append(k)
    print("indexed %s (model, prompt) cells\n" % format(len(idx), ","))

    if a.only == "safety":
        print("=" * 100)
        print("SAFETY DATA -- Tulu SFT with and without it")
        print("=" * 100)
        safety(st, idx)
        if _WRITE[0]:
            _write(_WRITE[0], a.only)
        return 0

    if a.only != "frontier":
        print("=" * 100)
        print("LADDERS -- which step does the work")
        print("=" * 100)
        for name, ms in LADDERS:
            ps = set.intersection(*[{p for (mm, p) in idx if mm == m} for m in ms]) \
                if all(any(mm == m for (mm, _) in idx) for m in ms) else set()
            report(name, ms, stage_profiles(st, idx, ms, ps),
                   ["step1 SFT-base", "step2 DPO-SFT", "TOTAL DPO-base"])

    if a.only != "ladders":
        print("=" * 100)
        print("FRONTIER -- ONE SYSTEM LINE, SAME WEIGHTS")
        print("=" * 100)
        print("""  NOT an alignment contrast. Both conditions are the same RLHF'd model; the
  only difference is whether a system prompt tells it to continue text. So this
  measures HOW MUCH OF THE REGISTER IS PROMPT-RECOVERABLE, which is a different
  and narrower question than what the ladders answer. Whatever a system line
  fails to move is the part that is in the weights.
""")
        for name, ms in FRONTIER:
            ps = set.intersection(*[{p for (mm, p) in idx if mm == m} for m in ms]) \
                if all(any(mm == m for (mm, _) in idx) for m in ms) else set()
            report(name, ms, stage_profiles(st, idx, ms, ps), ["chat - continuation"])

        print("=" * 100)
        print("REGIMES -- where closed deployed models sit on the open base->DPO axis")
        print("=" * 100)
        regimes(st, idx)

        print("=" * 100)
        print("REGIMES II -- WITHIN-MODEL EFFECT SIZES, open pipeline vs closed system line")
        print("=" * 100)
        within(st, idx)

    #: THE WRITE IS CALLED, and it is called on both exit paths. A `_write`
    #: defined and never invoked is the same debt with a file in front of it --
    #: this campaign has shipped that exact shape before.
    if _WRITE[0]:
        _write(_WRITE[0], a.only)
    return 0


def within(st, idx):
    """The regime comparison with NO cross-family levels in it.

    The projection above has one unavoidable weakness: its numerator compares
    LEVELS across model families, because frontier models have no base of their
    own. This table avoids that entirely by comparing two sets of WITHIN-model,
    prompt-matched deltas -- every number is a model against itself.

        open   base -> DPO             the full SFT+DPO pipeline, weights changed
        closed continuation -> chat    one system line, weights identical

    THESE ARE NOT THE SAME MANIPULATION and the comparison is only interesting
    because they are not. One is retraining, the other is a sentence at inference
    time. Reading a larger closed number as "more alignment" is wrong; what it
    licenses is "a system line moves this measure further than retraining does."

    THE CONFOUND, AND IT IS SPECIFIC. The system prompt reads "Continue the
    following text. Write only the continuation, no commentary or explanation."
    It NAMES commentary. So `language_and_communication` and
    `logical_modal_and_discourse_operators` are partly a compliance check on the
    instruction and cannot be read as register discovery. Measures the
    instruction does not name -- concreteness, sensory perception, physical
    appearance, emotion -- carry no such shortcut, and those are the informative
    rows. Marked NAMED below so the distinction survives being quoted.
    """
    NAMED = {"F:language_and_communication",
             "F:logical_modal_and_discourse_operators"}

    def deltas(models, _label=None):
        ps = set.intersection(*[{p for (mm, p) in idx if mm == m} for m in models])
        prof = stage_profiles(st, idx, models, ps)
        out = {}
        for g in WATCH:
            d = [pr[-1][g] - pr[0][g] for pr in prof.values()
                 if g in pr[0] and g in pr[-1]]
            if len(d) < 12:
                continue
            wp, _ = wilcoxon(d)
            lo, hi = boot_ci(d)
            out[g] = (statistics.median(d), (lo > 0 or hi < 0) and wp == wp and wp < 0.05)
            #: THE PROMPT COUNT IS THE POPULATION AND IT WAS DISCARDED. Section
            #: 5 is prompt-matched, so its open column is a SUBSET of section
            #: 1's -- OLMo-2-1B dominance is +0.26 there and +0.05 here on the
            #: same data. If the intersection grows, section 5 drifts and
            #: section 1 does not, and without n nobody can tell that from an
            #: arithmetic error. Recorded per (side, family, measure).
            if _label:
                CAPTURE.setdefault("WITHIN_N|%s" % _label, {})[g] = {
                    "n_prompts": len(d), "shared_prompts_in_chain": len(ps)}
        return out

    op = {name: deltas([ms[0], ms[-1]], 'open|%s' % name) for name, ms in LADDERS}
    cl = {name: deltas(ms, 'closed|%s' % name) for name, ms in FRONTIER}
    print("  open = base->DPO (retraining, %d families)   closed = cont->chat "
          "(one system line, %d models)" % (len(op), len(cl)))
    print("  * = both tests clear in that model. A DIRECTION CLAIM NEEDS STARS ON"
          " BOTH SIDES:\n  an unstarred column is a measure that did not move there, which is not"
          " the\n  same thing as a measure that moved the other way.\n")
    #: NAME THE COLUMNS. An unlabelled row of four numbers gets quoted with the
    #: families guessed back in the wrong order.
    print("  open order:   %s" % " | ".join(op))
    print("  closed order: %s\n" % " | ".join(cl))
    print("  %-38s %-30s %-24s %8s %s"
          % ("measure", "OPEN per family", "CLOSED per model", "|cl|/|op|", "verdict"))
    print("  " + "-" * 122)
    for g in WATCH:
        o = [op[f][g] for f in op if g in op[f]]
        c = [cl[f][g] for f in cl if g in cl[f]]
        if not o or not c:
            continue
        ov, cv = [v for v, _ in o], [v for v, _ in c]
        nso, nsc = sum(s for _, s in o), sum(s for _, s in c)
        mo, mc = (statistics.median([abs(v) for v in ov]),
                  statistics.median([abs(v) for v in cv]))
        ratio = ("%.1fx" % (mc / mo)) if mo > 0.0005 else "-"
        #: A comparison of directions requires an effect on each side to compare.
        #: Without that, "OPPOSITE" is asserted about a measure that simply did
        #: not move -- which is how a null becomes an inversion in a write-up.
        if nso < 2 or nsc < 2:
            verdict = "not comparable (%d/%d open, %d/%d closed clear)" % (
                nso, len(o), nsc, len(c))
        elif statistics.median(ov) * statistics.median(cv) > 0:
            verdict = "SAME DIRECTION, both sides clear"
        else:
            verdict = "INVERTS, both sides clear"
        #: CAPTURE-ONLY. Section 5's table is printed here with its OWN
        #: formatter and never went through `mark()`, so the ladders hook did
        #: not reach it -- the frontier run captured 27 report() cells and zero
        #: of these. Recording the same v/s pairs the line below prints.
        CAPTURE.setdefault("WITHIN|%s" % g, {})["row"] = {
            "open": [{"value_pp": 100 * v, "clears": bool(s)} for v, s in o],
            "closed": [{"value_pp": 100 * v, "clears": bool(s)} for v, s in c],
            "median_abs_open_pp": 100 * mo, "median_abs_closed_pp": 100 * mc,
            "ratio": ratio, "verdict": verdict, "named": g in NAMED,
            "n_open": len(o), "n_closed": len(c),
            "n_clear_open": nso, "n_clear_closed": nsc}
        print("  %-38s %-30s %-24s %8s %s%s"
              % (g,
                 " ".join("%+5.2f%s" % (100 * v, "*" if s else " ") for v, s in o),
                 " ".join("%+5.2f%s" % (100 * v, "*" if s else " ") for v, s in c),
                 ratio, verdict, "  [NAMED]" if g in NAMED else ""))
    print("\n  ratio compares MEDIAN ABSOLUTE within-model movement: magnitude only."
          "\n  [NAMED] = the system prompt names commentary, so that row is partly a"
          "\n  compliance check on the instruction and not register discovery.\n")


TULU_BASE = "meta-llama/Llama-3.1-8B"
TULU_SFT = "allenai/Llama-3.1-Tulu-3-8B-SFT"
TULU_NOSAFE = "allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data"
TULU_DPO = "allenai/Llama-3.1-Tulu-3-8B-DPO"


def safety(st, idx):
    """Does safety data change what SFT does to the register?

    The ladder above uses the no-safety ablation as Tulu's SFT rung because
    plain SFT is thin. This runs the contrast that rung was standing in for.

    THE ARM IS UNDERPOWERED AND THE ASYMMETRY IS THE REASON. Plain SFT holds 231
    passages over 41 prompts, a median of THREE per cell, against the ablation's
    100. Thirty-five prompts have both arms at >=3; ZERO have both at >=5. Three
    passages still give an unbiased estimate of that cell's mean, so the contrast
    is valid, but the noisy arm sets the variance of every per-prompt delta and
    the whole comparison inherits it.

    SO A NULL HERE IS NOT EVIDENCE OF NO EFFECT. It is what this many passages
    can resolve, which is not much. Effects in the four-family ladder run 0.5 to
    3pp; anything smaller than that is below what 35 prompts against a 3-passage
    arm will show. Read the direction and the size, treat the p as advisory, and
    do not report an absent star as "safety data does nothing".

    Three contrasts, because the direct one alone cannot separate two readings:

        SFT      - base     what SFT does WITH safety data      (39 prompts)
        NOSAFE   - base     what SFT does WITHOUT it            (71 prompts)
        SFT      - NOSAFE   the direct difference               (35 prompts)

    The first two are on different prompt populations and different per-cell
    depths, so they are NOT subtractable one from the other; the third is the
    only prompt-matched statement about safety data itself.
    """
    combos = [("SFT - base", [TULU_BASE, TULU_SFT]),
              ("NOSAFE - base", [TULU_BASE, TULU_NOSAFE]),
              ("SFT - NOSAFE", [TULU_NOSAFE, TULU_SFT]),
              ("DPO - NOSAFE", [TULU_NOSAFE, TULU_DPO])]
    cols = {}
    for label, ms in combos:
        ps = set.intersection(*[{p for (mm, p) in idx if mm == m} for m in ms])
        prof = stage_profiles(st, idx, ms, ps)
        depth = min(statistics.median([len(cell(idx, m, p)) for p in prof] or [0])
                    for m in ms)
        cols[label] = (prof, int(depth))
    for label, (prof, depth) in cols.items():
        print("  %-16s %3d prompts, thinnest arm median %d passages/cell"
              % (label, len(prof), depth))
    print()
    print("  %-38s %s" % ("measure", " ".join("%20s" % l for l, _ in combos)))
    print("  " + "-" * (39 + 21 * len(combos)))
    for g in WATCH:
        cells = []
        for label, _ in combos:
            prof, _ = cols[label]
            cells.append(mark([pr[-1].get(g, 0) - pr[0].get(g, 0) for pr in prof.values()
                               if g in pr[0] and g in pr[-1]]))
        print("  %-38s %s" % (g, " ".join(cells)))
    print("\n  * both tests agree   ~ they disagree, not significance")
    print("  THE `SFT - NOSAFE` COLUMN IS THE ONLY PROMPT-MATCHED STATEMENT ABOUT"
          "\n  SAFETY DATA, and it is the weakest-powered column here. An absent star"
          "\n  in it means unresolved, not absent.\n")


def regimes(st, idx, min_families=3):
    """Locate frontier output on the axis the open ladders define.

    NOT A PAIRED TEST AND CANNOT BE. Frontier models have no base checkpoint, so
    there is no closed-model alignment delta to measure. What is available is a
    LOCATION: the open families define an axis (base -> DPO, prompt-matched and
    tested above), and frontier output can be placed on it. Every cross-model
    comparison here is confounded by scale, pretraining corpus and architecture,
    exactly as the human-prose baseline is. No p-values on the frontier side.

    FRAMING DOES NOT MATCH ACROSS THE OPEN/CLOSED LINE, and it cannot be made to.
    Open generations are bare completions with no chat template. Neither frontier
    condition is that: `-raw` is the chat endpoint with no system prompt, and the
    bare name is the chat endpoint instructed to continue. So the closed side is
    reported as a RANGE spanned by its two conditions rather than a point, and
    the width of that range is the framing uncertainty made visible instead of
    hidden by picking one.

    THE AXIS FRACTION is (frontier - base) / (DPO - base) computed on medians,
    not per prompt: a per-prompt ratio explodes wherever a prompt's denominator
    is near zero, and the median of such ratios is dominated by those prompts.
    Read >1 as "beyond open DPO in the same direction", 0..1 as "less shifted
    than open DPO", <0 as "the other way".
    """
    fam = {name: ms for name, ms in LADDERS}
    fcond = {name: ms for name, ms in FRONTIER}
    #: prompt -> family -> (base level, dpo level); and prompt -> frontier levels
    lvl = collections.defaultdict(dict)
    for name, ms in LADDERS:
        two = [ms[0], ms[-1]]
        ps = {p for (m, p) in idx if m == two[0]} & {p for (m, p) in idx if m == two[1]}
        for p, per in stage_profiles(st, idx, two, ps).items():
            lvl[p][name] = per
    fl = collections.defaultdict(dict)
    for name, ms in FRONTIER:
        ps = set.intersection(*[{p for (mm, p) in idx if mm == m} for m in ms])
        for p, per in stage_profiles(st, idx, ms, ps).items():
            fl[p][name] = per

    shared = [p for p in lvl if len(lvl[p]) >= min_families and p in fl]
    print("  prompts with >=%d open families and frontier data: %d" % (min_families, len(shared)))
    print("  open families: %d   frontier models with data: %d\n"
          % (len(fam), len({n for p in shared for n in fl[p]})))
    if len(shared) < 12:
        print("  too few shared prompts to locate anything\n")
        return

    print("  %-36s %7s %7s %14s %13s %11s %s"
          % ("measure", "opnBASE", "paired", "frontier c..x", "gap vs base", "axis frac", "reading"))
    print("  %-36s %7s %7s %14s %13s %11s"
          % ("", "level", "span", "levels", "pp", "gap/span"))
    print("  " + "-" * 116)
    for g in WATCH:
        b, d, fc, fx = [], [], [], []
        #: family -> per-prompt WITHIN-FAMILY deltas, kept separately from the
        #: levels below. See the span note.
        fam_d = collections.defaultdict(list)
        for p in shared:
            bs = [v[0].get(g) for v in lvl[p].values() if g in v[0]]
            ds = [v[-1].get(g) for v in lvl[p].values() if g in v[-1]]
            cs = [v[0].get(g) for v in fl[p].values() if g in v[0]]
            xs = [v[-1].get(g) for v in fl[p].values() if g in v[-1]]
            if len(bs) < min_families or len(ds) < min_families or not cs or not xs:
                continue
            b.append(statistics.median(bs))
            d.append(statistics.median(ds))
            fc.append(statistics.median(cs))
            fx.append(statistics.median(xs))
            for fname, v in lvl[p].items():
                if g in v[0] and g in v[-1]:
                    fam_d[fname].append(v[-1][g] - v[0][g])
        if len(b) < 12:
            continue
        B, D, C, X = (statistics.median(v) for v in (b, d, fc, fx))
        #: THE SPAN IS A POOLED DIFFERENCE, NOT A DIFFERENCE OF POOLED LEVELS.
        #: `D - B` is the latter: it medians base levels across families, medians
        #: DPO levels across families, and subtracts -- which discards the
        #: prompt-and-family pairing that the ladder result rests on, and lets
        #: between-family level differences swamp the within-family shift. On
        #: concreteness it collapsed a real axis to 0.97pp against per-family
        #: totals of -3.12/-0.53/-2.02/-1.66, and eight of nine measures then
        #: printed as "too flat to project onto". So the denominator is the
        #: ladder's own quantity: median across families of that family's
        #: prompt-matched delta. The frontier NUMERATOR is unavoidably a levels
        #: comparison -- frontier models have no base of their own -- and that
        #: asymmetry is the standing weakness of this whole table.
        fd = [statistics.median(v) for v in fam_d.values() if len(v) >= 12]
        if len(fd) < min_families:
            continue
        span = statistics.median(fd)
        #: A FRACTION IS A RATIO AND ITS DENOMINATOR IS THE OPEN AXIS, which on
        #: several of these measures is a quarter of a percentage point wide. A
        #: frontier gap of 1pp then prints as "4x open DPO" and reads as a huge
        #: effect when the absolute distance is small and the multiplier is an
        #: artefact of dividing by something near zero. So: the absolute gap is
        #: printed beside the fraction, and a span under 1pp disqualifies the
        #: fraction rather than merely qualifying it.
        gap = "%+5.2f..%+5.2f" % (100 * (C - B), 100 * (X - B))
        if abs(span) < 0.010:
            frac = "%11s" % "-"
            read = "open axis only %.2fpp wide, no ratio" % (100 * abs(span))
        else:
            lo, hi = sorted(((C - B) / span, (X - B) / span))
            frac = "%5.2f..%-5.2f" % (lo, hi)
            if lo > 1:
                read = "BEYOND open DPO, same direction"
            elif hi < 0:
                read = "opposite direction to open alignment"
            elif lo > 0:
                read = "same direction, short of open DPO"
            else:
                read = "brackets the base level"
        print("  %-36s %7.2f %+7.2f %6.2f..%-6.2f %13s %11s %s"
              % (g, 100 * B, 100 * span, 100 * C, 100 * X, gap, frac, read))
    print("\n  frontier column is a RANGE over its two framings, not a measurement"
          "\n  gap = frontier - openBASE in pp; frac = gap / (openDPO - openBASE)"
          "\n  READ THE GAP, NOT THE FRACTION, wherever the open span is narrow\n")

def _write(path, argv_only):
    import json
    meta = {"produced_by": "meta/M01_displacement/scripts/z_ladders.py --write",
            "capture": "capture-only; no computation altered (producer-debt 1A)",
            "only": argv_only, "cells": len(CAPTURE),
            "note": "keys are 'ladder|measure'; steps are the printed column labels"}
    json.dump({"_meta": meta, "cells": CAPTURE}, open(path, "w"), indent=1)
    print("wrote %d cells -> %s" % (len(CAPTURE), path))



if __name__ == "__main__":
    sys.exit(main())
