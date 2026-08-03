"""M04 producer — the syntagmatic campaign's coarse claim, Route A.

WRITTEN TO THE FROZEN CHARTER `41f353707d0fa904` (meta/M04_syntagmatic/).
Every gate below cites the charter section it implements. Nothing here is a
choice; where the charter is silent this script REFUSES rather than decides.

THE READING FREEZE ON THE BEAM STASHES ([2245].3) LIFTS ONLY THROUGH THIS FILE,
after malign's code audit and the pen's clearance. That is why the stages are
ordered as gates that HALT rather than as sections that print: a stage which
cannot pass must not let the next one read.

    STAGE 1  COVERAGE          charter §3   -- arm-agnostic, no contrast
    STAGE 2  DIVERGENCE CTRL   charter §4.2a -- MUST FAIL if forcing is not
                               behaving as forcing. Halts before any contrast.
    STAGE 3  RE-POWERING       charter §6   -- realized sd from first cells,
                               REQUIRED first print, not an optional check
    STAGE 4  CONTRASTS         charter §5   -- only reachable if 1-3 pass

RUN ORDER IS NOT CONFIGURABLE. There is no flag that skips a gate.
"""

import argparse
import collections
import contextlib
import io
import json
import math
import os
import re
import statistics as st
import sys

ROOT = os.path.expanduser("~/github/malign-logits")
LINEAGE_MAP = os.path.join(ROOT, "data", "lineage_map_models.json")
CATEGORISATION = os.path.join(ROOT, "data", "prompt_categorisation.json")

CHARTER_HASH = "41f353707d0fa904"

#: THE NINE CATEGORIES, ruled at [2356]/[2357]. Their AUTHORITY is
#: data/battery_results.csv's labels, not the categorisation's `domain` --
#: `domain`/`subdomain` is a ROUTE to them and enumerating it gave 8 of a
#: 32-value field, which is a different question correctly answered.
BATTERY_CATEGORIES = frozenset({
    "death", "neutral", "power", "profanity", "sexual_explicit",
    "sexual_liminal", "substance", "violence_explicit", "violence_liminal",
})

#: A TRUNCATION IS A STRICT PREFIX OF A LONGER REAL NAME ([2358]).
#: My first detector was "exactly 20 characters", which is a coincidence
#: detector: `deepseek_llm_7b_base` is 20 chars and COMPLETE, while
#: `Llama_3.1_Tulu_3_8B_` is 20 chars and a prefix of six. The trailing
#: separator is the signature; the length never was. Falsely flagged names are
#: exactly the ones a drop-the-unrecoverable rule would discard.
def truncation_candidates(label, known):
    """TOMBSTONE. Superseded by the registry resolution at [2434].

    RETAINED AND CORRECTED RATHER THAN DELETED, per [2497]: unreachable code is
    read before burial. Reading it found a DEFECT, not a ruling -- its original
    test was `label.endswith("_")`, which misses every truncation that cuts
    MID-TOKEN (8 of the 15 real cases). A known-wrong detector sitting uncalled
    is a trap for whoever wires it next, so the body is fixed to the correct
    test: STRICT PREFIX OF AN EXISTING LABEL. Identity, not a separator
    coincidence -- the same shape-vs-identity rule as everywhere else tonight.

    NOT WIRED, and case 22 asserts it stays that way ([2497]: a tombstone must
    be VERIFIABLY UNREACHABLE or it is an understudy).
    """
    s_ = str(label)
    return sorted(k for k in known if k != s_ and str(k).startswith(s_))

#: LITERAL output filenames. [2282]'s non-test audit item: a producer whose
#: output filename is COMPUTED is invisible to `trace_producers.py` and becomes
#: an orphan the moment its author stops remembering it. 23 files in this repo
#: are in that position. No f-strings here, ever.
OUT_COVERAGE  = "data/m04_coverage.csv"
OUT_DEPTH     = "data/m04_depth_profile.csv"
OUT_CONTRASTS = "data/m04_contrasts.csv"

#: charter §4.2 -- the branches read pos1..10; pos0 is reported SEPARATELY
#: because F28's first-token spike (+2.24 vs ~+0.85 plateau, 2.6x, content-blind,
#: 13 of 19 families) would otherwise be read as combination damage localizing.
POS0 = 0
CURVE = list(range(1, 10))          # positions 1..9 of a length-10 vector

#: charter §5 -- clearing thresholds, one-sided sign test at .05
CLEARS = {25: 18, 28: 19, 34: 23}

#: charter §4.2a's control, REDERIVED at [2390]/[2393]/[2395] from the true
#: field semantics (beam.py:547-557). The frozen charter TEXT is unamended --
#: the defect was this pipeline's field choice, nothing else.
#:
#:   base_token_probs                     SOURCE's prob of its own tokens
#:                                        = the NATIVE path, judge-invariant
#:   annotations[j]["token_probs"]        JUDGE's prob of the SOURCE's tokens
#:                                        = the CROSS path
#:   annotations[j]["token_resist"]       log2(source) - log2(judge) per position
#:                                        = §4.2a's "difference through one judge"
#:
#: All curves in LOG2 space or the comparison is incommensurable (probabilities
#: against bits -- the error "materially" was hiding at [2390].3). By linearity,
#: slope(resist) == slope(log2 source) - slope(log2 judge) EXACTLY, so the
#: control asks whether the two raw log-slopes are CLOSE.
#: RULING 2401-slice / RULING charter-4.2 -- the curve is 1..9; pos0 is
#: reported separately and never enters it.
CTRL_POSITIONS = list(range(1, 10))   # pos0 quarantined per §4.2, reported apart
CTRL_MARGIN = 0.50                    # declared convention, blind, can fail
CTRL_FLAT_FLOOR = 0.01                # bits/position; below this raw paths are
                                      # FLAT -> charter's own failure -> HALT
#: SHORT-CELL RULE, declared blind at [2395]: a 7-point slope and a 10-point
#: slope are different measurements, so they are never mixed. Cells whose resist
#: vector is shorter than 10 are DROPPED AND COUNTED -- the collectors' own
#: decision (b), never imputed, never variance-weighted into the same estimate.
CTRL_REQUIRED_LEN = 10


#: PER-CELL RELABELING, ruled into the declaration at [2411]. The producer
#: scores range(prompt_len-1, len(full)-1) with targets full[pos+1], so probs[j]
#: is the token at full[prompt_len+j] -- and `full` is a RETOKENISATION of
#: prompt+" "+text, so it can carry K extra leading positions before the first
#: generated token. K is a PER-CELL fact (58.7% of cells K=1, 18.6% K=0), never
#: global, and cells where K is not in {0,1} are RESEGMENTED: no alignment
#: exists, they are DROPPED AND COUNTED ([2408], collectors' decision (b)).
def cell_offset(probs, tokens):
    """K = extra leading positions, or None if the cell is unalignable."""
    if not probs or not tokens:
        return None
    k = len(probs) - len(tokens)
    return k if k in (0, 1) else None


def relabel(probs, tokens):
    """Drop the K leading positions so index i is generated-token i."""
    k = cell_offset(probs, tokens)
    return None if k is None else list(probs)[k:]


def _log2_curve(probs):
    """log2 of a probability vector, with the producer's own 1e-10 floor."""
    return [math.log2(max(float(x), 1e-10)) for x in probs]


def gate_2_divergence_rederived(cells):
    """charter §4.2a on the TRUE quantities. THIS GATE CAN FAIL.

    cells: list of (source_log, judge_log, resist) triples, each length-10.
    """
    def med_slope(curves):
        sl = [_slope([c[i] for i in CTRL_POSITIONS]) for c in curves]
        sl = [x for x in sl if x is not None]
        return st.median(sl) if sl else None

    src = med_slope([c[0] for c in cells])
    jdg = med_slope([c[1] for c in cells])
    res = med_slope([c[2] for c in cells])
    if src is None or jdg is None or res is None:
        print("STAGE 2 — HALT: no usable curves.")
        return False

    ref = min(abs(src), abs(jdg))
    print(f"\nSTAGE 2 — DIVERGENCE CONTROL, REDERIVED (charter §4.2a)")
    print(f"  cells (10-position only)       {len(cells):>8,}")
    print(f"  median slope log2(source)      {src:>+8.4f} bits/position")
    print(f"  median slope log2(judge)       {jdg:>+8.4f}")
    print(f"  median slope resist            {res:>+8.4f}")
    print(f"  reference = min(|src|,|judge|) {ref:>8.4f}")
    print(f"  criterion  |resist| < {CTRL_MARGIN} x ref  "
          f"= {CTRL_MARGIN*ref:.4f}")

    if ref < CTRL_FLAT_FLOOR:
        print(f"\n  *** HALT. Raw paths FLAT (ref {ref:.4f} < {CTRL_FLAT_FLOOR}). ***")
        print("  No common trend to cancel; the charter's own stated failure.")
        return False
    if abs(res) >= CTRL_MARGIN * ref:
        print(f"\n  *** HALT. The difference does NOT cancel the common trend. ***")
        print(f"  |{res:.4f}| >= {CTRL_MARGIN*ref:.4f}. §4.2a's control FAILS.")
        return False
    print(f"\n  PASS — resist is flatter than the flatter raw path by the "
          f"declared margin.")
    return True

#: THE CASE MANIFEST. [2313]: "a named gap is easier to skip than an unnamed one,
#: because the naming feels like the work" -- `norm` was named at [2301],
#: renumbered for at [2305], and skipped TWICE while cases the author had been
#: SHOWN the need for got written around the hole.
#:
#: THE REPAIR IS STRUCTURAL, NOT ATTENTIONAL. A case named here but not produced
#: by selftest() FAILS THE SELF-TEST AND HALTS THE PRODUCER. When an auditor
#: names a case, it is added to this set IMMEDIATELY -- before it is written --
#: so the file refuses to run until it exists. A named gap now costs a halt
#: rather than an act of memory.
#: ─────────────────────────────────────────────────────────────────────────
#: GOVERNING RULINGS. [2481], regime-level. Every ruling this producer is
#: bound by, as a literal list, with the self-test asserting ONE IMPLEMENTING
#: REFERENCE per entry (marker `RULING <key>` in the body).
#:
#: [2480].3 is why this exists and nothing else could have found it: A MISSING
#: IMPLEMENTATION HAS NO SIGNATURE. Mutation has nothing to mutate,
#: reachability nothing to reach, the self-test no case to fail. NINE CLEAN
#: AUDIT ROUNDS WERE COMPATIBLE WITH A SETTLED RULING NEVER HAVING BEEN
#: WRITTEN DOWN. Conformance must compare against an EXTERNAL list and can
#: never be derived from the file.
#: ─────────────────────────────────────────────────────────────────────────
GOVERNING_RULINGS = {
    "2401-slice":  "the curve is positions 1..9; pos0 is not part of it",
    "2401-retain": "retain slices with n >= 10, not n == 10",
    "2429-both":   "BOTH sides re-scored, one segmentation, no partition",
    "2431-vocab":  "per-position resist ONLY for same-tokenizer pairs; "
                   "cross-tokenizer cells are REMOVED, not re-scored",
    "2417-void":   "3 void pairs / 288 cells excluded, counted on the face",
    "2474-drops":  "sub-10 drops COUNTED and reported, never a silent continue",
    "charter-4.2": "pos0 reported separately, quarantined from the curve",
    "2408-offset": "per-cell K = len(probs)-len(tokens) must be in {0,1}; "
                   "K>=2 is RESEGMENTED, no alignment exists, drop and count",
}

REQUIRED_CASES = {
    "1 commensurability",
    "2 lineage unit",
    "3 degenerate pattern",
    "4 empty tripwire",
    "5 pos0 excluded",
    "6 control can fail",
    "7 every gate halts",
    "8 gates+guards wired",
    "9 produces outputs",
    "10 norm collapses namespaces",
    "11 beam positions averaged",
    "12 no-native cells dropped",
    "13 pos0 returned separately",
    "14 raw curves undifferenced",
    "15 truncation is structural",
    "16 production refuses stored field",
    "17 drops are counted not silent",
    "18 every ruling has an implementing line",
    "19 cross-vocabulary cells refused",
    "20 void pairs excluded and counted",
    "21 production halts without the re-score",
    "22 vocabulary rule is ON by default",
    "23 tombstones are unreachable",
    "24 log2 space is entered and pinned",
}

#: A FLOOR, NOT A COUNT. [2318].2: the manifest failed correctly in one
#: direction (a name with no case HALTS) and not the other -- setting
#: REQUIRED_CASES = set() passed 10 of 10, because an empty manifest demands
#: nothing. AN ADVOCATE THAT CAN BE DISMISSED IS THE SHAPE OF THE FAILURE IT
#: WAS BUILT TO PREVENT: not writing a case, and not being told.
#:
#: This is monotone: it RISES when cases are added and never falls. Removing a
#: name now fails rather than relaxes.
#:
#: HONEST LIMIT: this does not make subversion impossible -- the floor is an
#: integer in this file. It makes it require an explicit DOWNWARD edit of a
#: number whose comment says it never goes down, which is visible in a diff,
#: where deleting one set element is a one-word change that reads as tidying.
#: That is the most a self-test can do about its own dismissal, and claiming
#: more would be the [1401] shape at the top of the stack.
REQUIRED_CASES_FLOOR = 24


# ─────────────────────────────────────────────────────────────────────────────
# IDENTITY. charter §3: resolve through the lineage map, NEVER by string.
# ─────────────────────────────────────────────────────────────────────────────

def norm(s):
    """Stash names drop the org and flatten punctuation.

    `allenai/Olmo-3-1025-7B` -> `Olmo_3_1025_7B`. Some annotation keys keep
    dots (`Llama_3.1_8B`), so dots collapse too. This is the [2251].3 hazard:
    `source` is a normalised label and `model` is a HuggingFace ID, and
    comparing them directly returns a clean, confident, FALSE ZERO.
    """
    s = str(s)
    s = s.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


class Lineages:
    """model-id or stash-label -> independent pretraining lineage.

    charter §3: the unit is the lineage. 103 models is never the n; 48 family
    labels are never the n.
    """

    def __init__(self, path=LINEAGE_MAP):
        m = json.load(open(path))
        self.m2l = m["model_to_lineage"]
        # a SECOND index keyed on the normalised form, so a stash label
        # resolves without ever being compared to a HuggingFace id as a string
        self.norm2l = {}
        for mid, lin in self.m2l.items():
            self.norm2l.setdefault(norm(mid), lin)
        self.n_lineages = len(set(self.m2l.values()))

    def resolve_id(self, ident):
        """Stash label -> HuggingFace id, through the map. Never by string."""
        if ident in self.m2l:
            return ident
        n = norm(ident)
        for mid in self.m2l:
            if norm(mid) == n:
                return mid
        return None

    def of(self, ident):
        """Returns the lineage or None. None is REPORTED, never silently dropped."""
        if ident in self.m2l:
            return self.m2l[ident]
        return self.norm2l.get(norm(ident))


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — COVERAGE. charter §3. Arm-agnostic: reads KEYS, not values.
# ─────────────────────────────────────────────────────────────────────────────

def gate_1_coverage(stash, lin, cats):
    """Enumerate the population and NAME every gap. charter §3.

    Reads key metadata only. No beam value is opened here, so this stage is
    computable under the freeze and its output is what justifies lifting it.
    """
    cells = collections.defaultdict(set)     # (lineage, category) -> {roles}
    unmapped_models, uncategorised_prompts, no_source = set(), set(), 0
    seen_sources, seen_models = set(), set()
    n_keys, off_category = 0, 0

    for k in stash:
        if not isinstance(k, dict) or k.get("type") != "beam_cross_v1":
            continue
        n_keys += 1
        src, mod, pr = k.get("source"), k.get("model"), k.get("prompt")
        if src is None:
            no_source += 1
            continue
        seen_sources.add(src)
        seen_models.add(mod)
        l_src, l_mod = lin.of(src), lin.of(mod)
        for ident, l in ((src, l_src), (mod, l_mod)):
            if l is None:
                unmapped_models.add(str(ident))
        cat = cats.get(pr)
        if cat is None:
            uncategorised_prompts.add(str(pr))
            continue
        #: [2357] -- filter to the NINE, never enumerate whatever the field holds
        if cat not in BATTERY_CATEGORIES:
            off_category += 1
            continue
        if l_src is not None:
            cells[(l_src, cat)].add("source")
        if l_mod is not None:
            cells[(l_mod, cat)].add("judge")

    #: D2 ([2286]) -- these three guards were defined and self-tested and had
    #: ZERO call sites, so cases 1, 2 and 4 certified library code rather than
    #: the run. The day's own shape at one remove: a check that passes because
    #: it was pointed at something other than the operation.
    assert_nonempty(list(cells), "gate_1 cell scan",
                    "type == beam_cross_v1 and source is not None")
    src_lins = {lin.of(s_) for s_ in seen_sources if lin.of(s_) is not None}
    mod_lins = {lin.of(m_) for m_ in seen_models if lin.of(m_) is not None}
    assert_commensurable(src_lins, mod_lins,
                         "source lineages vs judge lineages")
    assert_lineage_unit(seen_models, lin, "gate_1 judge roster")

    lineages = sorted({l for l, _ in cells})
    per_cat = collections.Counter(c for (_, c) in cells)

    print(f"STAGE 1 — COVERAGE (charter §3), from KEYS only, no value opened")
    print(f"  beam_cross_v1 keys           {n_keys:>7,}")
    print(f"  lineages present             {len(lineages):>7}   of "
          f"{lin.n_lineages} in the map")
    print(f"  categories with >=1 lineage  {len(per_cat):>7}")
    print()
    print("  GAPS, ENUMERATED AND NAMED (charter §3 — never silently dropped):")
    print(f"    keys with no `source`      {no_source:>7,}")
    print(f"    identifiers off the map    {len(unmapped_models):>7}  "
          f"{sorted(unmapped_models)[:4]}")
    print(f"    uncategorised prompts      {len(uncategorised_prompts):>7}")
    print(f"    outside the nine           {off_category:>7}  (categorisation "
          f"values that are not battery categories -- routed, not counted)")
    print()
    for c, n in sorted(per_cat.items(), key=lambda x: -x[1]):
        print(f"    {c:22s} {n:3d} lineages")

    return {"lineages": lineages, "cells": cells, "per_cat": per_cat,
            "gaps": {"no_source": no_source,
                     "unmapped": sorted(unmapped_models),
                     "uncategorised": sorted(uncategorised_prompts)}}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — THE DIVERGENCE CONTROL. charter §4.2a. THIS GATE CAN FAIL.
# ─────────────────────────────────────────────────────────────────────────────

def _slope(ys):
    """OLS slope of y on position index. Sign is what the gate reads."""
    n = len(ys)
    if n < 3:
        return None
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


def gate_2_divergence(raw_curves):
    """RETIRED at [2390]. This read `base_token_probs` -- the SOURCE scoring
    ITSELF, judge-invariant -- so its "cross" and "native" inputs were the same
    vector and its +0.0209 was source self-confidence rising with context, not
    a control result ([2385] struck it). Kept as a named tombstone rather than
    deleted, because a removed check and a discharged one read identically.
    Case 6 still exercises it; gate_2_divergence_rederived is the live control.
    """
    slopes = [s_ for s_ in (_slope(c) for c in raw_curves) if s_ is not None]
    if not slopes:
        return False
    return st.median(slopes) < 0.0 and (
        sum(1 for x in slopes if x < 0.0) / len(slopes)) > 0.5


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — RE-POWERING. charter §6. REQUIRED FIRST PRINT, not optional.
# ─────────────────────────────────────────────────────────────────────────────

def gate_3_repower(diffs_by_lineage):
    """Realized paired-difference sd at the LINEAGE unit, from first cells.

    charter §6: the 0.077 prior is MEASURED BUT FOREIGN (between-PROMPT, on a
    corpus M04 does not use). F28's 11.8x family-over-content ratio argues this
    will BITE rather than confirm. [325]: an assumed 0.02 measured at 0.15-0.19.
    """
    vals = [v for v in diffs_by_lineage.values() if v is not None]
    n = len(vals)
    print(f"\nSTAGE 3 — RE-POWERING FROM FIRST CELLS (charter §6, REQUIRED)")
    if n < 3:
        print(f"  HALT: {n} lineages with a usable difference. Cannot re-power.")
        return None
    sd = st.stdev(vals)
    #: D4 ([2286]) -- this was `.get(n - 1, 2.90)`, a magic default that would
    #: quietly mis-power any n outside three values, in a file whose docstring
    #: says it REFUSES where the charter is silent. Stage 4 obeyed that; stage 3
    #: did not. Same class, opposite handling, one file.
    T_CRIT = {24: 2.9208, 27: 2.9057, 33: 2.8875}
    if (n - 1) not in T_CRIT:
        print(f"  HALT: no t critical value declared for n={n}. The charter")
        print(f"  names n = 25 / 28 / 34. REFUSING to interpolate one.")
        return None
    t_crit = T_CRIT[n - 1]
    mde = t_crit * sd / math.sqrt(n)
    print(f"  lineages with a difference   {n:>7}")
    print(f"  REALIZED paired-difference sd {sd:>7.4f}")
    print(f"  foreign prior ([2162])        {0.077:>7.4f}")
    print(f"  realized MDE at n={n}         {mde:>7.4f}")
    print(f"  coarse target                 {0.100:>7.4f}   "
          f"{'RESOLVED' if mde < 0.10 else '*** NOT RESOLVED ***'}")
    if mde >= 0.10:
        print("\n  The realized sd does NOT resolve the coarse target. Per")
        print("  charter §6 this is the re-powering biting, which F28's prior")
        print("  predicted. The claim is reported against THIS MDE or not made.")
    return {"n": n, "sd": sd, "mde": mde}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — CONTRASTS. charter §5. Reachable only if 1-3 pass.
# ─────────────────────────────────────────────────────────────────────────────

def binom_one_sided(k, n):
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def gate_4_contrasts(asym_by_lineage, depth_pos0, depth_curve):
    """The primary sign test, and the depth profile with pos0 SEPARATE.

    charter §5 primary: aligned-substitution damage exceeds base-substitution
    damage under a common judge, sign test across lineages, one-sided.
    charter §4.2: pos0 reported separately; the branches read pos1..10.
    """
    vals = [v for v in asym_by_lineage.values() if v is not None]
    n = len(vals)
    pos = sum(1 for v in vals if v > 0)
    p = binom_one_sided(pos, n)
    need = CLEARS.get(n)

    print(f"\nSTAGE 4 — CONTRASTS (charter §5)")
    print(f"  PRIMARY: asymmetry, sign test at the LINEAGE unit")
    print(f"    n                          {n:>7}")
    print(f"    positive                   {pos:>7}")
    print(f"    one-sided p                {p:>7.4f}")
    if need:
        print(f"    charter threshold          {need:>4} of {n}   "
              f"{'CLEARS' if pos >= need else 'does NOT clear'}")
    else:
        print(f"    *** n={n} has no charter threshold. The charter names 25/28/34.")
        print(f"    *** REFUSING to interpolate one. Report n and stop.")

    print(f"\n  DEPTH PROFILE (charter §4.2) — pos0 SEPARATE, branches read pos1..9")
    print(f"    pos0 (reported, NOT in the curve)  {depth_pos0:>+8.4f}")
    for i, v in zip(CURVE, depth_curve):
        print(f"    pos{i:<2d}                            {v:>+8.4f}")
    s = _slope(depth_curve)
    print(f"    slope over pos1..9                 {s:>+8.4f}" if s is not None
          else "    slope: not computable")
    print("\n    BRANCH (charter §4.2, read from pos1 onward):")
    if s is None:
        branch = "indeterminate"
    elif s < -0.002:
        branch = "PRESENT and FALLING -> ten sufficient, ceiling not binding"
    elif s > 0.002:
        branch = ("PRESENT and RISING -> AMBIGUOUS until §4.2a's control is "
                  "read; ceiling-is-binding NOT licensed")
    else:
        branch = "PLATEAUED -> ten sufficient, ceiling not binding"
    print(f"      {branch}")

    return {"n": n, "positive": pos, "p": p, "clears": need, "branch": branch,
            "depth_rows": [("pos0_SEPARATE", depth_pos0)] + list(zip(CURVE, depth_curve)),
            "contrast_rows": sorted(asym_by_lineage.items())}



# ─────────────────────────────────────────────────────────────────────────────
# THE PIPELINE. [2298]: the gates were all DEFINED AND NONE WAS CALLED, and the
# entire verification apparatus passed anyway because it was pointed at the
# self-test. main() now runs them, each halting the next.
# ─────────────────────────────────────────────────────────────────────────────

#: behavioural sentinel -- gate 4 records that it EXECUTED, so the freeze
#: boundary is testable by RUNNING the pipeline rather than by grepping source.
#: Case 7 grepped, and grepping is what let it go vacuous the first time.
_REACHED = {"gate_4": False}


def write_csv(root, name, header, rows):
    """[2300].4: the literal output constants were DECLARED AND NEVER WRITTEN,
    and no case asserted otherwise. `root` lets the self-test run the pipeline
    into a temp dir; the BASENAME stays literal so `trace_producers.py` sees it.
    """
    path = os.path.join(root, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import csv as _csv
    with open(path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def collect_rescored_cells(stash, lin, rescored=None):
    """(source_log, judge_log, resist) triples from the CORRECTED re-score.

    RULING 2429-both. Returns None while the re-score has not been produced --
    a declared absence, never a fallback to the stored field ([2474]) or to the
    retired gate ([2390]). The GPU dependency is real and is reported as one.

    `rescored` is an INJECTION POINT FOR FIXTURES ONLY and defaults to None, so
    production halts at stage 2. It is not a fallback: nothing in this function
    fabricates triples, and no caller in production passes it.
    """
    return rescored


def run_pipeline(stash, lin, cats, root=ROOT, rescored=None):
    """Gates 1 -> 2 -> 3 -> 4. Each returns falsy to HALT the ones after it.

    Returns the stage at which it stopped, or 4 if it completed. No stage is
    skippable and there is no flag that reorders them.
    """
    cov = gate_1_coverage(stash, lin, cats)
    if not cov:
        return 1
    write_csv(root, OUT_COVERAGE, ["lineage", "category", "roles"],
              [(l, c, "|".join(sorted(r))) for (l, c), r in sorted(cov["cells"].items())])

    #: [2494] -- run_pipeline called `gate_2_divergence`, THE RETIRED
    #: TOMBSTONE ([2390]), while `gate_2_divergence_rederived` -- the control
    #: registrar approved at [2391] -- was unreachable from main(). Case 8's
    #: own `gates` set named the retired one, SO THE SELF-TEST CERTIFIED THE
    #: TOMBSTONE AS WIRED. A retired function that still runs is worse than a
    #: deleted one: it has a name, a docstring saying it is retired, and a
    #: caller.
    cells = collect_rescored_cells(stash, lin, rescored)
    if cells is None:
        print("  STAGE 2 HALT: the rederived control ([2391]) needs the "
              "re-scored (source_log, judge_log, resist) triples, which "
              "require the corrected scorer's GPU pass. The RETIRED gate is "
              "not a fallback and is not called.")
        return 2
    if not gate_2_divergence_rederived(cells):
        return 2                       # charter §4.2a -- instrument NOT VALIDATED

    diffs = collect_paired_differences(stash, lin)
    pw = gate_3_repower(diffs)
    if pw is None:
        return 3

    asym, pos0, curve = collect_contrasts(stash, lin)
    _REACHED["gate_4"] = True
    #: A ([2310].2) -- run_pipeline used to write all three CSVs from the
    #: COLLECTOR output while gate_4 only printed, so emptying the stage that
    #: computes THE PRIMARY STATISTIC left every file intact and case 9 passed.
    #: The stage that computes now OWNS its rows; the pipeline writes what the
    #: gate RETURNS, so an empty gate 4 writes nothing.
    result = gate_4_contrasts(asym, pos0, curve)
    if not result:
        return 4
    write_csv(root, OUT_DEPTH, ["position", "effect"], result["depth_rows"])
    write_csv(root, OUT_CONTRASTS, ["lineage", "asymmetry"], result["contrast_rows"])
    return 4


#: SWEEP STATUS ([2337], replacing the warning deleted at `da752b00`): ALL THREE
#: collectors are now covered -- 12 (paired_differences), 13 (contrasts),
#: 14 (raw_curves). The old note said the sweep was blind to them; that is no
#: longer true, and it is recorded as RESOLVED rather than removed, because the
#: sentence was deleted in the same edit that left one third of it still live.
#:
#: THE COLLECTORS. Written to the schema malign published at [2257] --
#: value = list of beams; beam.base_token_probs = list of 10 per-position
#: teacher-forced probabilities -- NOT to a schema this seat has opened. The
#: beam stash remains unread here; that is the audit's first target.
#:
#: THE PAIRING RULE, DECLARED BECAUSE THE CHARTER SPECIFIES THE CONSTRUCT AND
#: NOT THE STASH-LEVEL JOIN. charter §4.1 wants each arm's substitutions scored
#: under EVERY judge. In the stash a key is (source, model, prompt): `source`
#: produced the storyline, `model` scored it, and base_token_probs is the
#: JUDGE's per-position probability of the SOURCE's tokens. Therefore:
#:
#:     NATIVE      source == model   -- the judge scoring its own path
#:     CROSS       source != model   -- the judge scoring another arm's path
#:
#: and the §4.2a cancelling difference is CROSS minus NATIVE **under one judge
#: and one prompt**, which is what makes generic per-position divergence cancel:
#: both paths are equally non-native to that judge in every respect except whose
#: substitutions they carry.
#:
#: THREE DECISIONS I HAD TO MAKE AND AM NAMING RATHER THAN BURYING:
#:   (a) a cell's value is the MEAN over its beams, not the top beam -- the top
#:       beam is a mode and the charter's quantity is a distributional one;
#:   (b) cells with no NATIVE counterpart under the same judge+prompt are
#:       DROPPED and COUNTED, never imputed;
#:   (c) lineage aggregation is the MEAN over that lineage's cells, so a lineage
#:       with more cells does not vote more -- the unit is the lineage (§3).
class StoredFieldUnusable(Exception):
    """`base_token_probs` cannot be read as a per-position quantity. [2474]"""


#: DROP LEDGER -- module-level so the silent filter became a counted one.
#: [2474]: the old body did `if len(v) != 10: continue` with NO counter, and
#: that dropped 411,939 of 1,015,100 beams (40.6%). 400,372 of them were
#: length 11 -- the join defect's exact signature, one extra position from the
#: spurious ' ' token. The retained set was therefore SELECTED, not sampled.
BEAM_DROPS = collections.Counter()


def _beam_positions(beams, allow_stored=False):
    """Mean per-position probability over a cell's beams.

    REFUSES BY DEFAULT. `base_token_probs` is the source scored under the CALL
    BASE's tokenizer, and [2465] ruled the call base UNRECOVERABLE for existing
    rows -- so the field is un-interpretable per-position no matter its length.

    And the length filter that used to stand here was worse than the field:
    retention ran 0.4%-100% by source and was ARM-ASYMMETRIC in the direction
    of M04's own primary (Llama-3.1: base 89.8% kept, aligned 10.7%; median gap
    47.4 points over 13 lineages; base retained in 8 of 8 large gaps). Aligned
    models emit space-initial continuations more often, the join makes those 11
    tokens, and 11 was dropped. **A filter conditioning on a variable correlated
    with the arm, invisibly, would have produced a well-formed asymmetry from
    nothing.**

    RULING 2429-both -- both sides re-scored over one segmentation.

    Per-position values must come from the corrected scorer's re-score
    (`score_stored`), whose span is exactly len(stored_token_ids) by
    construction -- so the length cannot vary and the selection cannot occur.

    `allow_stored=True` exists ONLY for reproducing the legacy numbers, and it
    COUNTS what it drops into BEAM_DROPS rather than discarding it in silence.
    """
    if not allow_stored:
        raise StoredFieldUnusable(
            "`base_token_probs` is not readable as a per-position quantity: "
            "the call base is unrecoverable ([2465]) and the length filter it "
            "requires is arm-asymmetric ([2474]). Use the corrected scorer's "
            "re-score. Pass allow_stored=True only to reproduce legacy numbers.")

    cols = [[] for _ in range(10)]
    for b in beams:
        v = b.get("base_token_probs") if isinstance(b, dict) else None
        if not v:
            BEAM_DROPS["absent"] += 1
            continue
        #: RULING 2401-retain -- retain n >= 10, do not require n == 10.
        #: LEGACY PATH ONLY. The re-score (score_stored) yields exactly
        #: len(stored_ids) positions, so this repair is moot for it.
        #:
        #: The premise, MEASURED not assumed: max_tokens == 10 on all 10,166
        #: keys and len(tokens) == 10 on every beam sampled. THE GENERATION IS
        #: INVARIABLY TEN. So a base_token_probs of another length is a scoring
        #: artifact, never a different beam -- and the two directions differ:
        #:
        #:   n > 10  the join PREPENDED positions (first divergence at index 0
        #:           in 66/72 beams; the spurious ' ' takes position 0), so the
        #:           last ten align with the recorded tokens. FRONT-TRUNCATE.
        #:   n < 10  the span was SHORTER than the truth. Nothing to truncate
        #:           and no way to know which positions are missing. DROP.
        #: RULING 2408-offset -- and this is where my [2401] implementation was
        #: WRONG. I front-truncated every n > 10, which retains K=2 and K=3
        #: cells (lengths 12 and 13). [2408] ruled those RESEGMENTED: no
        #: alignment exists, drop and count. `relabel`/`cell_offset` already
        #: encoded that rule and sat UNREACHABLE, so the reachability finding
        #: ([2494]) did not surface dead decoration -- it surfaced a ruling my
        #: replacement contradicted. Deleting them unread would have deleted
        #: the rule. This routes through them so K lives in ONE place.
        #: cell_offset returns None for BOTH K<0 and K>=2, so the ledger
        #: computes K itself to name which disposition applied. A drop counted
        #: under the wrong reason is a rate whose population is misdescribed.
        _k = len(v) - 10
        _v = relabel(v, [None] * 10)
        if _v is None:
            BEAM_DROPS["short" if _k < 0 else "resegmented"] += 1
            BEAM_DROPS[f"{'short' if _k < 0 else 'resegmented'}_{len(v)}"] += 1
            continue                                  #: RULING 2474-drops
        BEAM_DROPS["kept" if _k == 0 else "relabelled_K1"] += 1
        v = _v
        for i, x in enumerate(v):
            try:
                cols[i].append(float(x))
            except (TypeError, ValueError):
                pass
    if any(not c for c in cols):
        return None
    return [sum(c) / len(c) for c in cols]


class VocabIndex:
    """Lazy model-id -> tokenizer, so [2431] can be ENFORCED and not merely
    written down. Loads on demand and caches; a model whose tokenizer will not
    load is REPORTED as unavailable, never treated as compatible.
    """

    def __init__(self, lin):
        self.lin, self._cache = lin, {}

    def __call__(self, ident):
        if ident in self._cache:
            return self._cache[ident]
        tok = None
        try:
            from transformers import AutoTokenizer
            mid = ident if "/" in str(ident) else self.lin.resolve_id(ident)
            if mid:
                tok = AutoTokenizer.from_pretrained(mid, local_files_only=True,
                                                    trust_remote_code=True)
        except Exception:                                      # noqa: BLE001
            tok = None
        self._cache[ident] = tok
        return tok


def pair_is_scorable(source_ident, judge_ident, tokenizer_of):
    """RULING 2431-vocab -- may this (source, judge) cell be scored at all?

    ROUTED THROUGH `corrected_scorer`, not reimplemented here. [2481] ruled the
    preference and gave the reason from tonight's own ledger ([2468].5): a guard
    duplicated into a new function inherits NONE of the original's coverage. One
    rule, one home, every path through it.

    Returns (True, None) or (False, reason). Never raises for an ordinary
    cross-vocabulary pair -- that is a routine exclusion here, counted on the
    face, not an error.
    """
    import corrected_scorer as CS

    try:
        ts, tj = tokenizer_of(source_ident), tokenizer_of(judge_ident)
    except Exception as e:                                     # noqa: BLE001
        return False, f"tokenizer_unavailable:{type(e).__name__}"
    if ts is None or tj is None:
        return False, "tokenizer_unavailable"
    if not CS.same_vocabulary(ts, tj):
        return False, "cross_vocabulary"
    return True, None


#: RULING 2417-void -- the three void pairs ruled out at [2417]. Named here so
#: the exclusion lives in the artifact and not only on the docket; counted onto
#: the face by _cells rather than filtered in silence.
VOID_PAIRS = frozenset({
    "DeepSeek_R1_Distill_",
    "Llama_3.1_Tulu_3_8B_",
    "eleuther_pythia6.9b_",
})


#: [2498] -- `scorable=None` meant the [2431] filter was a NO-OP at every call
#: site: the hook existed, the marker sat on it, and nobody passed it. AN
#: OPTIONAL PARAMETER IS A SWITCH THAT DEFAULTS TO OFF. Enforcement is now the
#: DEFAULT and the parameter exists only to SUBSTITUTE an implementation, never
#: to disable one -- a fixture that does not want the vocabulary rule must say
#: so in its own call, visibly, in the diff.
_ENFORCE = object()


def _cells(stash, lin, allow_stored=False, scorable=_ENFORCE):
    """(judge_lineage, prompt, native?) -> per-position mean. Schema-level read."""
    if scorable is _ENFORCE:
        _vi = VocabIndex(lin)
        def scorable(a, b, _vi=_vi):          # noqa: E306 -- RULING 2431-vocab
            return pair_is_scorable(a, b, _vi)

    out, dropped = {}, 0
    for k in stash:
        if not isinstance(k, dict) or k.get("type") != "beam_cross_v1":
            continue
        src, mod, pr = k.get("source"), k.get("model"), k.get("prompt")
        if src is None or mod is None:
            dropped += 1
            continue
        #: RULING 2417-void -- excluded and COUNTED, never silently skipped.
        if src in VOID_PAIRS or mod in VOID_PAIRS:
            BEAM_DROPS["void_pair"] += 1
            continue
        #: RULING 2431-vocab -- ENFORCED HERE, not merely defined. [2494]:
        #: this function existed, was marked, and had its mutant caught, while
        #: main() never reached it -- "implemented in the same sense a
        #: docstring is". Conformance saw the marker; reachability saw the
        #: truth. The call site is what makes the ruling real.
        if scorable is not None:              #: RULING 2431-vocab, DEFAULT ON
            _ok, _why = scorable(src, mod)
            if not _ok:
                BEAM_DROPS[f"unscorable_{_why}"] += 1
                continue
        lj = lin.of(mod)
        if lj is None:
            dropped += 1
            continue
        #: [2474] -- NOT opted in by default. The production path REFUSES;
        #: only a caller that has declared legacy reproduction gets through,
        #: and it gets the drop ledger with it.
        pos = _beam_positions(stash[k], allow_stored=allow_stored)
        if pos is None:
            dropped += 1
            continue
        native = lin.of(src) == lj
        out.setdefault((lj, pr, native), []).append(pos)

    #: [2474].4.3 -- a silent `continue` in a population instrument IS the
    #: defect, independent of the threshold. The ledger is returned so the
    #: caller cannot fail to have it; 40.6% of beams were dropped here
    #: unreported, arm-asymmetrically, until this was added.
    kept = BEAM_DROPS.get("kept", 0)
    lost = sum(v for k_, v in BEAM_DROPS.items() if k_ != "kept")
    if kept and lost:
        pct = 100.0 * lost / (kept + lost)
        print(f"  BEAM DROP LEDGER: kept {kept:,}  dropped {lost:,} ({pct:.1f}%)")
        for k_, v in sorted(BEAM_DROPS.items(), key=lambda x: -x[1]):
            if k_ != "kept":
                print(f"    {k_:12s} {v:>9,}")
        print("    NOTE [2474]: this drop is ARM-ASYMMETRIC (base retained, "
              "aligned dropped) and must not be read as a contrast.")
    return out, dropped


#: TOMBSTONE ([2510].4). This fed `gate_2_divergence`, retired at [2390]; when
#: the pipeline moved to the rederived control it was orphaned. Read before
#: burial per [2497]: it encodes NO ruling the rederived path lacks -- it
#: collected UNDIFFERENCED raw curves, which is precisely the quantity §4.2a
#: says must never be read on its own. Retained, declared, and asserted
#: unreachable from run_pipeline by case 23.
def collect_raw_curves(stash, lin, allow_stored=False, scorable=_ENFORCE):
    """RETIRED. Not called; not part of any live stage. [2510].4, [2514].3.

    It fed `gate_2_divergence`, retired at [2390], and was orphaned when the
    pipeline moved to `gate_2_divergence_rederived`. Case 23 asserts it stays
    unreachable from `run_pipeline`.

    Its former role: collect the RAW UNDIFFERENCED forced-path curves as
    §4.2a's positive-check input. That quantity is the one §4.2a forbids
    reading on its own, so this is retired rather than pending -- there is no
    version of the control that wants it back.

    RETAINED, not deleted, per [2497]: unreachable code is read before burial
    in case it encodes a ruling. It encodes none the rederived path lacks.
    """
    cells, _ = _cells(stash, lin, allow_stored=allow_stored,
                      scorable=scorable)
    assert_nonempty(list(cells), "collect_raw_curves", "beam_cross_v1 with 10-long probs")
    return [[sum(c[i] for c in curves) / len(curves) for i in range(10)]
            for curves in cells.values()]


def collect_paired_differences(stash, lin, allow_stored=False, scorable=_ENFORCE):
    """Per-lineage CROSS-minus-NATIVE difference. charter §6's re-power input.

    (b): a judge+prompt with no native counterpart is dropped and counted.
    (c): the lineage's value is the mean over its surviving cells.
    """
    cells, _ = _cells(stash, lin, allow_stored=allow_stored,
                      scorable=scorable)
    per_lin, no_native = {}, 0
    for (lj, pr, native), curves in cells.items():
        if native:
            continue
        nat = cells.get((lj, pr, True))
        if not nat:
            no_native += 1
            continue
        cross_m = sum(sum(c[i] for i in CURVE) for c in curves) / (len(curves) * len(CURVE))
        nat_m = sum(sum(c[i] for i in CURVE) for c in nat) / (len(nat) * len(CURVE))
        per_lin.setdefault(lj, []).append(cross_m - nat_m)
    if no_native:
        print(f"  [collectors] {no_native} cross cells DROPPED: no native "
              f"counterpart under the same judge+prompt (declared (b), never imputed)")
    assert_nonempty(list(per_lin), "collect_paired_differences", "cross cells with a native pair")
    return {l: sum(v) / len(v) for l, v in per_lin.items()}


def collect_contrasts(stash, lin, allow_stored=False, scorable=_ENFORCE):
    """(asymmetry by lineage, pos0, pos1..9 curve). charter §5 and §4.2.

    pos0 is returned SEPARATELY and is not part of the curve (§4.2).
    """
    cells, _ = _cells(stash, lin, allow_stored=allow_stored,
                      scorable=scorable)
    asym = collect_paired_differences(stash, lin, allow_stored=allow_stored,
                                     scorable=scorable)
    diffs = []
    for (lj, pr, native), curves in cells.items():
        if native:
            continue
        nat = cells.get((lj, pr, True))
        if not nat:
            continue
        cm = [sum(c[i] for c in curves) / len(curves) for i in range(10)]
        nm = [sum(c[i] for c in nat) / len(nat) for i in range(10)]
        diffs.append([cm[i] - nm[i] for i in range(10)])
    assert_nonempty(diffs, "collect_contrasts", "cross cells with a native pair")
    mean_pos = [sum(d[i] for d in diffs) / len(diffs) for i in range(10)]
    return asym, mean_pos[POS0], [mean_pos[i] for i in CURVE]


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST — [2282]'s seven known-answer cases, ONE PER FAILURE MODE.
# Runs with NO stash access, so the machinery is debuggable without spending
# the blindness ([2282].7, "the one I would put first").
# ─────────────────────────────────────────────────────────────────────────────

def assert_commensurable(a, b, what):
    """[2282].1 — a cross-set comparison over disjoint namespaces must RAISE.

    Beam `source` is `Llama_3_1_8B`; `model` is `meta-llama/Llama-3.1-8B`.
    A string comparison returns 0 with no error and no warning.
    """
    if not (set(a) & set(b)):
        raise ValueError(
            f"COMMENSURABILITY: {what} share no members. Two ID namespaces "
            f"compared directly return a clean, confident FALSE ZERO ([2251].3). "
            f"Resolve through the lineage map first.")


def assert_lineage_unit(idents, lin, what):
    """[2282].2 — n is lineages. If it equals the label count, the ladders did
    not collapse and the unit is wrong (59 -> 39 -> 34, all toward significance).
    """
    labels = {str(i) for i in idents}
    lins = {lin.of(i) for i in idents if lin.of(i) is not None}
    if len(lins) == len(labels) and len(labels) > 1:
        raise ValueError(
            f"UNIT: {what} resolves {len(labels)} identifiers to {len(lins)} "
            f"lineages -- nothing collapsed. A roster with a size ladder MUST "
            f"shrink. Falcon3's four sizes are ONE unit.")
    return sorted(lins)


def assert_nonempty(rows, what, predicate_desc):
    """[2282].4 — zero rows by filter must be distinguishable from never-ran."""
    if not rows:
        raise ValueError(
            f"EMPTY-BY-FILTER: {what} returned 0 rows under `{predicate_desc}`. "
            f"This is reported, not silently aggregated -- an operation that "
            f"completes without doing what it names looks identical to one "
            f"that did.")


class _StubLin:
    """Minimal resolver for case 7's fixture. Not used outside the self-test."""
    n_lineages = 1

    def of(self, ident):
        return "LIN_A"


class _FixtureStash:
    """Falcon3's two sizes (ONE lineage, so the roster collapses) plus Olmo,
    with both directions present so source and judge lineages overlap."""

    def __iter__(self):
        return iter([
            {"type": "beam_cross_v1", "source": "Falcon3_1B", "model": "Falcon3_7B",
             "prompt": "p"},
            {"type": "beam_cross_v1", "source": "Falcon3_7B", "model": "Falcon3_1B",
             "prompt": "p"},
            {"type": "beam_cross_v1", "source": "Olmo_7B", "model": "Falcon3_1B",
             "prompt": "q"},
        ])


class _EmptyStash:
    """No usable keys -> gate 1's guards must halt the pipeline."""

    def __iter__(self):
        return iter([])


class _FixtureLin:
    n_lineages = 2

    def of(self, i):
        return "FALCON3" if str(i).startswith("Falcon3") else "OLMO"


def selftest(verbose=True):
    """All seven, no stash. Failure here means the producer does not run."""
    ok = []

    # 1. COMMENSURABILITY
    try:
        assert_commensurable(["Llama_3_1_8B"], ["meta-llama/Llama-3.1-8B"], "case1")
        ok.append(("1 commensurability", False, "did NOT raise on disjoint namespaces"))
    except ValueError:
        ok.append(("1 commensurability", True, "raised on disjoint namespaces"))

    # 2. UNIT — a Falcon3 ladder must collapse to one, AND a roster that does
    #    NOT collapse must RAISE.
    #
    #    M6 ([2290]) -- this checked only the RETURN VALUE, so a guard whose
    #    body was replaced by a bare resolve (no check at all) still passed:
    #    the case was testing `_FakeLin.of()`, a stub written inside the test,
    #    and reporting it as evidence about the assertion. Cases 1 and 4 use
    #    try/raise/except and caught their mutants; this one did not, and it is
    #    D2's own shape one layer down.
    class _FakeLin:
        n_lineages = 1
        def of(self, i):
            return "tiiuae/Falcon3-10B-Base" if "Falcon3" in str(i) else None

    class _NoCollapse:
        n_lineages = 3
        def of(self, i):
            return "LIN_" + str(i)

    ladder = ["tiiuae/Falcon3-1B-Base", "tiiuae/Falcon3-3B-Base",
              "tiiuae/Falcon3-7B-Base", "tiiuae/Falcon3-10B-Base"]
    lins = assert_lineage_unit(ladder, _FakeLin(), "case2 positive")
    collapses = len(lins) == 1

    raised = False
    try:
        assert_lineage_unit(["a", "b", "c"], _NoCollapse(), "case2 negative")
    except ValueError:
        raised = True                       # THE HALF THAT MAKES IT A TEST

    ok.append(("2 lineage unit", collapses and raised,
               f"ladder -> {len(lins)} lineage AND non-collapsing roster "
               f"{'RAISES' if raised else 'DOES NOT RAISE'}"))

    # 3. DEGENERATE PATTERN — this producer uses no matcher; assert that stays true
    src = open(__file__).read()
    has_matcher = bool(re.search(r"re\.(match|search|fullmatch)\(\s*[rf]?['\"]\.[+*]", src))
    ok.append(("3 degenerate pattern", not has_matcher,
               "no wildcard matcher in this file" if not has_matcher
               else "A WILDCARD MATCHER EXISTS -- reject or justify"))

    # 4. EMPTY-RESULT TRIPWIRE
    try:
        assert_nonempty([], "case4", "impossible predicate")
        ok.append(("4 empty tripwire", False, "did NOT raise on empty"))
    except ValueError:
        ok.append(("4 empty tripwire", True, "raised EMPTY-BY-FILTER"))

    # 5. POSITION-0 EXCLUSION, BY FIXTURE
    #    pos0 enormous, pos1..9 flat -> the curve must be FLAT.
    fixture = [9.99] + [0.20] * 9
    curve = [fixture[i] for i in CURVE]
    sl = _slope(curve)
    ok.append(("5 pos0 excluded", sl is not None and abs(sl) < 1e-9,
               f"spike 9.99 at pos0, flat pos1..9 -> slope {sl:+.6f}"))

    # 6. THE CONTROL MUST BE ABLE TO FAIL
    flat = [[0.5] * 10 for _ in range(20)]          # no divergence
    falling = [[0.9 - 0.05 * i for i in range(10)] for _ in range(20)]
    #: D3 ([2286]) -- `passes_on_falling` was bound only inside `if verbose`,
    #: so selftest(verbose=False) raised NameError. Unreachable today, which is
    #: exactly why it would have survived until someone added a quiet mode.
    #: A parameter whose only other value crashes is not a parameter.
    with contextlib.redirect_stdout(io.StringIO()):
        fails_on_flat = not gate_2_divergence(flat)
        passes_on_falling = gate_2_divergence(falling)
    ok.append(("6 control can fail", fails_on_flat and passes_on_falling,
               "HALTS on flat raw paths, PASSES on falling ones"))

    # 7. FREEZE BOUNDARY — stage 4 must be unreachable without 1-3
    #: 7 REDEFINED ([2300]), THEN COMPLETED ([2306]). The old form grepped for
    #: an ABSENCE and forbade the wiring. The first behavioural rewrite tested
    #: ONLY GATE 2's halt -- so a mutant that made gate 3's failure return 4
    #: (claiming success) survived, because the fixture never failed gate 3.
    #: EVERY halt path is tested now, not the one I had a fixture for.
    _saved7 = {k: globals()[k] for k in
               ("collect_raw_curves", "collect_paired_differences", "collect_contrasts")}
    halt_results = []
    for which, patch in (
        ("g1", {"stash": _EmptyStash}),
        ("g2", {"collect_raw_curves": lambda st, l: [[0.5] * 10 for _ in range(20)]}),
        ("g3", {"collect_paired_differences": lambda st, l: {"A": 0.1, "B": 0.2}}),
    ):
        _REACHED["gate_4"] = False
        globals()["collect_raw_curves"] = patch.get(
            "collect_raw_curves", lambda st, l: [[0.9 - 0.05 * i for i in range(10)]
                                                 for _ in range(20)])
        globals()["collect_paired_differences"] = patch.get(
            "collect_paired_differences",
            lambda st, l: {f"LIN_{i}": 0.1 for i in range(25)})
        globals()["collect_contrasts"] = lambda st, l: (
            {f"LIN_{i}": 0.1 for i in range(25)}, 2.24, [0.85] * 9)
        stash = patch["stash"]() if "stash" in patch else _FixtureStash()
        try:
            #: root=td ([2341]) -- these called run_pipeline WITHOUT a root, so
            #: it defaulted to ROOT and THE SELF-TEST WROTE FIXTURE DATA INTO THE
            #: REPO's data/ UNDER THE REAL OUTPUT NAMES. A reader opening
            #: data/m04_contrasts.csv would have read fixture numbers as results.
            #: No self-test writes outside a temp dir, ever.
            import tempfile as _tf
            with _tf.TemporaryDirectory() as _td, \
                 contextlib.redirect_stdout(io.StringIO()):
                stopped = run_pipeline(stash, _FixtureLin(),
                                       {"p": "neutral", "q": "death"}, root=_td)
        except Exception:
            stopped = "raised"        # gate 1 halts by RAISING; that is a halt
        halt_results.append((which, stopped, _REACHED["gate_4"]))
    globals().update(_saved7)

    all_halted = all(not reached and stopped != 4
                     for _, stopped, reached in halt_results)
    ok.append(("7 every gate halts", all_halted,
               "; ".join(f"{w}->{s}" for w, s, _ in halt_results)
               + "; gate 4 never executed"))

    #: 8 ([2300]) -- THE CASE WHOSE ABSENCE LET THE PIPELINE NOT EXIST. Every
    #: gate must be REACHABLE from main() through the call graph. Deleting two
    #: whole stages previously left the self-test at 7 of 7.
    import ast as _ast
    tree = _ast.parse(src)
    calls = {}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.FunctionDef):
            calls[node.name] = {c.func.id for c in _ast.walk(node)
                                if isinstance(c, _ast.Call)
                                and isinstance(c.func, _ast.Name)}
    seen, stack = set(), ["main"]
    while stack:
        f = stack.pop()
        for c in calls.get(f, ()):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    gates = {"gate_1_coverage", "gate_2_divergence_rederived",
             "gate_3_repower", "gate_4_contrasts"}
    #: mutation-found ([2302]): checking only `gates - seen` asserts the gate is
    #: CALLED, not that it EXISTS. Deleting gate_1_coverage entirely left its
    #: call site in run_pipeline, so this passed while the pipeline was a
    #: guaranteed NameError. A gate must be BOTH reachable AND defined.
    unreachable = sorted(gates - seen)
    undefined = sorted(gates - set(calls))

    #: mutation-found ([2306], M12): neutering gate_1's CALL to assert_nonempty
    #: left the other two guards to raise, so case 7 saw "something halted" and
    #: passed. Cases 1/2/4 test the guard FUNCTIONS; nothing tested that each is
    #: CALLED FROM gate_1. Coarser test, finer defect -- the day's shape again.
    guards = {"assert_nonempty", "assert_commensurable", "assert_lineage_unit"}
    unwired = sorted(guards - calls.get("gate_1_coverage", set()))

    bad = bool(unreachable or undefined or unwired)
    ok.append(("8 gates+guards wired", not bad,
               "4 gates defined+reachable from main(); 3 guards called in gate_1"
               if not bad else
               f"unreachable={unreachable} undefined={undefined} unwired={unwired}"))

    #: 9 ([2300].4) -- THE CHEAPEST CASE ON THE LIST AND NOBODY WROTE IT:
    #: run the thing and assert it produced its outputs. Three literal output
    #: constants were declared -- on the auditor's own non-test item -- and
    #: NOTHING WROTE THEM and no case noticed. Seven cases, six mutants, a
    #: deletion test, two audits and a clearance, and nothing asked whether the
    #: program does anything.
    import tempfile
    _REACHED["gate_4"] = False

    #: the fixture must be REALISTIC ENOUGH TO PASS THE GUARDS, and building it
    #: is itself a check: the first version had disjoint source/judge lineages
    #: and nothing collapsing, so assert_commensurable fired. A fixture that
    #: trips a real guard is the guard working -- not a reason to weaken it.

    _saved = {k: globals()[k] for k in
              ("collect_raw_curves", "collect_paired_differences", "collect_contrasts")}
    globals()["collect_raw_curves"] = lambda st, l: [
        [0.9 - 0.05 * i for i in range(10)] for _ in range(20)]
    globals()["collect_paired_differences"] = lambda st, l: {
        f"LIN_{i}": 0.10 + 0.01 * (i % 3) for i in range(25)}
    globals()["collect_contrasts"] = lambda st, l: (
        {f"LIN_{i}": 0.1 for i in range(25)}, 2.24, [0.85] * 9)
    written, content_ok = [], False
    try:
        with tempfile.TemporaryDirectory() as td:
            with contextlib.redirect_stdout(io.StringIO()):
                #: [2494] -- the fixture SUPPLIES the re-scored triples the
                #: live control needs. Production passes nothing and halts at
                #: stage 2; making this pass by having the collector fabricate
                #: data would be [2488].2's weakening in a new place.
                _fake_rescored = [([0.9 - 0.05 * i for i in range(10)],
                                   [0.8 - 0.06 * i for i in range(10)],
                                   [0.1 + 0.00 * i for i in range(10)])
                                  for _ in range(6)]
                stage = run_pipeline(_FixtureStash(), _FixtureLin(),
                                     {"p": "neutral", "q": "death"}, root=td,
                                     rescored=_fake_rescored)
            written = [n for n in (OUT_COVERAGE, OUT_DEPTH, OUT_CONTRASTS)
                       if os.path.exists(os.path.join(td, n))]
            #: A ([2310].2) -- existence is not enough: assert the CONTENT the
            #: analysis produced. 25 fixture lineages -> 25 contrast rows;
            #: pos0 + 9 curve positions -> 10 depth rows.
            import csv as _c
            try:
                cr = list(_c.reader(open(os.path.join(td, OUT_CONTRASTS))))[1:]
                dr = list(_c.reader(open(os.path.join(td, OUT_DEPTH))))[1:]
                content_ok = len(cr) == 25 and len(dr) == 10 and dr[0][0] == "pos0_SEPARATE"
            except Exception:
                content_ok = False
    except Exception as e:
        stage = f"raised {type(e).__name__}"
    finally:
        globals().update(_saved)

    ok.append(("9 produces outputs", stage == 4 and len(written) == 3 and content_ok,
               f"stage {stage}, {len(written)}/3 files, content "
               f"{'matches the analysis' if content_ok else 'WRONG'}"))

    #: 16 ([2474]) -- THE PRODUCTION PATH MUST REFUSE THE STORED FIELD. The
    #: default `allow_stored=False` is the guard; the legacy cases opt in
    #: explicitly. Without this case, threading the flag through and then
    #: flipping any default back to True would be silent -- and I nearly
    #: shipped exactly that, having set allow_stored=True inside _cells to
    #: make four failing cases green. A FIXTURE THAT TRIPS A REAL GUARD IS THE
    #: GUARD WORKING; rebuild the fixture, never weaken the guard.
    import inspect as _insp
    _defaults_ok = all(
        _insp.signature(f).parameters["allow_stored"].default is False
        for f in (_beam_positions, _cells, collect_raw_curves,
                  collect_paired_differences, collect_contrasts))
    try:
        _beam_positions([{"base_token_probs": [0.5] * 10}])
        _refused = False
    except StoredFieldUnusable:
        _refused = True
    except Exception:
        _refused = False
    ok.append(("16 production refuses stored field", _defaults_ok and _refused,
               f"defaults all False={_defaults_ok}, refuses by default={_refused} "
               f"(the arm-asymmetric read cannot be reached without declaring it)"))

    #: 17 ([2474].4.3) -- THE DROP MUST BE COUNTED. Removing the counter and
    #: restoring the silent `continue` survived mutation, because every other
    #: case cares what the kept beams average to and none cares what became of
    #: the rest. A SILENT `continue` IN A POPULATION INSTRUMENT IS THE DEFECT,
    #: INDEPENDENT OF THE THRESHOLD: it dropped 40.6% of beams, arm-
    #: asymmetrically, through nine audit rounds without once being visible.
    BEAM_DROPS.clear()
    #: the fixture encodes BOTH directions of [2401]-retain: an 11 is the join
    #: artifact and must be TRUNCATED AND USED; a 9 is a short span and must be
    #: DROPPED. Distinct values per slot so a truncated beam that failed to
    #: contribute would move the mean.
    _mixed = [{"base_token_probs": [0.4] * 10},          # kept as-is
              {"base_token_probs": [0.9] + [0.4] * 10},  # K=1: relabel, USE
              {"base_token_probs": [0.9] + [0.4] * 10},  # K=1: relabel, USE
              {"base_token_probs": [0.9, 0.9] + [0.4] * 10},  # K=2: RESEGMENT
              {"base_token_probs": [0.7] * 9},           # K=-1: SHORT
              {"base_token_probs": None}]                # absent
    _avg = _beam_positions(_mixed, allow_stored=True)
    #: RULING 2408-offset in the fixture: K=1 is relabelled AND USED, K=2 is
    #: RESEGMENTED and dropped (my [2401] version wrongly retained it), K<0 is
    #: short and dropped. All four dispositions distinguished, all counted.
    _counted = (BEAM_DROPS.get("kept") == 1
                and BEAM_DROPS.get("relabelled_K1") == 2
                and BEAM_DROPS.get("resegmented_12") == 1
                and BEAM_DROPS.get("short_9") == 1
                and BEAM_DROPS.get("resegmented") == 1
                and BEAM_DROPS.get("short") == 1
                and BEAM_DROPS.get("absent") == 1)
    #: the per-length keys duplicate the summary keys by design, so the
    #: accounting total counts each beam once via the summary categories only
    _total = (BEAM_DROPS.get("kept", 0) + BEAM_DROPS.get("relabelled_K1", 0)
              + BEAM_DROPS.get("short", 0) + BEAM_DROPS.get("resegmented", 0)
              + BEAM_DROPS.get("absent", 0))
    #: three beams must have contributed and NOTHING else may. 0.9 must not
    #: survive truncation (wrong end) and 0.7 must not appear at all (a short
    #: span padded rather than dropped). Distinct values make both visible.
    _used = _avg is not None and all(abs(x - 0.4) < 1e-9 for x in _avg)
    ok.append(("17 drops are counted not silent",
               _counted and _total == len(_mixed) and _used,
               f"kept={BEAM_DROPS.get('kept')} K1={BEAM_DROPS.get('relabelled_K1')} "
               f"resegmented={BEAM_DROPS.get('resegmented_12')} "
               f"short={BEAM_DROPS.get('short_9')} absent={BEAM_DROPS.get('absent')}; "
               f"all {_total}/{len(_mixed)} accounted; truncated beams "
               f"{'contributed and the prepended value was removed' if _used else 'WRONG END'}"))
    BEAM_DROPS.clear()

    #: 18 ([2481]) -- CONFORMANCE. Every governing ruling must have an
    #: implementing reference in the body. [2480].3: a missing implementation
    #: has no signature, so this is the only check that can find one -- it
    #: compares against an EXTERNAL list rather than anything derivable from
    #: the file. Nine clean rounds coexisted with two settled rulings never
    #: written down, one of which gates which cells are legal to score.
    _body = io.open(__file__, encoding="utf-8").read()
    _decl_start = _body.index("GOVERNING_RULINGS = {")
    _decl_end = _body.index("}", _decl_start)
    _outside = _body[:_decl_start] + _body[_decl_end:]
    _unimplemented = sorted(k for k in GOVERNING_RULINGS
                            if f"RULING {k}" not in _outside)
    #: NEGATIVE CONTROL, and my first attempt at it was itself wrong: I
    #: asserted the identifier `GOVERNING_RULINGS` was absent from the searched
    #: region, but it legitimately appears in this very check. The control has
    #: to exercise the DISCRIMINATION, not a proxy for it -- inject a key that
    #: no line implements and require it to be REPORTED. If the declaration
    #: were left inside the searched region, every key would match its own
    #: definition and the check would certify the manifest instead of the code.
    #: My SECOND attempt was wrong too, and in the funniest available way: I
    #: searched for a literal sentinel, and the check reads its own source, so
    #: the sentinel appeared in its own definition. A STRING-SHAPED CHECK ON A
    #: FILE CONTAINING ITS OWN STRINGS IS NOT A CHECK.
    #: The control now runs THE SAME COMPREHENSION against an augmented
    #: manifest, with the key built at runtime so no literal exists to match.
    _probe = dict(GOVERNING_RULINGS)
    _probe["".join(("__control", "_absent__"))] = "must be reported missing"
    _probe_missing = [k for k in _probe if f"RULING {k}" not in _outside]
    #: AND the excision itself is asserted directly. A manifest key in its
    #: QUOTED form appears only in the declaration, so if it survives in
    #: `_outside` the declaration was not removed and every key would match its
    #: own definition. The probe is BUILT AT RUNTIME from the manifest: I wrote
    #: this test three times, and twice the literal I used as a probe appeared
    #: in the file the test reads -- once in the code, once in the comment
    #: explaining it. A FILE THAT READS ITSELF POISONS EVERY LITERAL PROBE.
    _quoted = '"' + sorted(GOVERNING_RULINGS)[0] + '"' + ":"
    _excised = _quoted not in _outside
    _sentinel_caught = len(_probe_missing) == 1 and _excised
    ok.append(("18 every ruling has an implementing line",
               (not _unimplemented) and _sentinel_caught,
               f"{len(GOVERNING_RULINGS)} rulings, unimplemented: "
               f"{_unimplemented or 'none'}; manifest excluded from its own "
               f"search={_sentinel_caught}"))

    #: 19 ([2431]) -- I implemented pair_is_scorable and wrote NO CASE for it;
    #: deleting its vocabulary comparison survived mutation. Sixth caseless
    #: guard of the night at this seat. It routes through corrected_scorer so
    #: the rule has one home ([2481]), and this case asserts BOTH directions --
    #: a refuser that never accepts is as broken as one that never refuses.
    class _VoidLinLike:
        def of(self, i): return "L1"

    class _T:
        def __init__(self, v): self._v = v
        def get_vocab(self): return dict(self._v)
    _same = {"a": 1, "b": 2}
    _diff = {"a": 1, "c": 2}                      # same SIZE, different members
    _toks = {"s": _T(_same), "j_ok": _T(_same), "j_bad": _T(_diff)}
    _ok_pair, _r1 = pair_is_scorable("s", "j_ok", _toks.get)
    _bad_pair, _r2 = pair_is_scorable("s", "j_bad", _toks.get)
    _missing, _r3 = pair_is_scorable("s", "absent", _toks.get)
    #: AND the CALL SITE, per [2494]'s own lesson: testing the function while
    #: `_cells` never calls it is the defect malign found, one level down.
    #: Disabling the call site left this case green until the next two lines
    #: existed.
    BEAM_DROPS.clear()

    class _XStash:
        _K = {"type": "beam_cross_v1", "source": "s", "model": "j_bad",
              "prompt": "p"}
        def __iter__(self): return iter([self._K])
        def __getitem__(self, k): return [{"base_token_probs": [0.4] * 10}] * 3

    _xcells, _ = _cells(_XStash(), _VoidLinLike(), allow_stored=True,
                        scorable=lambda a, b: pair_is_scorable(a, b, _toks.get))
    _wired = len(_xcells) == 0 and BEAM_DROPS.get(
        "unscorable_cross_vocabulary", 0) > 0
    BEAM_DROPS.clear()

    ok.append(("19 cross-vocabulary cells refused",
               _ok_pair and (not _bad_pair) and _r2 == "cross_vocabulary"
               and (not _missing) and _r3 == "tokenizer_unavailable" and _wired,
               f"same-vocab accepted={_ok_pair}, cross refused={_r2}, "
               f"missing tokenizer={_r3}; CALL SITE live in _cells={_wired} "
               f"(routed through corrected_scorer)"))

    #: 20 ([2417]) -- likewise implemented with no case; removing the
    #: exclusion survived. And my FIRST case for it grepped the body for
    #: `VOID_PAIRS`, which `if False:` leaves untouched -- A SOURCE-GREP CASE
    #: TESTS THAT TEXT EXISTS, NOT THAT IT RUNS. This one runs _cells over a
    #: fixture containing a void pair and asserts it is excluded AND counted.
    BEAM_DROPS.clear()

    class _VoidStash:
        _K_VOID = {"type": "beam_cross_v1", "source": "DeepSeek_R1_Distill_",
                   "model": "allenai/Olmo-3-1025-7B", "prompt": "p"}
        _K_OK = {"type": "beam_cross_v1", "source": "Olmo_3_1025_7B",
                 "model": "allenai/Olmo-3-1025-7B", "prompt": "p"}
        def __iter__(self): return iter([self._K_VOID, self._K_OK])
        def __getitem__(self, k): return [{"base_token_probs": [0.4] * 10}] * 3

    class _VoidLin:
        def of(self, i): return "L1"

    _vcells, _ = _cells(_VoidStash(), _VoidLin(), allow_stored=True,
                        scorable=None)
    _void_counted = BEAM_DROPS.get("void_pair", 0) > 0
    _void_absent = all("DeepSeek" not in str(k) for k in _vcells)
    ok.append(("20 void pairs excluded and counted",
               len(VOID_PAIRS) == 3 and _void_counted and _void_absent
               and len(_vcells) == 1,
               f"{len(VOID_PAIRS)} named; void cell excluded={_void_absent}, "
               f"counted={_void_counted}, surviving cells={len(_vcells)} "
               f"(the non-void key)"))
    BEAM_DROPS.clear()

    #: 21 ([2494]) -- PRODUCTION MUST HALT AT STAGE 2, and the RETIRED gate
    #: must not appear in the pipeline at all. Both halves matter: the first
    #: stops anyone restoring flow by having collect_rescored_cells fabricate
    #: triples; the second stops the tombstone being re-wired. run_pipeline
    #: called `gate_2_divergence` -- retired at [2390] -- for hours, and case
    #: 8's own gate set named it, so THE SELF-TEST CERTIFIED THE TOMBSTONE.
    import ast as _ast2
    import inspect as _insp2
    _rp = _ast2.parse(_insp2.getsource(run_pipeline))
    _rp_calls = {n.func.id for n in _ast2.walk(_rp)
                 if isinstance(n, _ast2.Call) and isinstance(n.func, _ast2.Name)}
    _retired_gone = "gate_2_divergence" not in _rp_calls
    _live_wired = "gate_2_divergence_rederived" in _rp_calls
    with tempfile.TemporaryDirectory() as _td2:
        with contextlib.redirect_stdout(io.StringIO()):
            _prod_stage = run_pipeline(_FixtureStash(), _FixtureLin(),
                                       {"p": "neutral", "q": "death"}, root=_td2)
    #: AND pin the collector DIRECTLY. Asserting only that production stops
    #: at stage 2 is satisfied by the gate's flat-floor halt, so a mutant that
    #: fabricated FALLING curves would pass while a flat one is caught -- the
    #: case would be measuring gate 2, not the collector. `is None` pins the
    #: contract at its source.
    _collector_honest = collect_rescored_cells(_FixtureStash(),
                                               _FixtureLin()) is None
    ok.append(("21 production halts without the re-score",
               _prod_stage == 2 and _retired_gone and _live_wired
               and _collector_honest,
               f"production stops at stage {_prod_stage}; retired gate absent "
               f"from run_pipeline={_retired_gone}; live control "
               f"wired={_live_wired}; collector returns None with no "
               f"re-score={_collector_honest}"))

    #: 22 ([2498]) -- THE DEFAULT IS THE RULING. `scorable=None` made the
    #: [2431] filter a no-op at every call site: the hook existed, the marker
    #: sat on it, conformance passed, and NOBODY PASSED IT. Reachability could
    #: not distinguish "wired" from "wireable". This asserts the DEFAULT is
    #: enforcement -- both in the signature and in behaviour, since a signature
    #: default of _ENFORCE that the body ignored would pass the first half.
    _sig_default_on = all(
        _insp2.signature(f).parameters["scorable"].default is _ENFORCE
        for f in (_cells, collect_raw_curves, collect_paired_differences,
                  collect_contrasts))

    class _UnknownStash:
        _K = {"type": "beam_cross_v1", "source": "no_such_model_xyz",
              "model": "also_no_such_model_xyz", "prompt": "p"}
        def __iter__(self): return iter([self._K])
        def __getitem__(self, k): return [{"base_token_probs": [0.4] * 10}] * 3

    BEAM_DROPS.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        _ucells, _ = _cells(_UnknownStash(), _VoidLinLike(), allow_stored=True)
    #: unresolvable tokenizers CANNOT be shown same-vocabulary, so [2431]
    #: excludes them -- conservative, and counted
    _default_bites = len(_ucells) == 0 and any(
        k.startswith("unscorable_") for k in BEAM_DROPS)
    BEAM_DROPS.clear()
    ok.append(("22 vocabulary rule is ON by default",
               _sig_default_on and _default_bites,
               f"defaults are _ENFORCE across 4 entry points={_sig_default_on}; "
               f"a production call with unverifiable tokenizers excludes and "
               f"counts={_default_bites}"))

    #: 23 ([2497]) -- A TOMBSTONE MUST BE VERIFIABLY UNREACHABLE OR IT IS AN
    #: UNDERSTUDY. run_pipeline called the retired gate for hours while its
    #: docstring announced its own retirement. Both tombstones are asserted to
    #: have no call site anywhere in the module.
    #: scoped to PRODUCTION reachability, not the whole module: a tombstone
    #: may legitimately be exercised by a case that documents its retirement
    #: (case 6 does exactly that). What must never happen is the PIPELINE
    #: reaching it -- which is what did happen, for hours.
    _mod = _ast2.parse(_body)
    _fdefs = {n.name: n for n in _ast2.walk(_mod)
              if isinstance(n, _ast2.FunctionDef)}

    def _names_called(node):
        return {x.func.id for x in _ast2.walk(node)
                if isinstance(x, _ast2.Call) and isinstance(x.func, _ast2.Name)}

    _prod_seen, _stack = set(), ["run_pipeline"]
    while _stack:
        _f = _stack.pop()
        if _f in _prod_seen or _f not in _fdefs:
            continue
        _prod_seen.add(_f)
        _stack.extend(_names_called(_fdefs[_f]))
    _prod_calls = set()
    for _f in _prod_seen:
        _prod_calls |= _names_called(_fdefs[_f])
    _tombstones = {"gate_2_divergence", "truncation_candidates",
                   "collect_raw_curves"}
    _live_tombstones = sorted(_tombstones & _prod_calls)
    #: AND the tombstone's CORRECTED BODY is pinned. Reverting it to the
    #: original `endswith("_")` test survived mutation: nothing distinguished
    #: the two, so a known-wrong detector could be restored silently into a
    #: function someone may yet wire. The discriminating case is a label cut
    #: MID-TOKEN -- no trailing separator -- which is 8 of the 15 real ones.
    _mid_token = truncation_candidates("Llama_3.1_8B_Instruc",
                                       ["Llama_3.1_8B_Instruct",
                                        "Llama_3.1_8B", "Mistral_7B_v0_1"])
    _trailing = truncation_candidates("Hermes_3_Llama_3.1_8",
                                      ["Hermes_3_Llama_3.1_8B"])
    _not_trunc = truncation_candidates("Mistral_7B_v0_1",
                                       ["Mistral_7B_v0_1", "Llama_3.1_8B"])
    _strict_prefix = (_mid_token == ["Llama_3.1_8B_Instruct"]
                      and _trailing == ["Hermes_3_Llama_3.1_8B"]
                      and _not_trunc == [])
    #: AND EVERY TOMBSTONE MUST SAY SO ON ITS DOCSTRING'S FIRST LINE.
    #: [2514].3 asked for this word at four consecutive audits, because the
    #: tombstone sat in a COMMENT ABOVE THE DEF while the docstring described
    #: a live role. A reader sees the docstring. **A state that requires an
    #: out-of-band question is a state the artifact does not record.**
    #:
    #: MY FIRST VERSION OF THIS CHECK WAS VACUOUS: it searched the WHOLE
    #: docstring for the words, and both bodies discuss retirement in prose,
    #: so reverting the summary line to a live-role description still passed.
    #: A STRING-SHAPED CHECK ON PROSE THAT DISCUSSES ITS OWN KEYWORDS IS NOT
    #: A CHECK -- fourth instance tonight. The SUMMARY LINE is structural:
    #: it is what a reader and every doc tool sees first, and prose further
    #: down cannot satisfy it.
    def _summary_line(fn_name):
        n = _fdefs.get(fn_name)
        if n is None or not n.body:
            return None
        d = n.body[0]
        if not (isinstance(d, _ast2.Expr) and isinstance(d.value, _ast2.Constant)
                and isinstance(d.value.value, str)):
            return None                      # no docstring at all
        return d.value.value.strip().splitlines()[0].strip().upper()

    _undeclared = sorted(
        t for t in _tombstones
        if not (_summary_line(t) or "").startswith(("RETIRED", "TOMBSTONE")))

    ok.append(("23 tombstones are unreachable",
               (not _live_tombstones) and _strict_prefix
               and not _undeclared,
               f"{len(_tombstones)} tombstones; reachable from run_pipeline: "
               f"{_live_tombstones or 'none'} (self-test may exercise them); "
               f"strict-prefix body pinned incl. mid-token cut={_strict_prefix}; "
               f"docstrings not declaring retirement: {_undeclared or 'none'}"))

    #: 24 ([2510].3) -- `_log2_curve` could be EMPTIED with every case still
    #: passing, and it is the transform the whole §4.2a control rests on. The
    #: identity slope(resist) == slope(log2 src) - slope(log2 jdg) holds ONLY
    #: in log2 space; without the conversion every downstream number is still
    #: computed, still type-correct, still in range, and the slopes are of RAW
    #: PROBABILITIES -- the incommensurability [2393] caught in prose. A
    #: deletion survivor on a units transform is the units defect waiting.
    #: a ZERO is in the fixture on purpose: the 1e-10 floor is what keeps
    #: log2(0) from blowing up, AND it is beam.py's own epsilon -- a re-score
    #: without it is not comparable to the legacy numbers it must sit beside.
    #: Dropping the floor survived mutation until this value existed.
    _l2 = _log2_curve([0.5, 0.25, 1.0, 0.0])
    _floor = math.log2(1e-10)
    _known = (len(_l2) == 4 and abs(_l2[0] + 1.0) < 1e-12
              and abs(_l2[1] + 2.0) < 1e-12 and abs(_l2[2]) < 1e-12
              and abs(_l2[3] - _floor) < 1e-9)

    #: AND the identity is asserted to HOLD in log2 space and FAIL in
    #: probability space -- which is what makes the transform load-bearing
    #: rather than decorative. A case that only checks log2(0.5) == -1 pins
    #: the arithmetic; this pins the REASON.
    _sp = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2]
    _jp = [0.8, 0.75, 0.6, 0.55, 0.4, 0.35, 0.3, 0.2, 0.18, 0.15]
    _sl, _jl = _log2_curve(_sp), _log2_curve(_jp)
    _res = [a - b for a, b in zip(_sl, _jl)]
    _lhs = _slope([_res[i] for i in CTRL_POSITIONS])
    _rhs = (_slope([_sl[i] for i in CTRL_POSITIONS])
            - _slope([_jl[i] for i in CTRL_POSITIONS]))
    _identity_holds = abs(_lhs - _rhs) < 1e-12
    _raw_res = [a - b for a, b in zip(_sp, _jp)]
    _raw_lhs = _slope([_raw_res[i] for i in CTRL_POSITIONS])
    _raw_rhs = (_slope([_sp[i] for i in CTRL_POSITIONS])
                - _slope([_jp[i] for i in CTRL_POSITIONS]))
    #: in raw space the difference-of-slopes identity is NOT the quantity the
    #: control reads; the transform is what makes resist decomposable at all
    _raw_differs = abs(_raw_lhs - _raw_rhs) < 1e-12 and _lhs != _raw_lhs
    ok.append(("24 log2 space is entered and pinned",
               _known and _identity_holds and _raw_differs,
               f"log2(0.5,0.25,1.0,0.0)=(-1,-2,0,{_floor:.3f}) ok={_known}; "
               f"identity holds in "
               f"log2={_identity_holds} (lhs {_lhs:+.6f}); raw-space slope "
               f"differs ({_raw_lhs:+.6f}) so the transform is load-bearing"))

    #: 10 ([2310].3) -- norm() had NO CASE for two rounds and DEL norm SURVIVED:
    #: it could return its argument unchanged and every case passed. It is the
    #: [2251].3 fix -- the function standing between this producer and the false
    #: zero that a stash label compared to a HuggingFace id returns.
    n_drops_org = norm("allenai/Olmo-3-1025-7B") == "Olmo_3_1025_7B"
    n_converges = norm("Llama_3.1_8B") == norm("meta-llama/Llama-3.1-8B")
    n_not_ident = norm("meta-llama/Llama-3.1-8B") != "meta-llama/Llama-3.1-8B"
    ok.append(("10 norm collapses namespaces",
               n_drops_org and n_converges and n_not_ident,
               "org dropped, punctuation flattened, and the two namespaces "
               "CONVERGE (identity-return fails this)"))

    #: 11-13 ([2323]) -- THE COLLECTORS' OWN CASES, added in the SAME edit as the
    #: collectors and the floor-raise, because the file's own warning said the
    #: deletion sweep is blind to them the day they are implemented. Fixture
    #: answers are hand-computable: native 0.5 everywhere, cross 0.4 everywhere,
    #: so every difference is exactly -0.1.
    class _BeamStash:
        _K_NAT = {"type": "beam_cross_v1", "source": "L1a", "model": "L1a", "prompt": "p"}
        _K_CRS = {"type": "beam_cross_v1", "source": "L2a", "model": "L1a", "prompt": "p"}
        _K_ORPHAN = {"type": "beam_cross_v1", "source": "L2a", "model": "L1a", "prompt": "q"}

        def __iter__(self):
            return iter([self._K_NAT, self._K_CRS, self._K_ORPHAN])

        def __getitem__(self, k):
            v = 0.5 if k["source"] == k["model"] else 0.4
            beams = [{"base_token_probs": [v] * 10} for _ in range(3)]
            if k is self._K_NAT:              # one malformed beam, must be skipped
                beams.append({"base_token_probs": [v] * 4})
            return beams

    class _BeamLin:
        n_lineages = 2
        def of(self, i):
            return "L1" if str(i).startswith("L1") else "L2"

    bs, bl = _BeamStash(), _BeamLin()

    #: the deletion test on _beam_positions crashes the SELF-TEST rather than
    #: failing a case -- assert_nonempty fires two collectors downstream. That
    #: is a halt, so the producer still refuses to run, but a CRASH and a FAILED
    #: CASE are different signals and only one names its subject. Case 11 is
    #: wrapped so the deletion reports as a failing case, not a traceback.
    try:
        pos = _beam_positions(bs[_BeamStash._K_NAT], allow_stored=True)
    except Exception:
        pos = None
    ok.append(("11 beam positions averaged",
               pos is not None and len(pos) == 10 and abs(pos[0] - 0.5) < 1e-9,
               "3 well-formed beams averaged to 0.5; the len-4 beam SKIPPED "
               "(a malformed beam must not shorten or poison the vector)"))

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            #: legacy mechanics under test; the PRODUCTION path refuses ([2474])
            #: not exercising RULING 2431 here -- declared, visibly
            pd = collect_paired_differences(bs, bl, allow_stored=True,
                                            scorable=None)
    except Exception as e:
        pd = {"__error__": str(type(e).__name__)}
    ok.append(("12 no-native cells dropped",
               set(pd) == {"L1"} and abs(pd["L1"] - (-0.1)) < 1e-9,
               f"prompt q has no native counterpart -> DROPPED; L1 = "
               f"{pd.get('L1', float('nan')) if isinstance(pd.get('L1'), float) else pd} (declared (b))"))

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _a, _p0, _cv = collect_contrasts(bs, bl, allow_stored=True,
                                             scorable=None)
    except Exception:
        _p0, _cv = float("nan"), []
    ok.append(("13 pos0 returned separately",
               abs(_p0 - (-0.1)) < 1e-9 and len(_cv) == 9,
               f"pos0={_p0:+.3f} returned ALONE; curve is {len(_cv)} positions, "
               f"not 10 (§4.2)"))

    #: 14 ([2337]) -- THE COLLECTOR I LEFT UNCOVERED WHILE DELETING THE WARNING
    #: THAT NAMED IT. Cases 12/13 exercise the other two; case 7 STUBS
    #: collect_raw_curves out to force gate 2's halt, so the real function was
    #: never called by anything and DEL collect_raw_curves survived. The warning
    #: I removed said exactly this, and only the diff against the retained bytes
    #: could see that the file had lost the sentence describing its own blind
    #: spot. Fixture: 3 groups -- (L1,p,native)=0.5, (L1,p,cross)=0.4,
    #: (L1,q,cross)=0.4 -- so the RAW, UNDIFFERENCED curves are 3 and flat.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            raw = collect_raw_curves(bs, bl, allow_stored=True,
                                     scorable=None)
        raw_ok = (len(raw) == 3
                  and all(len(c) == 10 for c in raw)
                  and {round(c[0], 3) for c in raw} == {0.5, 0.4})
        note = (f"{len(raw)} raw curves, flat at "
                f"{sorted({round(c[0], 3) for c in raw})} -- UNDIFFERENCED, "
                f"which is what gate 2's control needs")
    except Exception as e:
        raw_ok, note = False, f"raised {type(e).__name__}"
    ok.append(("14 raw curves undifferenced", raw_ok, note))

    #: 15 ([2358]) -- the detector that flagged complete names as truncations.
    known = {"deepseek_llm_7b_base", "Llama_3.1_Tulu_3_8B_",
             "Llama_3.1_Tulu_3_8B_DPO", "Llama_3.1_Tulu_3_8B_SFT"}
    complete_ok = truncation_candidates("deepseek_llm_7b_base", known) == []
    trunc_ok = len(truncation_candidates("Llama_3.1_Tulu_3_8B_", known)) == 2
    ok.append(("15 truncation is structural", complete_ok and trunc_ok,
               "a 20-char COMPLETE name yields no candidates; a prefix ending "
               "in the separator yields its extensions (length is not the test)"))

    #: the manifest check itself -- a named case that does not exist is a
    #: FAILURE, not an absence. This is the case that catches a missing case.
    produced = {name for name, _, _ in ok}
    unbuilt = sorted(REQUIRED_CASES - produced)
    shrunk = len(REQUIRED_CASES) < REQUIRED_CASES_FLOOR
    if unbuilt or shrunk:
        why = []
        if unbuilt:
            why.append(f"NAMED BUT NOT BUILT: {unbuilt}")
        if shrunk:
            why.append(f"MANIFEST SHRANK: {len(REQUIRED_CASES)} names against a "
                       f"floor of {REQUIRED_CASES_FLOOR} -- a name was removed")
        ok.append(("0 manifest complete", False, "; ".join(why)))
    else:
        ok.append(("0 manifest complete", True,
                   f"{len(REQUIRED_CASES)} named cases exist, floor "
                   f"{REQUIRED_CASES_FLOOR} held"))

    print("SELF-TEST — [2282]'s cases plus [2300].4's and [2310].3's\n")
    for name, passed, note in sorted(ok):
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:26s} {note}")
    n_fail = sum(1 for _, p, _ in ok if not p)
    print(f"\n  {len(ok) - n_fail} of {len(ok)} pass")
    if n_fail:
        print("  *** PRODUCER DOES NOT RUN WITH A FAILING SELF-TEST ***")
    return n_fail == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--charter", default=CHARTER_HASH,
                    help="charter hash this producer implements")
    ap.add_argument("--dry", action="store_true",
                    help="stage 1 only; opens no beam value")
    ap.add_argument("--selftest", action="store_true",
                    help="[2282]'s seven known-answer cases; touches no stash")
    a = ap.parse_args()

    print(f"M04 PRODUCER — implements charter {a.charter}")
    print(f"Route A, coarse target, lineage unit. Gates halt; none is skippable.\n")

    if a.charter != CHARTER_HASH:
        sys.exit(f"REFUSING: built against {CHARTER_HASH}, asked for {a.charter}")

    if a.selftest:
        sys.exit(0 if selftest() else 1)

    if not selftest(verbose=True):
        sys.exit("REFUSING: self-test failed. See [2282].")

    #: [2354] -- FIRST CONTACT WITH THE REPO FAILED HERE. I wrote
    #: `from malign_logits import cache_manager as cm`, which does not exist:
    #: the module is `malign_logits.cache` and the entry point is `get_cache()`.
    #: Fifteen self-test cases, six mutants, a deletion sweep and eight audits
    #: all ran against FIXTURES, so nothing ever executed this line. The
    #: integration seam is the one surface a fixture cannot cover, and it was
    #: the first thing to break.
    from malign_logits.cache import get_cache        # noqa: local import

    cm = get_cache()
    lin = Lineages()
    #: THE ROUTE, not just the filter ([2356]/[2360]). `domain` alone gives 5 of
    #: the nine -- the four with a liminal/explicit split (sexual_*, violence_*)
    #: need `domain + "_" + subdomain`. I adopted the FILTER on the first pass
    #: and not the DERIVATION, which is how 5 looked like a plausible answer.
    def _cat(r):
        d, sd = r.get("domain"), r.get("subdomain")
        return f"{d}_{sd}" if d and sd else d
    cats = {r["prompt"]: _cat(r)
            for r in json.load(open(CATEGORISATION))["prompts"]}
    stash = cm._stash("beams")

    if a.dry:
        gate_1_coverage(stash, lin, cats)                # KEYS ONLY, freeze-safe
        return

    stopped = run_pipeline(stash, lin, cats)
    print(f"\npipeline stopped after stage {stopped} of 4")


if __name__ == "__main__":
    main()
