"""READ GATE ON THE FORMAT BATTERY. Offered [2100].2, taken up [2101].

    from scripts.f20x_read_gate import load
    d = load()                      # every column EXCEPT `text`
    d = load(unfreeze="A")          # requires RH's route on the docket

WHY. [2099].1 froze arm-level reading of `data/f20x_format_battery.parquet`
until RH picks a route: **the asset is not the 9,280 completions, it is their
UNREADNESS, and unreadness is spent the first time anyone looks.**

**BUT THE FREEZE IS A RULING OVER SEATS AND THE FILE IS READABLE BY ANY
COMMAND.** It holds exactly as long as three seats remember it holds — which is
the care-based remedy this docket has spent two days replacing with mechanism.
**This converts three attestations into one property of the file.**

WHAT IT ALLOWS AND WHAT IT REFUSES.

    ALLOWED   every structural column: family, base_model_id, model_id, arm,
              stim_id, condition, word, level, prompt, draw, temperature.
              Row counts, cross-tabs, completeness grids, N-per-cell. All of
              [2098] and [2103] were built from these.
    REFUSED   `text` — the completions themselves, and therefore every
              arm mean, per-level statistic and family contrast.

**`prompt` IS ALLOWED AND `text` IS NOT, and the line is exactly there: the
prompt is the stimulus we authored, the text is the model's answer.** Reading
what we wrote unblinds nobody; reading what came back is the whole of it.

UNFREEZING IS NAMED, NOT SILENT. `load(unfreeze="A")` or `"B"` returns `text`
and PRINTS the route it was unfrozen under. **A gate that can be opened without
saying so is a comment.** The route is not validated against the docket — that
would be a lock, and custody does not hold the key to RH's decision — but an
unfreeze that has to name a route cannot happen by reflex or by autocomplete.
"""

import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(ROOT, "data", "f20x_format_battery.parquet")
OUTCOME = "text"
ROUTES = {"A": "exploratory read of the two existing levels",
          "B": "freeze-then-complete; read the full grid under the frozen plan"}


def load(unfreeze=None, path=PARQUET):
    """The battery WITHOUT its outcome column, unless a route is named."""
    d = pd.read_parquet(path)
    if unfreeze is None:
        if OUTCOME in d.columns:
            d = d.drop(columns=[OUTCOME])
        return d
    if unfreeze not in ROUTES:
        raise ValueError(
            f"unfreeze must name RH's route, one of {sorted(ROUTES)} — "
            f"got {unfreeze!r}. A gate opened without naming why is a comment.")
    print(f"*** F20x READ GATE OPENED under route {unfreeze}: {ROUTES[unfreeze]}")
    print(f"*** `{OUTCOME}` is returned. Arm-level reading is no longer frozen.")
    return d


#: THE FOUR JOIN KEYS, AND THE EXISTING ROWS ARE THE AUTHORITY ON ALL FOUR.
#: [2117].3: **a completion run is a JOIN, and every join key is a claim that
#: two populations were built the same way.** A completion run would naturally
#: be written from the SPEC, and the spec disagrees with the artifact on the
#: one that matters: [248] enumerates FIFTEEN stimuli and templates §2 lists
#: fourteen plus the deictic, while the artifact holds SIXTEEN — the sixteenth
#: is a nonce.
#:
#: **THE PRIMARY IS A WITHIN-STIMULUS PAIRED SIGN TEST, so a stimulus present
#: at `rung` and absent at `document` DROPS OUT OF THE PAIRING SILENTLY.** The
#: contrast still computes, over 15 pairs instead of 16, and reports nothing
#: about the missing one — and every count on the way looks clean: 29 families,
#: balanced arms, complete levels. A defect that survives every gate because
#: the gate counts what is there.
LEVELS_DECLARED = ("rung", "spelled_rung", "prose_q", "narrative", "document")


def assert_join_compatible(new, path=PARQUET):
    """Call BEFORE the first generation call of a completion run.

    `new` is the DataFrame (or dict of iterables) the run is about to produce.
    Raises on any join-key divergence from the existing rows. Refusing to start
    costs minutes; discovering it after 13,920 completions costs the run.
    """
    old = pd.read_parquet(path)
    errs = []

    #: (a) stimulus set — the one the spec gets wrong
    o, n = set(old.stim_id), set(new["stim_id"])
    if o != n:
        errs.append(f"stim_id set differs: only-existing {sorted(o - n)}, "
                    f"only-new {sorted(n - o)} (existing has {len(o)}; the SPEC "
                    f"says 15 and is WRONG — the artifact is the authority)")

    #: (b) level vocabulary, byte-for-byte. `prose_question` or `spelled-rung`
    #: would split the level factor into six values where the design has five,
    #: and every per-level count would still look balanced.
    bad = set(new["level"]) - set(LEVELS_DECLARED)
    if bad:
        errs.append(f"level values not in the declared five: {sorted(bad)}")
    if set(new["level"]) & set(old.level):
        errs.append(f"level already present in the artifact: "
                    f"{sorted(set(new['level']) & set(old.level))} — a "
                    f"completion run must add levels, not re-generate them")

    #: (c) draw numbering CONTINUES rather than restarting, or the merged file
    #: carries two conventions and nothing on its face says so.
    if min(new["draw"]) <= max(old.draw):
        errs.append(f"draw restarts at {min(new['draw'])}; existing max is "
                    f"{max(old.draw)} — continue the numbering")

    #: (d) SEED DISJOINTNESS, COMPUTED FROM THE SCRIPT, NOT ASSUMED.
    #:
    #: WITHDRAWN AND REPLACED [2123]. This branch used to refuse because
    #: NEITHER population carries a `seed` column, on the reading that
    #: templates §3 requires seeds in the output. **THE CLAUSE WAS
    #: MISQUOTED**: §3 is DERIVATION-BASED -- `SEED0` declared in the script,
    #: resume keys derived and never read back from disk -- so a stored column
    #: was never part of the design and the absence is not a violation. My
    #: refusal was calibrated to a clause, and the clause was wrong.
    #:
    #: THE REAL HAZARD SURVIVES AND NEITHER FRAMING CAUGHT IT. The seed is a
    #: PURE FUNCTION of `SEED0 + cell` (f20x_format_battery.py:64, :204), so a
    #: completion run that restarts the counter REISSUES THE IDENTICAL TORCH
    #: SEEDS. Different prompts, so not a validity failure -- but the two runs
    #: draw from the same RNG states, and any claim of independent sampling
    #: across levels would be false.
    #:
    #: AND THE STRIDE DIFFERS, which is why "continue the counter" is not
    #: "continue by the same stride": the existing run advances `cell` by
    #: len(STIMULI)*len(LEVELS) per model-arm (:159) with LEVELS=2 -> 32; the
    #: completion has LEVELS=3 -> 48. Measured on the artifact: 58 model-arms
    #: x 16 stimuli x 2 levels = 1,856 cells consumed, seeds
    #: [20260729, 20262585). The completion needs 58 x 16 x 3 = 2,784 and MUST
    #: START AT cell >= 1856.
    #:
    #: DIRECTION, CORRECTED [2127]. I first wrote "overlaps after 38
    #: model-arms", which reads as first-38-fine-then-bad. IT IS THE REVERSE.
    #: A restarted model-arm k occupies cells [48k, 48(k+1)), so it collides
    #: with the existing [0, 1856) iff 48k < 1856, i.e. k <= 38:
    #:     CONTAMINATED  k = 0..38   THE FIRST 39
    #:     CLEAN         k = 39..57  THE TAIL OF 19
    #: **A triage performed on my sentence would KEEP the contaminated head and
    #: DISCARD the only clean rows**, and both 39 and 19 are plausible
    #: partitions of 58, so no count would object. **A QUOTIENT FROM A BOUNDARY
    #: DIVISION NAMES A CROSSING POINT, NOT A DIRECTION — the side comes from
    #: the inequality, never from the number.**
    SEED0, EXISTING_CELLS = 20260729, 1856
    if "seed" in new:
        lo = min(new["seed"])
        if lo < SEED0 + EXISTING_CELLS:
            errs.append(
                f"seed {lo} falls inside the range the existing run consumed "
                f"[{SEED0}, {SEED0 + EXISTING_CELLS}) — restarting `cell` "
                f"reissues identical torch seeds. Start at cell >= "
                f"{EXISTING_CELLS} (seed >= {SEED0 + EXISTING_CELLS}).")
    #: OUTCOME is exempt: this runs BEFORE generation, so the completions do
    #: not exist yet. A schema check that demanded them would make the guard
    #: uncallable at the only moment it is useful.
    miss = set(old.columns) - set(new) - {OUTCOME}
    if miss:
        errs.append(f"columns missing from the new rows: {sorted(miss)}")

    if errs:
        raise AssertionError("JOIN INCOMPATIBLE — do not generate:\n  " +
                             "\n  ".join(errs))
    return True


if __name__ == "__main__":
    d = load()
    print(f"gated load: {len(d):,} rows, {len(d.columns)} columns")
    print(f"  columns: {list(d.columns)}")
    print(f"  `{OUTCOME}` present: {OUTCOME in d.columns}  <- must be False")
    try:
        load(unfreeze="nope")
    except ValueError as e:
        print(f"  bad route refused: {str(e)[:70]}...")
