# F20x format battery: completion-run specification

**STATUS: FREEZE CANDIDATE, written at [2119] under [2118].2. Freezes on its own
hash BEFORE generation.**

**This document governs the generation of the three missing levels only. It makes
no claim about readouts** — those are frozen separately at
`meta/M02_frame_exit/f20x_format_battery_registration.md`,
sha256[:16] `39e6fae722b0be4a`. **The two are deliberately separate: a
registration records what was ruled before the run and must stay checkable
against its own dates. This one is about a run that has not happened.**

## 0. What is generated

    levels        spelled_rung, prose_q, document        (rung, narrative exist)
    families      29                                     unchanged
    arms          2 per family                           unchanged, per [2106].1(ii)
    stimuli       16                                     see §3 -- SIXTEEN, not the spec's 15
    draws         5 per cell                             matched to what ran, per [2112].2
    temperature   1.0                                    unchanged
    max tokens    200                                    templates §3

    3 x 29 x 2 x 16 x 5 = 13,920 completions

**On n=5.** It is matched, not chosen. [2112].2: **mixing n is worse than a low n
here** — the primary is a within-stimulus paired contrast and the secondary an
ordering over levels, so interior cells at a higher n against extremes at n=5 put
unequal precision on the two ends of every contrast and heteroskedastic points on
the slope, with the noisiest points at the ends, which carry the most weight in an
ordering test. The clean alternative is a full five-level rebuild at n=20
(92,800 completions), which buys five power points on the readout that already
clears and ten on the one that does not reach 0.7 either way ([325] table, measured
branch).

## 1. THE GOVERNING PRINCIPLE

**A COMPLETION RUN IS A JOIN, AND EVERY JOIN KEY IS A CLAIM THAT TWO POPULATIONS
WERE BUILT THE SAME WAY.** ([2118].3)

**The EXISTING ROWS are the authority on every key — not this document, and above
all not `docs/f20x_format_templates.md`, which is what a fresh implementer would
naturally read and which disagrees with the artifact on the key that matters.**

## 2. Schema and `draw` numbering

Same schema, same column names, same dtypes as `data/f20x_format_battery.parquet`.

**`draw` CONTINUES; it does not restart.** Whatever numbering the existing rows
carry (`draw.nunique() == 5` in every cell, per [2106].1(iii)), the new levels
adopt the same convention and the same value set. **A merged battery carrying two
draw conventions has nothing on its face that says so.**

## 3. Stimuli — SIXTEEN, and this is the item that would silently bite

**The artifact holds sixteen stimuli. [248] enumerates fifteen and templates §2
lists fourteen plus the deictic.** The sixteenth is a nonce ([2106].1(i)):

    person   1P, 3P-he, 3P-she, 3P-they                                    (4)
    nonce    N-fenmit, N-flant, N-glorp, N-gorpin, N-plost, N-quiln,
             N-tarnu, N-velbin, N-zendle                                   (9)
    object   O-adze, O-froe, O-quern                                       (3)

**WHY THIS IS THE DANGEROUS ONE. The primary is a WITHIN-STIMULUS PAIRED sign
test. A stimulus present at `rung` and absent at `document` drops out of the
pairing SILENTLY** — the contrast still computes, over 15 pairs instead of 16, and
reports nothing about the missing one. **Every count on the way looks clean: 29
families, balanced arms, complete levels.** This is the [2011] shape.

**REQUIRED, as an assertion and not a convention ([2118].2):**

```python
assert set(stim_ids) == set(existing_stim_ids)   # against the parquet, before the first call
```

**Fails loudly before any spend. A count-based gate cannot catch this defect —
15 is a perfectly plausible number of stimuli.**

## 4. Level vocabulary, byte-for-byte

The three new values are exactly `spelled_rung`, `prose_q`, `document`, as spelled
in templates §1. **`prose_question`, `spelled-rung` or any near-miss silently
splits the level factor into six values where the design has five, and every
per-level count still looks balanced.**

Assert against the design's five-value vocabulary after the write, not by eye.

## 5. Seeds

Templates §3 declares `SEED0 + cell`, cell incrementing across the run, and
**"seeds are in the output or the run is void."**

**`cell` CONTINUES from where the existing run ended. It does not restart at 0.**
A restart reissues the seeds already spent on `rung` and `narrative`. Not a
validity failure — the prompts differ — but it makes `seed` non-unique within the
merged file, and the declaration's entire point was that a seed identifies a draw.

## 6. Prompt text

The three levels' templates are already written, in templates §2, for all five
stimulus classes. **They are used verbatim.** The identical stimulus wording
appears in every cell ([241].4's constraint: the contrast is within-word and
paired, or the three bands are unresolvable by construction). Stipulated classes
carry the stipulation in every cell ([184]).

## 7. Post-run reconciliation, before the merged file is used for anything

    rows added                    == 13,920
    rows per (family, arm, level) == 80, constant, no exceptions
    level vocabulary              == exactly 5 values, spelled as §4
    stim_id set per level         == identical across all 5 levels
    draw value set per cell       == identical across all 5 levels
    seed                          == unique across the merged file
    merged total                  == 23,200

**Any failure is reported, not repaired in place.** A completion run that lands
wrong is regenerated from a corrected spec; a merged file patched after the fact
has no honest provenance.

## 8. What this does NOT govern

**No readout, no threshold, no band, no unit.** All of those are frozen at
`f20x_format_battery_registration.md` and were ruled before the original run. **A
completion spec that acquired a readout clause would convert a pre-registered
battery into a post-hoc one, which is the entire thing the separation exists to
prevent.**

**The reading freeze of [2099].1 is unaffected by this document and holds until
the grid is complete and the route is ruled.**
