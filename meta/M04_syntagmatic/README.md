# M04 — the continuation/combination axis

The charter is `m04_charter.md` (n at the lineage unit is 10, per the standing correction — the
charter's registered n=25 predates the count).

**M04 now has its own findings.** `findings/A_post_utterance_shock.md` (2026-08-08) — forced to
utter a word it had demoted, an aligned model finds the following region less probable for one
token, whoever writes it, and regardless of whether the word is transgressive. It is the
tiebreaker for W's discriminator table (cost small-positive, repair flat) and answers it as
neither account: the charge is local and does not propagate. Registered spec frozen at `85fd7d10`
before any statistic was computed.

That settles the question this file left open below — M04 accumulates its own findings from here,
and the continuation axis is where they belong. W stays in M01 for now: it is operationally
M01-rooted and moving it would break more pointers than it fixes, but new continuation-axis work
starts here rather than there.

## Findings index

One row per document in `findings/`. Grades are quoted from each document and are never restated in
other words; where two documents disagree the pointer is recorded and the disagreement is left where
it is.

| Doc | Claim (one line) | Status/grade (verbatim) | Unit and population | Figures | Corrects / superseded-by |
|---|---|---|---|---|---|
| `findings/A_post_utterance_shock.md` | Forced to utter a word alignment had demoted, the aligned model finds the region that follows less probable for one token, whoever wrote the continuation and whether or not the word is transgressive; the base forced at the same site shows nothing. Load-bearing sub-results: the primary is a single-token event at +1 (−0.04066, p 0.0018) and null at every other index; the robust asymmetry is `A\|A` against `A\|B` (self-relation), not aligned-scorer against base-scorer; the twin moderator is null (MARKED −0.00522, UNMARKED −0.00506). | (no status line). Header reads "**M04's first own finding.** Registered spec `meta/M02_frame_exit/registrations/spec_channel3_renewed_displacement.md`, frozen at `85fd7d10` before any statistic was computed." Per-section grades are inline: "EXPLORATORY, post-freeze, spec §10"; "(EXPLORATORY; 40 tests, no correction; per-index medians are noisier than pooled by construction)"; "**Robust:** decay-then-plateau rather than decay-to-zero"; "**Not robust:** whether the *base* terms move at all." | "Unit = pair. 33 pairs, 33 lineages, 5,112 sites with all four cells (599 half-present sites excluded and counted)." Two rosters: ALL n = 33, CLEAN n = 27 and CLEAN is the arbiter under [5011].3. Corpus `beam_fc` (`design=wave3-lexical`) plus `data/raw/fc_newlin_out/` (`design=newlin-lexical`); jais and deepseek excluded. | none | Registration and producer live in M02: `../M02_frame_exit/registrations/spec_channel3_renewed_displacement.md`, `../M02_frame_exit/scripts/channel3_run.py`. Declared tiebreaker for `../M01_displacement/findings/W_forced_continuation.md`. Self-corrects: the [5009].3 magnitude claim demoted at [5027]; the scorer-split framing posted at [5016] partly retracted by the per-index grid; an earlier "Y measured at 256 tokens" sentence corrected by lacan [5019]. Unresolved tension recorded, not resolved, per [5019].4: fc splits by SCORER, Y by TEXT, at the same 10-token window (Y write-ups `../M01_displacement/findings/Y_*.md`; the four-term grid itself is quoted only inside this file). Its cumulative long-window aggregation is questioned by `findings/attention_back_cross_own.md` §5, which also clears it of the `own`/`cross` confound (§8). |
| `findings/attention_back_cross_own.md` | Alignment shifts attention-back at the forced site by a large PAIR-SPECIFIC amount, +24% to −19%, with pair identity predicting the shift at Kruskal-Wallis p = 0.0019 over 6 pairs, but it does not shift it differently for a demoted, a promoted, or an unmoved word: 14 of 28 cells on the primary contrast. Load-bearing sub-results: the `cross`/`own` split is all that survives section 3; `D_norm` (§3b) supersedes raw `D` as the instrument; §3e records that the general effect was first normalised away and then pooled away. | "**PROVISIONAL. 6 PAIRS, 5 PROMPTS, 28 CELLS. The central prediction is REFUTED at the cell level (section 3c); sections 2 and 3 are superseded by 3b and 3c wherever they disagree.** Not registered, no spec frozen, and `registrations/plan_attention_back.md` is not amended." | "The CELL is the unit" (§3c): 28 cells over 6 pairs and 5 sexual prompts, arms auto-selected by the rule in `scripts/attn_norm_sweep.py` committed at `3de8c16c` before the sweep returned; heads (250 to 480 per cell) are explicitly not replicates. Earlier sections are the two pilot cells, "Two pairs, two prompts, n = 24 (n = 16 at window 200)" (§7). §3e is a Kruskal-Wallis across the 6 pairs on 28 prompt-level values. | none | Supersedes itself repeatedly: §§2 to 3 by §§3b/3c; the FALLER < NONMOVER < RISER ordering retracted at §3b/3c; the "forcing distorts the level" correction written up and then withdrawn (§2); "alignment doesn't do anything to attention" reversed at §3e; the earlier claim that Weatherby's scoping claim fails here withdrawn at §4. Corrects `registrations/plan_attention_back.md`: head concentration is 7x not 17x, `D` should be the primary, and the plan is still unamended (§6, §9). Takes Finding A's forced-versus-chosen invalidity ruling as binding (§3) and disputes the aggregation behind its "nothing dies" reading (§5). Its 28 fp32 cells are to be SUPERSEDED, not merged, by the extraction in `plans/attn_parquet_extraction.md` ([5148], [5226], @malign [5227].3). |

Cross-index notes (pointers only):

- Finding A is an M04 finding whose registered spec and producer sit in
  `../M02_frame_exit/` (`registrations/spec_channel3_renewed_displacement.md`,
  `scripts/channel3_run.py`).
- Finding A is the declared tiebreaker for W's discriminator table,
  `../M01_displacement/findings/W_forced_continuation.md`.
- Finding A against the Y decomposition, same nominal measure and same nominal window, disagreeing
  about which axis carries the effect: unresolved by ruling [5019].4; Y write-ups are
  `../M01_displacement/findings/Y_*.md`.

**The forced-continuation campaign lives in M01, not here, and this pointer is the map.** Its
write-up is `../M01_displacement/findings/W_forced_continuation.md`: the resist asymmetry
(counterfactual continuations scored cross-model), the damage family (whose
bounded nulls were superseded by the wave-3 detection — the claims register in the article hub
governs its current state), the entropy competitor's refutation, and the SFT census. The question
is this axis's — what alignment does to continuation — but the campaign is operationally
M01-rooted: its sites, movers and populations all derive from M01 artifacts (`r_population_k2`,
the 210-prompt beam sample, CANONICAL), and the document sits beside the scripts and results that
produce it. If M04 accumulates its own findings, revisit whether W moves here; until then, one
letter in one place beats a copy in two.

M04's own substrate note: the cross-teacher-forcing data (17 bidirectional lineages) is alive and
distinct from M05, which never ran — two substrates, two claims (docket [4626]).
