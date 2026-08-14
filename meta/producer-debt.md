# Producer, artifact and write-up debt across the meta-experiments

Split out of `plot-debt.md` on 2026-08-12 per lacan's [5554], on RH's
reading: none of the open items in that document was plot debt, and the
filename named the least severe class it contained. Severity runs the
other way — a missing producer makes a published number unauditable; a
missing artifact makes it unreproducible; a missing write-up leaves a
result uncommunicated; only then does a missing figure leave a
communicated result undrawn. Figures stay in `plot-debt.md`; this file
owns the first three classes. Classifications below follow [5554]'s
morning verification (grepped against code, not read off status lines).

## Class 1 — PRODUCER MISSING: the number cannot be checked by anyone

Sub-type A: code exists, writes nothing (fix = add write calls; an
afternoon each):

- ~~**M01 Z**~~ **DISCHARGED 2026-08-12, `06b2ad1f`, WITH A FINDING ATTACHED.**
  `z_ladders.py --write` emits `results/z_ladders.json`, 79 cells,
  capture-only (`mark()` and `within()` record the values they already
  computed and discarded; no computation altered).
  **§1 REPRODUCES EXACTLY** -- all six cells, n = 73 as stated, the
  SFT/DPO cancellation intact (+1.00 / -0.78 / +0.26 p 0.195). §4's
  closed column reproduces exactly.
  **§5's OPEN column DOES NOT** -- concreteness -2.99 -> -2.89,
  dominance +0.05 -> +0.22, ratio 5.8x -> 6.2x; the closed column is
  byte-identical. **And the drift cannot be diagnosed**, because §5's
  open column runs on 118 prompts (base->DPO only) against §1's 73 (all
  three stages) and **that n was never published** -- so corpus growth,
  a changed intersection and an arithmetic change are indistinguishable.
  The artifact now records `n_prompts` and `shared_prompts_in_chain` per
  side per family. §5's magnitude ratios are not quotable until resolved;
  every sign and star is unchanged, so its direction claims stand.
  **This is the argument for the whole document in one entry: the debt
  was not that a number lacked a file, it was that a number could drift
  and nobody could tell.**
- **M01 E** (in `C_to_O_registered_letters.md`): producer writes nothing.
  (C's two `.txt` transcripts are the adjacent artifact-debt case.)

Sub-type B: no code at all (fix = write the producer; recoverability
varies):

- ~~**M02 opus_second_order artifacts**~~ **DISCHARGED 2026-08-14,
  `503da52d` ([5906] — DARIO'S work; the post is mis-signed as malign
  per [5907]'s seat-inheritance incident), pending lacan's second-seat
  on the logic
  ([5900] split). NO TRANSCRIPT NEEDED** — the four artifacts ARE the
  reader's raw per-passage verdicts; only the aggregation was missing
  (RH's produce-raw-rows rule [5562] proving its value from the
  recovery side). `opus_second_order_results.py` refuses to write on
  any booked-value failure; every English-arm number reproduces (OR
  3.37 [1.88, 6.30], Fisher 9.61e-06, ablation McNemar 5/8 exact).
  WHAT WAS ACTUALLY LOST = THE ESTIMATOR DEFINITIONS ([5884] shape):
  the point estimate is an ODDS ratio though the finding's phrasing
  reads as a rate ratio (rate ratio = 3.12); the CI is the
  conditional-MLE interval (Woolf misses the booked values); the
  pooling is over both rounds (4.62 / 2.70 alone). All three now in
  code with booked values asserted. lacan's z_second_order gate
  INAPPLICABLE, not satisfied: no regexes are involved in aggregating
  stored verdicts. SECOND-SEAT COMPLETE ([5910], fc0b7cbf): lacan
  reproduced independently BEFORE reading dario's account; estimator
  ruled COHERENT (OR + conditional-MLE matches Fisher's conditioning)
  but the prose claimed a rate ratio — doc now heads ODDS with 3.12
  beside it. AND THE LOGIC PASS FOUND WHAT NOBODY ASKED: the
  same-side control (5/300 vs 5/300, OR 1.00, CI [0.23, 4.39])
  CONTAINS 3.37 — it cannot establish specificity, and its per-round
  ORs are 0.25 / 4.12 (unity is their average, not an observation).
  The title's "and only the contradiction", reader-arm half, now
  reads NOT YET SHOWN by this instrument; the regex pole control
  (0.93x, 52,559 passages) carries the specificity weight.
  GENERALIZED ([5912]): NOT ONE of the three same-side controls can
  exclude its own treatment effect (second-order 3.37 in [0.23,4.39];
  moral 1.16 in [0.89,2.16]; clinical 1.46 in [0.55,2.87]) — the
  producer now prints intervals, counts, per-round ORs, and a
  contains-treatment flag per row. STILL open: zh arm, marker half,
  ablation, verdict soundness. Register status: no quotable form for
  the reader arm exists; enters fenced if it enters. DOC CLOSED
  ([5914], a5499d6e): SAMESIDE caveat now a COLUMN (intervals +
  contains-treatment note per row, severity graded); the layout had
  invited reading three near-unity ORs as specificity evidence —
  nothing false, rhetorical work the contents could not support.
  Registrar's register pass done same day: floor-flags on the
  cross-lingual sign-count p-values, zh anchors marked not-yet-
  quotable with lacan's interval rider attached.

- ~~**M05 C Result 4**~~ STALE, corrected 2026-08-14 ([5901] §2):
  `m05_pair_displacement.py --recapture` now writes
  `results/m05_recapture.json`; R4 itself is WITHDRAWN ([5781]) and
  plot-debt carries C-R4 as DEAD. What remains lost is the original
  session DEFINITION, not the digits — relevant only if R4 is ever
  revived.

- ~~**M06 cross-lingual MATCHED-PROMPT leg (both keys)**~~ **CLOSED BY
  WITHDRAWAL 2026-08-14, `5c0b2915` ([5935]) — a disposition this file
  did not previously have: neither discharged nor outstanding. The
  debt is closed because the CLAIM is gone, not because the code came
  back.** lacan exhausted both recovery routes and withdrew both
  matched-prompt legs in the finding (struck, not deleted), removed
  the `on the same prompts` clause at source, and stated the three
  disagreeing key counts (97/100/71) in the doc. Booked as Class 1B
  the same day on dario's referral: `crosslingual_arms.md`'s parse-free key block —
  23,677 passages, 71 keys, total_drift zh -0.0263 (5/20) / en -0.0171
  (3/22), mean_drift zh -0.0454 (1/24) / en -0.0409 (0/25), DiD .69 /
  1.0 — exists in NO artifact and has NO producer. The finding calls
  this leg "the one that travels" and the frontmatter calls the
  matched-prompt DiD "the strongest form"; the REGISTER's quotable
  upgrade rests on it and is SUSPENDED until recovery ([5934]).
  Persisted: {total_drift, mean_drift} x {pooled, n_sents-matched}
  only (`matched` = n_sents, not prompt). All eight persisted values
  reproduce. RECOVERY assigned by RH to lacan; favourable transcript
  case if inline (eight values + population sizes + key definition =
  rich needle strings; dario's pre-stand-down search located the
  densest candidate log, session log `cdbe9c9e-a018-45bf-95e9-6bf81e96e908` (a `~/.claude` transcript UUID; the full form is written out because a 36-char hyphenated UUID cannot be mistaken for a git short hash by a reader OR a grep) — the same log that carried the Y
  section 5 recovery). NEAR-MISS worth keeping: a grep for the medians
  hit `crosslingual_arms_full.json`, same median to four places but
  sign counts 1/24 against the parse-free 5/20 — a different
  population wearing the same digits.

- **M04 `a_decay_disjoint.json` + `a_decay_and_topic.json`** — NEW
  2026-08-14 ([5987], lacan), and a shape this file has not carried:
  **CLASS 1B FROM THE OTHER DIRECTION — an artifact with no claim.**
  Both hold decay against token distance at n_pairs = 42 with
  monotone decline and per-pair sign counts, i.e. a real population
  and a usable shape; and grep over the whole tree returns NO
  producer, NO citing finding, NO queue entry, no `_about`, no
  `_provenance`, and no underscore key of any kind. Added in a bulk
  commit naming neither. **What they measure is unknown** — the
  `pref_vs_topic` keys are `logq`/`logp`, which read as probability
  rather than attention, so this may be surprisal-decay and not the
  attention-decay shortlist 9 wants. **IDENTIFIED SAME DAY ([5989],
  malign, the owner): the A-LADDER ON THE PASSAGE SUBSTRATE, not
  attention work.** `logq`/`logp` are the aligned and base
  log-probabilities of the injected word — the ladder's own q and p —
  and `A|A`/`B|A` are the four-term decomposition, so the decaying
  quantity is the PREFERENCE term against token distance. And
  n_pairs = 42 is the tell: that is the PASSAGE corpus against Finding
  A's fc 33, **a different substrate as well as a different
  quantity**, carrying A_RESULTS's EXPLORATORY / nothing-quotable
  fence. Neither file can discharge an attention debt, so plot-debt
  item 12's measurement debt stands as a RUN and the hold is released.
  Residual declared by malign: it has NOT verified that the rho is the
  preference term specifically rather than another correlation from
  the same run, since no producer emitted these under a recorded name.
  **THE ORPHANHOOD IS THE DURABLE ENTRY: 31 of 43 JSON artifacts in
  `meta/M04_syntagmatic/results/` carry no leading-underscore key at
  all**, and the 12 that do are inconsistent among themselves
  (`_about`, `_fingerprint`, `_meta`, `_reading`, `_producer`) — the
  0-of-75 parquet finding landing one container over, in the finder's
  own directory. Cause, in malign's words: it fenced the load-bearing
  artifact and left its siblings from the same sitting, which is the
  copies clause generalised from values to ARTIFACTS. **A batch
  produced together needs fencing together, because unfenced siblings
  are indistinguishable from orphans to the next reader** — which is
  exactly what happened here. Batch being stamped with substrate and
  status.

- **M05 A_acquisition R4 statistics** — NEW Class 1B 2026-08-14
  ([5954], dario, search space stated with a working control):
  Spearman +0.61 p 1.3e-05 (SFT-arm level co-movement), co-drift rho
  -0.12 p .45, sep-leads .085 / ratio-leads .17. Zero hits for every
  term across `*.py`; unrestricted, exactly one file — the finding
  that quotes them. Same shape as the cross-lingual parse-free leg.
  With lacan (instrument owner).

- **M05 per-checkpoint `pole_sep` reduction** — NOT a missing producer
  but a MISSING DEFINITION, which this file did not previously
  distinguish ([5954]); **DISPOSITION RE-DECLARED — EXECUTED
  2026-08-14, `19240d87`, RH said go at [5958]. The fourth disposition
  in this file, and its first instance ([5965]).** Plan committed
  ALONE before the producer existed (`570afad4`), rule stated in the
  artifact's `_about`, six old values kept visible as
  superseded-not-reproduced, republished numbers WEAKER than the old
  ones at step16000 on both ladders, and the plan's own recorded
  prediction (co-movement) came out positive-but-not-significant and
  was REPORTED rather than re-reduced. History: `m05_pole_sep.csv` exists at (checkpoint,
  group, role, layer) grain, 166,255 rows, and every published
  per-checkpoint value needs a reduction that nothing states. Median-
  and mean-over-all-cells both miss the booked values (0.3675 against
  a booked 0.475 at stage1-step16000). Deliberately NOT swept — three
  targets against eighteen plausible reductions would yield a fitted
  recipe indistinguishable from a reproduction ([5935]). ESCALATED
  ([5958], lacan): a DECLARED role filter exists (`role == "both"`,
  m05_pole_sep.py:188) which dario's median-all did not apply — it
  moves stage1-step0 to 0.7975 against a booked 0.795 and leaves the
  others wrong, which is precisely the near-miss the stopping rule
  exists for. lacan then raised the bar to SIX simultaneous targets
  (real and null columns x three checkpoints) and got ZERO exact; no
  plan declares the reduction, neither producer performs one, and
  lacan does not remember it. **RECOMMENDED DISPOSITION — RE-DECLARE,
  NOT RECOVER:** the artifact is committed and at the right grain and
  is not in question; only the rule is missing. So declare the
  reduction in a one-paragraph plan BEFORE running, run it, republish
  the table with the NEW numbers, and mark the old six
  SUPERSEDED-NOT-REPRODUCED. Unlike a recovered recipe it carries its
  own provenance from the first line. The finding's claim is that the
  null collapses and recovers AS the real column does — a claim about
  two columns moving together, which every reduction tried reproduces
  qualitatively, and the finding itself says THE LEVEL GAP LICENSES
  NOTHING. **RH's call, because new numbers in a published table are
  his; item 10 stays held until he rules.**

- ~~**M03 ICC 0.855**~~ **DISCHARGED AS RE-DECLARED 2026-08-14,
  `17c7888e` ([6009], lacan; RH said go) — the fourth disposition's
  second instance.** Plan `plan_icc_redeclaration.md` committed ALONE
  at `b2b9a0cb` before the producer existed; rule in the artifact's
  own `_about`. **THE POPULATION REPRODUCES** — 12 scenarios f21_inst
  and 18 m03_slice ASSERTED IN ADVANCE, 52 rungs quoted from the
  producer's own AL rather than re-derived, 127,590 paired cells —
  which is exactly where malign's attempt ended, and lacan credits the
  population instinct as load-bearing, adding only that the plan made
  it an assert before rather than a discovery after. **NEW VALUES:
  ICC(1) 0.647 (IQR 0.541-0.735, 90 items) and 0.589 (0.479-0.682,
  143) against the booked 0.855 and 0.846** — well below the booked,
  far above malign's 0.085, as the plan predicted in writing before
  running. **AND THE ICC WAS NEVER THE DECISION-RELEVANT QUANTITY**:
  what licenses collapsing 52 rungs is the DESIGN EFFECT
  `1 + (k-1)*ICC`, which at k=50 is 32.4 and 25.2 — a scenario's 50
  rungs are worth about 1.5 observations, so the rung-unit p-values of
  2.4e-14 and 1.0e-11 were computed on an n roughly 30x larger than
  the data supports. **The finding's decision to discard them was
  right by a margin a 0.2 error in the ICC does not touch.** THE ONE
  THING THAT CHANGED RUNS THE OTHER WAY: effective n is 18.3 and 30.0
  against the 12 and 18 the scenario unit uses, so **the analysis
  UNDER-used its data** — the opposite of the over-collapse this was
  run to check, and lacan records having put 25% weight on the
  over-collapse reading and being wrong in DIRECTION. No result moves
  (7/18 is at chance and stays there at n=30); headroom recorded, not
  claimed. STATISTIC DECLARED WITH ITS BIAS: ICC(1) one-way random
  effects, and because rungs are ORDERED along training and therefore
  not exchangeable, a systematic trend is charged to within-group
  variance so **ICC(1) UNDERSTATES** — conservative toward the reading
  that would revive the rung unit, and the collapse survives it
  anyway, which is the strongest form the result could take.
  ICC(2,1)/(3,1) named in the plan as DECLINED, not tried. **AND THE
  PRINT STATEMENT IS FIXED AT THE MECHANISM**: `d_ladder_fields.py:157`
  now loads the artifact and prints what was computed, or says the
  value is unavailable and refuses to state one — because a corrected
  literal would have been the same defect with a better number in it.
  History: REGISTRAR ERRATUM: an earlier version of this entry
  authorized the recovery to dario citing "[5998]" — a post number I
  had not yet been issued and which belongs to malign's report doing
  the opposite. Never write an identifier you have not observed
  ([5921]/[5922]); corrected here, and dario's standing offer at
  [5997] is moot because the work is done. WHAT MALIGN FOUND:
  (1) **the value is a LITERAL inside a print statement** —
  `d_ladder_fields.py:157` prints "ICC of the paired difference across
  rungs is 0.85"; nothing computes it. Purest instance of the
  outward-claim class ([5978]) — a claim about the data, in a string,
  where nothing can ever test it — and it is the one carrying a
  campaign-wide rule. It also says **0.85 where the finding says 0.855
  and 0.846**, so producer prose and finding disagree before anyone
  attempts either. (2) **The substrate SURVIVES** —
  `d_ladder_fields.csv`, 511,242 rows, 52 alignment rungs, 30
  scenarios — so this is a READ, not a run; establishing that took
  twenty minutes against a measurement, which is lacan's [5987]
  asymmetry paying out a second time. (3) ONE honest reduction, and it
  fails badly: f21_inst ICC 0.565 against a booked 0.855, m03_slice
  0.085 against 0.846 — **and the population does not reproduce
  either** (11 scenarios where the finding says 12), which malign
  rightly calls a stronger refutation of its own recipe than the ICC
  gap. (4) It stopped there rather than sweeping a dozen reductions
  against two targets ([5954]'s situation, [5960]'s reading: one
  target snapping is what recipe-fitting feels like from inside).
  (5) **A THIRD CATEGORY: not a missing producer and not a missing
  definition, but a PRODUCER THAT ASSERTS ITS OWN OUTPUT.** ASK, and
  it is RH's: re-declare and re-run the ICC from the surviving CSV
  under a stated reduction, and until then `D_ladder_selection.md` §6
  and every citation of the not-an-observation rule carry that the
  number is unreproduced. **The RULE is not in question** — 52 rungs
  of one lineage are plainly not 52 observations, and
  `prompt_authoring_guide.md:318` already gives the structural reason
  needing no ICC at all. Only the 0.855 is. Originally booked Class 1B ([5901] §3, dario,
  search space stated): `0.855` has zero hits in any `*.py` in the
  tree; quoted in `M03/findings/D_ladder_selection.md:265` and
  `M03/README.md`, and it UNDERWRITES the standing rule that the rung
  is not an observation. Orphaned within one module (the other tree
  hits are different quantities, checked).

- **M05 fig12b / fig13** — cited-figure provenance 2026-08-14
  ([5901] §4): both actively embedded in findings, neither has a
  producer in any `.py` (fig12 also producer-less but DELIBERATELY
  retired, not adoptable).

- **M04 A_offset_slope.png + A_offset_slope_terms.png** — NEW Class 1B
  2026-08-14, SELF-REFERRED by malign ([5904] §3, verified: zero *.py
  hits for the basename; referenced only in A_RESULTS.md's own figure
  list). Worst-shaped instance: A_-prefixed (reads as the registered
  finding), exploratory passage substrate, AND unstampable — the
  producer-side substrate stamp protecting the other 12 cannot reach
  them. Recovery favourable (A_RESULTS.md §9 quotes the slope values);
  malign runs it (debt owner).

- **M01 Registration S odds-scale statistics** — CANDIDATE 1B
  2026-08-14 ([5901], reader tier, NOT yet re-verified): the odds
  ratios in S finding 3 (3.26, 1.56, 0.18 at p=4.6e-06; the
  harm-versus-prohibition gradient) may have no producing code —
  `s_analysis.py` declares itself model-free and neither output CSV
  carries an odds column. Verify before booking firm; the M05 C-R4
  case is the standing reason reader-tier items are leads.
- **M04 A, the exploratory half** — CORRECTED 2026-08-14 ([5901] §1,
  dario, verified with location): the per-index PRIMARY row HAS code,
  at `meta/M02_frame_exit/scripts/channel3_run.py:346-353` (the file
  lives in M02's scripts, not M04's — why it read as absent), and that
  block is headed DECLARED SECONDARIES, so "exploratory" was also
  wrong for that row. STILL UNLOCATED: the four TERM rows, the
  +1-only four terms, the long-window sweep. The long-window
  aggregation is not recoverable from the frozen slot spec ([5024].2)
  and MAY NOT BE RECONSTRUCTIBLE; the finding's own Robust/Not-robust
  split turns on it.

DISCHARGED FROM THIS CLASS, for the record and the method. **Two of the
four discharges so far surfaced something the debt was hiding** (Z's §5
drift; the M02 heredoc's dependencies about to vanish with a tmp
directory), which is the reason to work the class rather than annotate it:

- **M02 `second_order_naming.md` graded-stimulus control** — was never in
  this document; pure sub-type-B debt (inline heredoc, 2026-08-11 18:34,
  dependencies in a session scratchpad, numbers surviving only in commit
  e804b1c5's message). Discharged at `3ac2c124` by transcript recovery,
  logic unedited, REPRODUCES THE PUBLISHED TABLE TO THE DIGIT (V1
  1.04/1.24/2.22, V3_SAFE 1.15/1.30/2.26, 17 groups, 67,198 passages).
  Found because RH asked about a heredoc, not because any inventory
  pointed at it.
- **M01 H2**: `--json` written by malign ([5430]),
  `results/h2_depth_primary.json`.
- **M04 A, registered half**: `channel3_run.py --write` emits
  `results/A_post_utterance_shock.json`; capture-only, reproduces
  exactly ([5439]).

Sub-type C — **CODE THAT EXISTS, RUNS, AND IS IN NO COMMIT** (a class
this file did not carry until 2026-08-14, [6036], dario; registrar
reproduced both counts):

- **Ten untracked producers.** `git ls-files --others --exclude-standard
  | grep '/scripts/.*\.py$'` returns 10 — four in M01
  (`boundary_signatures`, `canonical_sites_newlineages`,
  `k_conjunctions`, `m05_sites_newlineages`), six in M02
  (`exit_logit_markers`, `l1_term_anchor`, `l1_three_anchor`,
  `l1_valence_pole`, `lens_from_cache`, `z_opposition_smoke`).
  **Severity is lower than it looks and dario said so rather than
  dramatising it**: none is cited by any findings doc, README,
  `plot-debt.md` or this file, and they read as pilot code. But they
  are one `git clean` from gone, nobody has decided about them, and
  **an uncommitted producer is unauditable by exactly the argument
  that makes a missing one unauditable** — while looking fine from
  inside the working tree. The state that is wrong and silent, one
  container further out.

- **23 untracked data files, ~424 MB, READ by committed producers that
  do not write them** — so ignoring them costs no disk and costs the
  reproduction path: the producer stays, the input vanishes, and the
  failure appears only when someone runs it. Upper bound, method
  declared: basename matching with a stem heuristic separating writes
  from reads, imperfect in both directions (a looser first pass gave
  43 files / 432 MB and was withdrawn — `.key` was matching
  `dict.keys()`).

- **AND THE FOUR THAT BITE, because a findings doc naming a file is
  not a heuristic** — registrar verified the citation counts directly:
  `y_confirmatory_coded.jsonl` (143.9 MB, cited by 2),
  `lens_group_layer.jsonl` (54.1 MB, cited by 2),
  `m05_norm_mass.parquet` (12.0 MB, cited by 2). All three untracked.
  `lens_prompt_layer.jsonl` (108.6 MB) is untracked and cited by 0 on
  my grep, so it belongs with the 23 rather than these. **If a cited
  artifact is ignored, a finding cites a file that is not in the
  repository** — which is a citation-integrity question and therefore
  the pen's to flag, though the disposal is RH's.

- **CORRECTED SAME DAY, [6040]-[6042]: THE SET IS FOUR, AND THE FOUR
  ARE FOUR DIFFERENT DECISIONS.** My amendment above verified the four
  names dario listed and dropped a fifth that belonged —
  `h2_depth_receipt.json`, cited by `H2_alignment_depth.md`. **Checking
  the members of a list is not checking the list.** Sizes below are
  read from `.githooks/pre-commit` (`core.hooksPath` is redirected
  there; `.git/hooks/` is empty and reasoning from it would have said
  no guard exists), which carries TWO thresholds: BLOCK=104857600
  (100 MiB) and WARN=52428800 (50 MiB).

      h2_depth_receipt.json         7,437 B    0.007 MiB  clean
      m05_norm_mass.parquet    12,095,948 B     11.5 MiB  clean
      lens_group_layer.jsonl   54,132,203 B     51.6 MiB  WARN, NOT blocked
      y_confirmatory_coded    143,913,692 B    137.2 MiB  BLOCKED

  **Committing is available for three of the four.** malign's [6042]
  self-correction (`51.6 MiB is not over 100 in any unit`) retires the
  both-are-blocked reading.

- **REPRODUCIBLE IS NOT REGENERABLE — the limit of lacan's
  ignore-plus-pointer rule, and the reason the fourth is different in
  kind.** That rule works on a DERIVATION (the mediation table:
  deterministic, one pass, frozen inputs). `y_confirmatory_coded.jsonl`
  is a RECORD OF A RUN: its producer `y_run_manifest.py` is a paid,
  sampled API annotation over 62,681 items, hours long, and
  `y_repair_rt.py` then rewrote the file IN PLACE (atomic replace) to
  fix rt_band after the HTML bug. A REGENERATE line naming the runner
  therefore describes a more expensive experiment that returns
  different bytes, minus the repair. **For a record, the copy IS the
  artifact of record.**

- **AND THE COPY DID NOT EXIST. `y_confirmatory_coded.jsonl` was
  found on NEITHER volume** — not diderot, not chambers — so the one
  artifact that cannot be committed and cannot be regenerated also had
  no backup, 143.9 MB of paid output in a single untracked file, on
  the same night a seat swept the shared tree by accident.
  **Registrar copied it to diderot as a protective, additive,
  non-deciding action** (RH's disposal question is untouched):

      /Volumes/diderot/malign-logits/meta/M01_displacement/results/y_confirmatory_coded.jsonl
      sha256 6b25cfa60dc9b3b3e3ca0930dbb2f9d741bd0fc21f8a11e2fc10f62be071cec8
      62,681 lines both sides — matches the manifest item count exactly

- **A CITATION IS A CLAIM THAT SOMETHING IS IMPORTANT; A READER IS A
  CLAIM THAT SOMETHING IS LOAD-BEARING** ([6063], malign; reader counts
  verified at this seat by opening the hits, not by grep). **This file
  had been recording the cheap one.** The two numbers do not
  substitute for each other and they diverge in both directions:

      y_confirmatory_coded.jsonl     6 citing docs   12 READERS + 1 rewriter
      m05_norm_mass.parquet          4 citing docs    1 reader
      verse_fleet/*.f16              prose only       0 readers, 2 booked

  **`y_confirmatory_coded.jsonl` is therefore the worst case of the four
  and it is not close.** Eleven analyses plus its own producer read it
  back, and **one of the twelve is `y_exit_typology.py` in M02** —
  verified here, an `IN =` reaching across the module boundary into
  M01's results — **so a module whose owner may not know it has an M01
  dependency loses an analysis the day this file goes.**

  Three properties make it the worst case and no two would: 137.2 MiB,
  over GitHub's hard block, so committing was never available; a paid,
  sampled API annotation over 62,681 items **then rewritten in place**
  by `y_repair_rt.py`, so lacan's precondition bites exactly here (*the
  pointer only works when the thing pointed at is derivable; where it
  is a record the pointer is a copy path or it is nothing*); and twelve
  dependents, which the citation count did not show.

  **The copies made on 2026-08-14 are currently the entire protection
  for those twelve dependents.**

  `lens_group_layer.jsonl` was already on diderot at identical size;
  `m05_norm_mass.parquet` is absent there but regenerable from a
  tracked deterministic producer, so its absence is a cost and not an
  exposure. **One caveat if lens takes a pointer: its regeneration
  reads `data/**/*.hidden.f32`, which is gitignored and mid-migration
  to external disk (190 still local) — a pointer whose input is moving
  must name where the input went.**

## Class 2 — ARTIFACT MISSING: producer works, output never committed

- **M01 J §1**: `results/arch_displacement.json` absent (`arch_did_*`,
  `arch_fields_*` exist). The section that stands has no artifact.
- **M01 W**: only `fc_checkAC_snapshot.csv`; the pair-level table lives
  in docket posts and the register.
- **M02 pole-axis next-word**: `results/dp.pkl` + `pole_axis_*.log`
  untracked/gitignored, machine-local. Re-run `pole_axis_build.py`
  (BGE-m3 encode, GloVe download) to regenerate.

- ~~**twp store / ClickHouse ingest gap**~~ **CLOSED BY RULING
  2026-08-14, RH via [6077]: DO NOT INGEST THE JULY CELLS.** Retired,
  not deferred. `data/twp_ingest_worklist.json` (5733fe2d) is KEPT with
  the ruling in its own `_about`, because **a list that vanishes teaches
  the next seat to re-derive it**, and re-deriving this one cost four
  seats an evening. Final accounting of the 10,540 distinct gap cells:

      4,843  EMPTY PAYLOAD — correctly absent. `twp_words` stores one
             row per WORD; a record with no words above theta=0.001
             writes no rows. Arithmetic, not a filter.
      5,696  JULY (twp_cloud 29-30 Jul pre-versioning, twp_grid_v3
             30-31 Jul, twp_phase32b 31 Jul) — RH: not to be ingested.
          1  cloud_run — not pursued.

  **So the store is COMPLETE for everything it should hold: 964,679 of
  970,376 non-empty cells = 99.41%**, with the 0.59% now formally
  excluded rather than outstanding. The receipt reads *empty payload at
  theta=0.001*, not *corrupted prompt* — which is what four wrong
  explanations had to be cleared to establish (registrar's
  "never offered", lacan's "one batch one cause", malign's "tokenizer
  damage excludes CJK", registrar's "every missing cell is CJK").
  **Each was measured, none careless, and each died to a case with the
  effect and none of the proposed mechanism.** The control was
  `Olmo-3-1025-7B@stage1-step0` — 1,073 cells, zero CJK, all empty —
  **the LARGEST entry in the gap and the least interesting one**, while
  four seats worked the multilingual cluster because it had a pattern.
  And it was settled by `n_words`, a column lacan put in the manifest
  with no specific purpose because RH asked for a manifest.

## Class 4 — ARTIFACT STALE AGAINST A DEPENDENCY: producer unchanged, numbers moved

> **INSTANCE WITHDRAWN 2026-08-14 ([6113]). THE FOUNDING CASE DID NOT
> HAPPEN.** `m05_field_flow.parquet`'s numbers never moved: compared on a
> sorted unique key, 08-11 / my 12:38 re-run / a third run today agree to
> **4.44e-16 with ZERO rows differing.** My "212,776 of 245,422 values
> changed by up to 0.884" was a positional subtraction of two frames whose
> ROW ORDER differs between runs. The alignment guard I ran auto-selected
> `ckpt_idx` alone — not a unique key, 245,422 rows over 95 values — **so
> it returned True and could not have returned anything else.** The
> 2,231-byte growth I took as corroboration was parquet compressing a
> different row order.
>
> **FIRST CONFIRMED INSTANCE, SAME DAY ([6118], dario): `t_fans.csv`.**
> Written 2026-08-06 over 2,182 cells; the fan now finds 2,174 present in
> all five arms and every mean moves in the fourth decimal (full
> 0.0651458 -> 0.0651986). The cause is the **prompt-catalogue refresh of
> 08-12** — a dependency that moved after the artifact was written, with
> the producer unchanged, which is exactly this class. **The RATIOS are
> unaffected to better than 0.03 points, and the ratios are what U §4
> claims and the figure draws**, so dario's producer asserts the ratios
> and DECLARES the absolute drift rather than asserting it away. **A guard
> should protect the claim being made, not the incidental value it was
> computed from.** So the class is observed after all — by someone else's
> folder, on a real dependency, at a magnitude that changes no conclusion.
>
> **THE CLASS IS RETAINED AS A HAZARD AND WAS DEMOTED TO UNOBSERVED**
> (for about an hour, until the above); It is
> a real shape and the structural test below is worth keeping, but no
> instance has ever been seen: three seats audited their own folders after
> this one was booked and all three came back clean, which was the signal
> and was read as reassurance. Everything below describes a mechanism, not
> an event.
>
> **WHAT WAS REAL, AND IS NOW FIXED:** the producer accumulated through
> unordered dict/set iteration, so its row order AND its float summation
> order varied per run, making the parquet byte-unstable across identical
> runs. `sorted()` on the output and on both accumulation loops; verified
> **byte-identical across two full recomputes** (`df15b49dd6657187`).
> Byte-stability is the cheapest "did this change?" check there is, and
> its absence is the whole reason a false alarm was possible.

New 2026-08-14. The first three classes ask whether the producer exists,
whether it wrote, and whether anyone wrote it up. **None of them catches
an artifact that went stale because a LIBRARY IT IMPORTS changed.** The
producer's own history looks clean, so every staleness check we have
returns green.

- **`data/m05_field_flow.parquet`** — committed 2026-08-11 (`8dd9b1fd`).
  `malign_logits/fields.py` then changed on 2026-08-12 (`3669da8e`,
  Chinese USAS + K coder ratings as lookups; also `f52b2767`). The
  producer `m05_field_flow.py` did not change at all between those dates.
  **I re-ran it on 08-14 to fix a truncated subtitle and 212,776 of
  245,422 `mass` values moved, by up to 0.884** — same rows, same order,
  same `covered` column, different lexicon underneath. A cosmetic fix
  re-based a measurement that `meta/M05_emergence/findings/B_field_flow.md`
  cites.
- **THAT RE-BASED PARQUET IS NOW IN HISTORY**, swept into `41830fb7`
  ("cloud_bad_machines: the verse-fleet download-starvation blocklist
  entry") by the shared-index incident lacan disclosed at [6087]. The
  commit message describes none of it. Nothing is lost — `8dd9b1fd` still
  holds the 08-11 version — but the current store of record changed
  lexicon basis inside a commit about a blocklist.
- **AND THE FOLDER IS NOW INTERNALLY INCONSISTENT, which is mine.**
  `fig8a/8b` recomputed on the CURRENT lexicon; `fig9a/9b` re-rendered by
  me from `data/m05_field_flow_fine.parquet` (mtime 08-11 16:10, so the
  OLD lexicon) via the `--figs` flag I added; `fig10 x4` likewise, since
  `m05_field_flow_per_namespace.py` reads that same 08-11 cache. **Three
  figure families of one measurement on two lexicon versions.**
- **A CHEAP RE-RENDER PATH RE-RENDERS THE OLD NUMBERS, WHICH IS THE POINT
  OF IT AND ALSO THE HAZARD.** I added `--figs` reasoning that a fix which
  costs a full recompute is a fix nobody makes; it is also the reason two
  of the three families never saw the new lexicon. Both halves are true
  and the flag now says so in its own comment.
- **THE STRUCTURAL TEST** (dario [6096], sharpened by malign [6098]).
  Class 4 does not threaten figures; it threatens **producers that
  recompute their own input**. So the property is not care at repair
  time, it is whether re-running a drawing script can write anything
  except pixels:

      reads a committed artifact, writes only PNGs
          -> a re-run is a RE-RENDER. Cannot re-base. Safe by construction.
      computes its artifact AND draws from it
          -> a re-run is a RECOMPUTATION. A moved library re-bases the
             data and the producer's own git history looks clean.

  **A cheap re-render path is safe not because it is cheap but because
  it does not recompute** — the cheapness is a consequence, and a cheap
  path that recomputed would be equally cheap and not safe at all.
  Corollary (malign): **a fence can be added WITHOUT re-running, and
  then the artifact is provably unchanged rather than presumed
  unchanged.** That is the default for any text-only repair.
- **CLASS 4's EXPOSURE HAS TWO SOURCES AND THIS FILE NAMED ONE**
  ([6102]). As written above the mechanism is a moved LIBRARY. The other
  is a moved STORE, and a producer reading ClickHouse is exposed whenever
  anyone ingests — so zero `malign_logits` imports is not immunity. The
  check that covers both: library commit history since the artifact was
  written, PLUS store insertion time for the cells read. **THREE, not
  two** (malign [6105]/[6108]): the third is a **SIBLING MODULE** — a
  script in the producer's own folder that reads the store on its
  behalf. `a_position_figures.py` has ZERO `malign_logits` occurrences
  in any form and imports `a_dose_response`, which has three ClickHouse
  hits. The library grep finds nothing, the producer's history is clean,
  and the store-insertion check reaches the store but not the sibling
  that reads it. **The dependency graph we audit is the one visible from
  the file, and a sibling import is a dependency the file names without
  the audit following it.**
  **AND THE SIBLING NEED NOT TOUCH A STORE — IT NEED ONLY HOLD A
  DEFINITION** (dario [6110]). `plot_displacement_network` reads no
  store at all; three of dario's producers import `FRAG` from it so that
  no two figures can disagree about what counts as a word. **A shared
  constant is a dependency the audit does not follow, and it is MORE
  common than a shared store because it is what good factoring
  produces.** The thing that moves need not be code that runs.
  M05's four instances, checked at this seat: `m05_lens_ladder` and
  `m05_pole_sep_pythia` import `m05_pole_sep`; `m05_licit_run` and
  `m05_licit_smoke` import `m05_syntax_tags`.
  **Detection note from the same post: for a case-1 producer, a
  BYTE-IDENTICAL re-rendered PNG is stronger evidence of re-render than
  any assert passing** — not a booked value holding, but the output
  being the same bytes.
- **M05 CLASSIFIED BY THAT TEST** (writes a non-PNG artifact AND draws
  AND imports `malign_logits`). **Three, not the sixteen my first grep
  reported** — that predicate flagged every genuine compute producer,
  which is not the hazard; the hazard is the conjunction:

      m05_field_flow.py           BOTH + LIB   <- the one that fired
      m05_field_flow_fine.py      BOTH + LIB
      m05_pair_displacement.py    BOTH + LIB

  Four more write-and-draw with NO library exposure (`m05_pythia_capacity`,
  `m05_sense_curve`, `m05_syntax_curve`, `verse_capacity_figs`): the
  shape is dangerous, the protection is circumstantial, and it lapses
  the day anyone gives one of them an import.
- **OWED**: `m05_pair_displacement.py` is the one of the three with no
  `--figs` path. Its 08-14 re-run moved nothing (max diff 9.99e-16, so
  the lexicon does not reach its outputs), which is luck about which
  lexicon moved rather than a property of the file. The flag is eight
  lines of the pattern already in the other two.
- **OPEN, AND RH's**: either re-run every field-flow artifact on the
  current lexicon deliberately and re-check `B_field_flow.md`'s quoted
  values, or pin them all to the 08-11 lexicon. **The one state nobody
  chose is the one we have.**

- **P §3's BOOKED +0.0229 HAS NO ARTIFACT, AND CANNOT GET ONE BY
  RE-RUNNING** — new 2026-08-14 ([6140], dario, blocked on plot-debt
  item 15). The finding books a MEAN OVER FIVE RUNS and fences the
  reader off the largest draw in bold: *"Do not quote 21%."* **The
  committed artifact `results/k/predict_embed_en_glove.json` IS that
  largest draw** (+0.0256, 21.2% of the 0.120678 headroom), so a figure
  drawn to this campaign's own discipline — booked numbers re-derived
  from the artifact — asserts the forbidden value and asserts it
  correctly. Drawing 19% instead puts a number on a panel that no
  committed artifact reproduces. **The other four draws exist only as
  prose**, in the doc and in the producer's docstring.
  **AND RE-RUNNING CANNOT FIX IT, BY DESIGN.** `k_predict_embed.py:69-82`
  states that `HistGradientBoosting` is thread-order dependent through
  OpenMP, that `random_state` does not control it, and that
  `OMP_NUM_THREADS=1` was CONSIDERED AND REJECTED because it "would
  report one arbitrary thread schedule as though it were the answer."
  So the booked mean is **unreproducible in principle**: five fresh draws
  give a different five and a different mean.
  **THEREFORE THIS IS A RECORD, NOT A DERIVATION** — the same class as
  `y_confirmatory_coded.jsonl` — and lacan's precondition applies
  unchanged: *the pointer only works when the thing pointed at is
  derivable; where it is a record, the pointer is a copy path or it is
  nothing.* **The fix is to PERSIST the five draws as data, not to
  recompute them**, and it must be add-beside: the producer overwrites
  its results file on every run (its own docstring says so), so a re-run
  replaces a cited artifact with a sixth draw and loses the record that
  raised the question. **RESOLVED 2026-08-14 ([6143], dario, on RH's word, by
  exactly this disposition):** `predict_embed_en_glove_draws.json`
  transcribes the five draws from both recording sites, marked
  `measured_by_this_seat: false` with the fence quoted in the file, and
  **`predict_embed_en_glove.json` is untouched — verified byte-identical
  to HEAD at this seat**, its k=50 row recomputing to +0.0256 = 21.2% of
  the 0.120678 headroom, which is the forbidden figure and confirms the
  diagnosis.
  **AND A SIXTH DRAW NOW EXISTS, BELOW THE RECORDED FLOOR.** dario took
  one before the ruling landed; it overwrote the cited artifact exactly
  as the docstring warns, and was restored. It came in at **+0.0213 =
  17.65%**, under the recorded minimum of 0.0216, so the observed range
  widens from **17.9–21.2% to 17.7–21.2%**.
  **NO RIDER IS OWED — registrar over-called this and dario corrected it
  ([6145]).** `P_unnamed_axis.md` §3 states an INTEGER range, "18-21%",
  and **17.65% rounds to 18%**, so the stated range still contains the
  draw. I compared a four-significant-figure measurement against a bound
  written to the unit and booked a defect that does not exist. **Same
  move as comparing a commit author against a diff author: both
  quantities real, not the same quantity** — and it is the day's own
  class with a number rather than a name in it.
  **WHAT IS ACTUALLY WORTH KNOWING, and it is smaller and live:** the
  range now rests on SIX draws and **its floor sits on the rounding
  boundary.** One further draw below 17.5% would break the stated range.
  Cheap for a future seat to know; requires touching no finding, which is
  as well since seat ownership is not establishable from git, every
  commit being under the shared identity.
  **The new figure's asserts check each draw within 0.003 rather than to
  the digit**, because an equality assert there would assert that a
  nondeterministic producer is deterministic and would fail on every
  honest re-run — the claim is the ladder's shape, not any draw.
  **Second-order note worth keeping either way** ([6140] §1): the share
  figures are the max of ten sweep rows and `k_predict_embed.py:207` says
  so — *"THE BEST ROW IS SELECTED AFTER SEEING THE SCORES, so this
  fraction is an upper bound on what the embedding recovers, not an
  estimate of it."* GloVe's quoted row is k=50 and bge's is its own max
  at k=100, so the two encoders are not compared at the same k. Nothing
  is mislabelled — the bge row names no k — but the table invites a
  same-k reading it does not support.

- ~~**P §7's TABLE: TWO ROWS CARRY THE SAME n AND R2**~~ **WITHDRAWN
  2026-08-14 ([6168]). THE DUPLICATION DOES NOT EXIST AND §7 IS RIGHT.**
  malign emitted the missing artifact and the two values are different:

      concreteness  r2_axis    0.0920581956367692   (concreteness_en.json)
      length        r2_length  0.09209674697588033  (length_en_glove.json)
      difference               3.855e-05 -- both round to 0.0921

  **A genuine four-decimal coincidence**: two different quantities on two
  different overlaps landing 3.9e-05 apart. Verified at this seat against
  the new artifact. dario's suspicion was reasonable, my confirmation of
  it was wrong, and the doc transcribed two correct numbers.
  **THE REASONING ERROR IS THE ENTRY** (malign's own, and sharper than
  anything the false alarm cost): the full-precision search found
  `0.09209674697588033` in exactly one file and that was read as proof
  the concreteness row had copied it. **A presence test cannot prove an
  absence claim.** Full precision establishes where a value THAT EXISTS
  ON DISK came from; the concreteness R2 was absent from the repository
  because its producer never wrote one — a fact established in the same
  post — and that absence was then read as evidence the number was not
  independently real. **A name with one referent has one STORED source,
  which is not the same as one source.**
  Kept as the inverse of the `.f16`-in-a-NOT-clause false positive:
  there a NOT-clause read as a presence, here an absence read as a
  derivation.
  **LEDGER STATE 2026-08-14, verified at this seat: FOUR OF SIX ROWS
  REPRODUCE.** Both length rows did all along; both concreteness rows now
  do, exactly on all four values —

      coder concreteness  n 6084  r2_axis 0.092058  cos 0.267139   §7: 0.0921 +0.267
      Brysbaert Conc.M    n 2916  r2_axis 0.118305  cos 0.240610   §7: 0.1183 +0.241

  **The two REGISTER rows still do not, and they are the rows the entry
  leads with** (*register ~half*). `register_en.json` gained an
  `index_table` carrying `n_words` and two rhos; it holds **no R2 and no
  cosine** — checked, the strings are absent. And they are not derivable
  from what is there: rho² = 0.1891 against the printed 0.1994, so the
  R2 is not the squared Spearman. **The row that most needs to travel
  with a caveat is the row that cannot be checked**, and the
  frequency-residualised weld ([5606]) is a register quantity too.
  Outstanding as Class 1A: `k_register` emitting `r2_axis` and
  `cos_axis_dir` alongside the rhos closes it. Not a hold — dario's
  design ground was always the blocker and is settled.
  **AND THE EMISSION IMPROVED THE FINDING RATHER THAN ONLY CHECKING IT**
  (dario [6170]): two independent concreteness instruments land at R2
  0.0921 and 0.1183 with cosines +0.267 and +0.241 — **agreeing on
  direction, differing by a quarter on magnitude.** Same shape as GloVe
  against bge on the headroom ladder, and it is the evidence that
  concreteness is a real minority component rather than one measure's
  artifact. **A number nobody could check became, on being written down,
  a replication.** — `concreteness_en.json` exists, and §7's rows are
  checkable rather than transcribed.
- ~~**SUPERSEDED DETAIL, kept because the correction chain is the
  record**~~ — new 2026-08-14 ([6164], dario; verified at this seat against
  the artifacts). The table reads:

      Brysbaert concreteness  2,916   0.1183   +0.241
      coder concreteness      6,084   0.0921   +0.267
      word length (en/bge)    6,120   0.1386   -0.213
      word length (en/glove)  6,084   0.0921   -0.211

  **`coder concreteness` and `word length (en/glove)` share BOTH n and
  R2.** The glove row reproduces to the digit — `length_en_glove.json`
  gives `r2_length` 0.09209674697588033 and `cos_axis_length_dir`
  -0.21115659582296298. The bge row reproduces too (0.13856858,
  -0.21282). **The concreteness row has no artifact anywhere**: no
  `concreteness_en.json` exists, and `scale_solo_en.json` carries none of
  0.0921 / 0.1183 / 0.0470 / 0.1994.
  **Three of six rows reproduce, three do not, and one of the three that
  does not duplicates a value from one of the three that does.** The
  thread-nondeterminism of §3 explains a SPREAD; it does not explain the
  same string twice.
  **CONCLUSIVE AT FULL PRECISION** (malign [6165], and the right
  instrument where mine was not): `0.0921` occurs as a coincidental
  substring in three or four unrelated CSVs, but **0.09209674697588033
  has exactly ONE hit in the repository** — the length row's own
  artifact. **A rounded value is a name; the full-precision value is
  closer to a relation.**
  **UNVERIFIABLE, NOT UNRECOVERABLE** (malign's distinction, and it is
  the one that unblocks the row): both producers run against committed
  inputs, so this is **Class 1A with a discharge path** — run them, emit
  the artifact, and three of six rows becomes six of six.
  **ONE CORRECTION TO THAT REPORT, checked here:** malign wrote that
  neither producer contains a `json.dump`. True of
  `k_concreteness.py` (zero). **`k_register.py` HAS one and writes
  `register_en.json`, which exists** — but that artifact holds
  `indices / used / n_cells / n_words / auc` and contains **no R2 at
  all**, none of the four cited values, and not even the substring `r2`.
  So the outcome is the same by a different mechanism: one producer
  stores nothing, the other stores a different quantity than the table
  cites. **A producer that writes AN artifact is not a producer that
  writes THE value**, which is the artifact-edge distinction arriving
  one level in.
- **AND PLOT-DEBT 15(2) IS HELD ON THE PEN'S RULING, NOT ONLY ON THE
  ABOVE.** The entry asks for a stacked decomposition — register ~half,
  concreteness ~a quarter, length zero, unnamed majority as remainder.
  **§7b states that interiority and abstraction are colinear BY
  CONSTRUCTION, and `confound_en.json` measures it** (register x
  concreteness rho 0.493). *"Removing it costs half"* and *"removing it
  costs a quarter"* are two counterfactuals, not two slices: they can sum
  past 1 and here they nearly do. **A stacked bar with an unnamed
  remainder asserts an additive partition the finding explicitly
  denies** — the same defect as the site-delta denominator in 15(3), and
  the queue entry asks for the pooled form both times. **Held. Do not
  draw half a ledger and do not draw a partition the finding refuses.**
  15(4) is unblocked and self-contained; take that instead.

## Class 3 — WRITE-UP MISSING: a real result no document owns

- **M02 L1 pilot family**: frame membership at chance from an LLM coder
  AND four geometric constructions, against pole axis AUC 0.995 and
  content/function 0.812. Two independent instrument families at chance
  is evidence about a construct; the README summarises, nothing owns it.

CORRECTED from the old inventory: the M02 lens depth series is NO LONGER
write-up debt — `ratio_moves_destination_unknown.md` has carried it since
2026-08-11 (the old entry was stale the day after it was written). The
old shape table's "M02: 6 finding docs" is also stale; M02 has 10.

## Standing rules that apply to every entry

Producers carry `_invocation` and input shas ([5467]/[5468]); a regex
character class is a population definition ([5524]); slices, thresholds
and denominators named wherever a number travels ([5500]); no audit
number quotable until a second seat reproduces it from the artifact
([5503]); a population that re-derives from a mutable registry is not
frozen, whatever digest sits beside it ([5559]); a negative existence
claim needs a search space, not a search history ([5562]); produce RAW
model x prompt x value and compute summaries after — a summary artifact
that discards its rows cannot say which rows it had (RH's rule, [5562],
demonstrated same-day by z_ladders' own discharge); THE PROMPT IS A UNIT
TOO — a result reported only pooled across prompts has not been shown to
exist at any of them; pooling can cancel or invert, not merely shrink
(lacan's proposal [5581] §4, K's 30x cancellation and two sign flips as
the demonstration; malign's corpus carries it natively [5582]; second
seat registrar [5584]; rider from [5585]/[5586]: in a minimal-pair
design the WRONG analysis always reports the larger n, because the
unpaired shortcut silently includes by-construction singletons the
paired test must drop — the smaller denominator reads as correctness;
and per [5587], any per-pair statistic on a corpus with non-uniform
pairability reports its own denominator per pair, never the corpus
mean — the frozen pairable fraction is 89.9%, progress snapshots drift;
GENERALISED per [5629], third instance on a third axis: a per-pair rate
hides behind every corpus mean on this corpus — pairability (89.9%,
singleton stems), sequence-scorability (Aquila2 9.92%, 300x the median),
and now empty text (bloom-7b1 0.538%, a 30x outlier) — so ANY
corpus-level completeness or coverage claim names its per-pair outliers
beside the mean, and the two completeness axes are ORTHOGONAL:
the worst-scoring pairs have perfect text, verified by count 1,142,400
sequences, 99.98% text-complete; and lacan's aggregation-level
clause [5632], three same-day instances at one seat plus plan A's window
coupling: FIXING A QUANTITY AT ONE LEVEL OF AGGREGATION DOES NOT FIX THE
SAME QUANTITY AT THE LEVEL ABOVE, and the instrument that would notice is
often disabled by the fix itself — when a fix normalises something, name
the level it lives at and ask what varies at the next one); where a gate is really a
threshold on a number, gate on the number, never on a categorical label
standing in for it ([5578] §5, two defects of one shape in a day; second
seat [5579]); AN IDENTIFIER IS BOOKED ONLY AFTER THE
POSITIVE CHECK — resolve it in its own namespace at the moment it crosses
from a post into a record (commit -> `git cat-file -e <sha>^{commit}`;
content pin -> digest the named file; run id -> the run record; session
id -> labelled as such), because verifying that a string is NOT an X is
not evidence about what it IS ([5604] fabricated hash booked into two
records within the hour, [5605]; malign's mechanisation proposal [5607];
lacan's namespace diagnosis [5608] — three eight-hex false positives, one
of them the freeze receipt itself, where the invited "repair" would have
swapped a content pin that survives a rebuild for a commit hash that does
not; malign's verified acceptance and the truncated-path grep lesson
[5609]: search the checker's own path field, never a retyped one; and
[5613], the third same-day defect of one shape: in all three the
machine-generated value was fine and the hand-built text channel was not
— an unquoted heredoc executed markdown backticks and deleted two paths
while displaying the hash correctly, so post bodies travel by FILE, never
by interpolating heredoc, and a deletion is the benign end of the class
only because it happened to un-parse: the dangerous version still
parses; and [5616]'s relay clause: do-not-relay covers FACTS ABOUT OTHER
SEATS' ENVIRONMENTS — write "both seats report it absent", never the
merged flat assertion, because a relayed fact with its provenance
stripped fails invisibly, and the receiving seat never observed it). And the
UNDECLARED-PROPERTY clause ([5639]/[5640], two instances in one exchange):
a design may not rely on a property of the current data that nobody
declared — the passage ORDER BY key was collision-safe only because no
checkpoint sits in two pairs (true, measured at [5640], never previously
written down, and false the day tulu and tulu-no-safety enter one
passage-style corpus), and the prompt-id split was safe only because no
pair name contains '|' — so either DECLARE the property and ASSERT it in
code, or prefer the operation that cannot break over the one that happens
not to (malign's form, [5640] §4); kin to pin-the-content-not-the-envelope
and gate-on-the-number.

THE ENGINE-STATE CLAUSES (drafted [5668], marked up [5670], countersigned
[5670]/[5671], committed on both signatures; the night of five instances
and four mechanisms, [5649]-[5667], is the trail throughout).

SCOPE LINE, above both clauses because it governs every claim made under
them: A VERIFICATION CLAIM CARRIES ITS SCOPE. "X is clean" is a fact
about X — the register, then meta/, then the tree, then scripts/, four
times in one night, the gap moving down a directory each time — so every
clean/complete/verified sentence names what it covered and THE NEAREST
THING IT DID NOT — the adjacent scope a reader would otherwise assume
was inside it, never the unbounded complement, which nobody can name and
which turns the rule decorative ([5674]) — and two known cases stand in
for a population only after the population is enumerated.

CLAUSE 1 — READS (engine-state). A figure read from a live table is a
claim about an ENGINE STATE, not only a time. Every ReplacingMergeTree
read WHOSE FIGURE TRAVELS goes through FINAL; a read taken deliberately
raw is a MEASUREMENT OF ENGINE STATE and says so at the call site
(ch_reconcile's raw column is the named instance — the rule must not
outlaw the instrument that motivated it, and the 57x cost of FINAL
[5671] is why the universal form would be routed around). FINAL is
necessary, not sufficient: A SORTING KEY IS A STORAGE DECISION; AN
ANALYSIS KEY IS A CLAIM ABOUT THE UNIT, and deduplication names which
one it collapses to — operational shape FINAL + GROUP BY the analysis
key, the aggregation over surviving re-measurements NAMED as a declared
choice with its harmlessness MEASURED, not assumed. Worked example, on
twp_words the night this was written: raw 68,243,599; FINAL 60,425,347;
uniqExact(model,prompt,word) 58,292,343 — the last two disagree by
2,133,004, the cross-source triples exactly, so "add FINAL and you are
done" is off by the entire span of the defect. A defect in a read
propagates into whatever the read was used to CHOOSE, not only what it
measured — re-verification re-runs the SELECTION, not just the values.
And an instrument's own report of ABSENCE — no coverage, insufficient n,
skipped — is a claim about the query at least as much as about the
world, and is re-derived when the query changes rather than carried
forward as a fact about the data (the 27 no-coverage rows; and
passage_reconcile's narrower precedent: absence is counted from the
DECLARED POPULATION, never from the absence of a failure file — a
precedent that sat in the tree as a special case for three weeks before
anyone generalised it; the sentence entered the law because a
no-coverage row cost something, not because anyone was careful, [5674]).

CLAUSE 2 — WRITES (resumption and provenance). An output artifact
records WHAT was produced, not what PRODUCED it, so a resumable producer
treats "already done" and "done correctly" as one fact. The skip
predicate therefore consults a PRODUCER FINGERPRINT — the code SHA or
the parameters that matter — never existence or row count alone
(fingerprint leads because it needs nobody to remember anything); where
a fingerprint is impractical, the producer refuses to resume across a
version change loudly, or resumption is opt-in per run. Correcting a
query is never sufficient: the results cache is invalidated BY HAND,
moved not deleted, staleness in the FILENAME (.RAWDUP / .CONTAMINATED /
.DESTROYED-<utc>-<id>-<reason>). Reconciliation counts rows against a
population and says nothing about provenance; only a fingerprint does
(the 240 bands a resume would have reprinted; vllm_y_run.py:396's
row-count skip, saved by an empty directory; Teuken's 2,692 drops as
ACCIDENTAL positive provenance — an accident, not a design).

THE ARM-BEHAVIOUR CLAUSE (named [5648], fourth instance 2026-08-13, rule
per its own threshold): ON A MATCHED CORPUS, A SURFACE QUANTITY IS
EXPOSED TO BEHAVIOURAL DIFFERENCES BETWEEN THE ARMS THAT MASQUERADE AS
THE QUANTITY — formatting (bullets-as-sentences, found by reading),
termination (length x mean_lp coupling, found by measuring), coverage
(window-fit missingness, found by counting), and degeneration
(repetition collapse and multilingual word-salad scoring as extreme
syntax, found by close reading the tails; the two modes evade OPPOSITE
screens — collapse is low-TTR, salad is HIGH-TTR). Every verdict on such
a corpus declares which behavioural strata it conditions on, reports
per-arm strata rates as description beside the verdict, and prefers
medians where the behaviour lives in the tails.

THE SOURCE-COMPLETENESS CLAUSE ([5710], with [5573] as its older narrow
form): A COMPLETENESS QUESTION CANNOT BE ANSWERED FROM THE ARTIFACT
ALONE — a store cannot report what never reached it. CH said count =
FINAL = distinct = 29,504, in normal company, and every number was true
while exactly half the pair's sequences sat outside the store;
16-samples-of-two-runs and 16-samples-of-one-run are the same row count.
Completeness is asked of the SOURCE against a declaration (the fleet's
manifests-never-absent-failure-records rule was this clause for one
corpus); the artifact answers only uniqueness and internal consistency.

THE DISJOINT-WINDOW CLAUSE ([5687]-[5689], three instances in one
analysis): A CUMULATIVE WINDOW CAN HIDE A REAL EFFECT BY AVERAGING IT
WITH THE RANGE WHERE NOTHING HAPPENS — it cost A's late A|A effect (five
cumulative windows, all null, over a band effect at p 1e-4) and made the
decay profile untestable (cumulative means share data). Cumulative
windows are this campaign's producer default and the WRONG default for
anything positional: where a claim is about WHERE in a sequence
something happens, the windows are disjoint or the test cannot see it.
