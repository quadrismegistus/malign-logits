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
  (0.93x, 52,559 passages) carries the specificity weight. STILL
  open: zh arm, marker half, ablation, verdict soundness. Register
  status: no quotable form for the reader arm exists; enters fenced
  if it enters.

- ~~**M05 C Result 4**~~ STALE, corrected 2026-08-14 ([5901] §2):
  `m05_pair_displacement.py --recapture` now writes
  `results/m05_recapture.json`; R4 itself is WITHDRAWN ([5781]) and
  plot-debt carries C-R4 as DEAD. What remains lost is the original
  session DEFINITION, not the digits — relevant only if R4 is ever
  revived.

- **M03 ICC 0.855** — NEW Class 1B 2026-08-14 ([5901] §3, dario,
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

## Class 2 — ARTIFACT MISSING: producer works, output never committed

- **M01 J §1**: `results/arch_displacement.json` absent (`arch_did_*`,
  `arch_fields_*` exist). The section that stands has no artifact.
- **M01 W**: only `fc_checkAC_snapshot.csv`; the pair-level table lives
  in docket posts and the register.
- **M02 pole-axis next-word**: `results/dp.pkl` + `pole_axis_*.log`
  untracked/gitignored, machine-local. Re-run `pole_axis_build.py`
  (BGE-m3 encode, GloVe download) to regenerate.

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
