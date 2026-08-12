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

- **M05 C Result 4** (registrar's own): riser-recapture (violence ~1,
  sexual 0.08–0.45). "recaptur" appears in exactly two files, both prose.
  Ad-hoc session computation; must be scripted into the
  `m05_pair_displacement` family and reproduced before quoting.
- **M04 A, the exploratory half**: per-index grid, +1-only four terms,
  long-window sweep — zero hits in `channel3_run.py`, nothing in
  M04/scripts. The long-window aggregation is not recoverable from the
  frozen slot spec ([5024].2) and MAY NOT BE RECONSTRUCTIBLE; the
  finding's own Robust/Not-robust split turns on it.

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
sequences, 99.98% text-complete); where a gate is really a
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
stripped fails invisibly, and the receiving seat never observed it).
