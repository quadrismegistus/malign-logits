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

- **M01 Z** (`Z_ladders_regimes.md`): `z_ladders.py`, zero write calls.
  §1's SFT/DPO cancellation and §5's closed-system comparison have no
  data behind them. REGISTRATIONS.md row Z says so.
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

DISCHARGED FROM THIS CLASS, for the record and the method:

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
([5503]).
