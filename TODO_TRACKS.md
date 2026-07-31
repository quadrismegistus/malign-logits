# Project tracks — parallel work and the findings register

MAINTAINED BY: registrar (the pen). Created 2026-07-31 at RH's request.
Organized BY BLOCKER so work parallelizes. (`TODO.md` is malign's
historical work log and is untouched; three live items from it are
carried into the register below.)
Seats: correct your own track on the docket; the pen edits the file.
Marks: [ ] open, [~] in flight, [RH] awaits RH's word, [GATED: x].

## Track R — RH's decision queue (nothing moves these but his word)

- [RH] ANNOTATION WORD (Option A, ~$15-45): gates M01
  `faller-riser-relation` + `slot-sensitivity` positive forms AND M02
  assembly. RH's own sequencing: after the prior clauses' producers
  land — measurably close. STANDING REMINDER: the SEQUENCE-rate
  prediction registers on the docket BEFORE any coder pass executes.
- [RH] NORMS-INSTRUMENT WORD: word-level norms (arousal / concreteness
  / dominance, en+zh) as exogenous instrument; recon agent running.
- [RH] F34 / F35 META-HOMES: cross-linguistic displacement and
  architecture independence have NO meta home (see register) —
  M-clause, book-only, or defer.
- [RH] Cursor-provenance docket fix (maude builds on RH's word).
- [RH] LLM-GENERATED NORMS as a SECOND instrument (pointer from the
  abstraction seat, 2026-07-31): Martinez/Conde/Reviriego family
  (GPT-4o; en 126K words + 63K MWEs, es 128K, de 185K w/ GPT columns
  BESIDE seven human sources in Conde2026.xlsx — a ready-made
  human-vs-model comparison; Trott 2024 r=.81 vs human concreteness).
  Excluded from the CURRENT exogenous instrument by design (would
  dissolve the exogeneity argument, mirror of abstraction's firewall);
  candidate question: does the displacement gradient look the same
  under human affect ratings and a model's own semantic priors?
  Book-native (LLM aesthetics). RIDER ([1196].3): inherits the
  function-word exclusion, AND reports the source's willingness to
  score closed-class words as a validity finding in its own right.
  Provenance: abstraction seat's memory
  project_llm_generated_norms_rejected.md. ANSWERED 2026-07-31: the
  English set carries AROUSAL + VALENCE + concreteness, and the
  single-word AFFECT columns are the directly validated arm (Study 2,
  benchmarked against Warriner) — so the second instrument could test
  affect, not only abstraction. Design sketch (abstraction seat's):
  run the gradient under Warriner on the ~14K overlap, under GPT norms
  on the overlap AND the 112K extension — separates human/model
  divergence from coverage artifact. English lacks familiarity (es/de
  are the four-variable sets). EN FILES PINNED (abstraction seat,
  OSF-verified against published digests): osf.io/k5a4x, CC BY-NC-SA;
  single-words sha256 878aa7b6cb44625e (126,397 rows, osf.io/download/
  mkghz), MWE 8710700f29f2ae50 (63,680 rows, 6p8sn). ERRATUM TRAP:
  arousal in the single-word file was MISASSIGNED in every copy
  downloaded before 2025-01-13 (publisher erratum kv8ay; caught by
  Westera, Leiden) — ONLY the pinned digest is arousal-valid; any
  other copy is POISONED for arousal. Schema per dimension:
  X_GPT_dom / X_GPT_prob / rank / rank% — X_GPT_prob is an EXPECTATION
  over the model's rating distribution, the closest object-type to
  this project's own instrument. Caveats: master list "includes some
  faulty entries (mainly from Hollis)" — budget a cleaning pass
  pre-join; model-version drift (list gpt-4o-2024-08-06, studies
  -05-13); two stray empty columns. DELIVERED at
  TheoryMachines/norms_sources/llm_martinez/ (SEPARATE from the
  exogenous human sources by design): both xlsx + Erratum.pdf
  (d48550a5, traveling WITH the data) + PROVENANCE.md (leads
  NOT-A-HUMAN-SOURCE; quotes the erratum; poisoned-copy warning;
  all caveats attributed). Hashes verified IN PLACE post-copy.
  NEW DEFECT (abstraction seat): the two files disagree on column
  naming — MWE file has Valence_GPT_probs / Arousal_GPT_probs
  (PLURAL) but Concreteness_GPT_prob (singular); single-words file
  singular throughout — a naive shared loader silently drops the MWE
  affect columns. Any future spec inherits this warning. Decision
  AFTER the current audit closes.
- Queued small items (RH: "still queued"): register-fidelity judge;
  literary-below-neutral design; eleventh-clause candidate.

## Track C — compute (the boxes; they land on their own)

- [~] 46383750 twp-32b: Olmo-3.1-32B SFT+DPO (1,958 cells) ->
  completes the only unfinished 3-arm family; enters stage-share.
- [~] 46383753 twp-falcon: Falcon x8 (7,832 cells) -> F35 breadth,
  scale-ladder top rung, falcon-H1 lineage toward claim (B).
- [ ] On landing: ingest -> registry regeneration -> counts re-pin
  (producers refuse-on-drift, then re-run — by design).

## Track M — malign (parallel-ready now; roughly in order)

- [ ] EXECUTE THE RETIREMENT PACKAGE ([1051].2 ratified; [1128]
  sequence): by string, duplicate sweep, F36 four included. Unblocks
  the re-freeze and everything behind it.
- [ ] Custody commits: m01_concentration.py @ 705789fa,
  m01_direction_agreement.py @ 30d8a9f1.
- [ ] SECOND-SEAT NUMBER CHECKS: concentration, direction-agreement
  (post-re-freeze numbers only, [1128].3).
- [ ] C3 DE-TRANSGRESSION COUNT CONFIRMATION — outstanding since
  [1038].2.
- [ ] M03 C2 second-seat verification (lexical mechanism, one-seat
  since [1015].2).
- [ ] Licensed-set re-verification under settled defaults (12/16
  single-seat, [1037].3).
- [ ] Ladder counts FROM THE ARTIFACT -> claim (B) wakes
  self-executingly ([1122].3).
- [ ] Anger audit ([714].2) — recipient-agreement's remaining flag.
- [ ] Numeric-reference remainder ([1048].3: docs/object_layer.md:221,
  f40_staged_massflow.py).

## Track L — lacan (parallel-ready now)

- [GATED: retirement execution] Emit blind + diff + pin update; re-run
  both producers on the re-frozen population; post hashes + numbers.
- [ ] STAGE-SHARE SUCCESSOR spec draft ([942]/[959]/[1107]: 21-family
  v3, absolutes + distributions, declared floor reading, ACTIVE
  population, corrected js(); frozen-population regime). Draft -> pen
  freeze -> malign audit.
- [ ] instrument_commitments.md entries 9-10 — confirm committed.
- [GATED: RH norms word] Norms-instrument spec draft.

## Track P — the pen

- [x] This file (2026-07-31).
- [ ] Register below kept current; M01/M03 rows as rulings land; M02
  stub current.
- [ ] After annotation: M02 assembly (the EXITED-rate adjudication is
  its spine).

## Gated chains (why the parallelism has this shape)

    retirement exec (M) -> re-freeze (M+L blind) -> producer re-runs (L)
      -> custody (M) -> 2nd-seat checks (M) -> concentration +
      direction-agreement VERIFIED -> [RH gate] ANNOTATION WORD
      -> coder passes (after the SEQUENCE-rate prediction registers)
      -> faller-riser + slot-sensitivity positive forms -> M02 assembly
      -> the CI article's §IV core fully provisioned

    boxes land (C) -> olmo-32b enters stage-share; F35 gets its data;
      (B) may gain the falcon-H1 lineage

## Findings -> meta register

Completeness derives from the zero list: every finding gets a
DISPOSITION; "unprovisioned" is a decision, not an accident.
M0x = provisioned there; STUB-GATED = in a stub with a named gate;
BOOK = book-chapter material, no M-claim needed; RH? = needs RH's
call; DEAD = rescoped/downgraded, no further work. BOOK and RH? rows
are PEN-PROPOSED, not decided. Under [1118].2 (no re-verification of
old results), DEAD and BOOK rows get no audit time unless a draft
cites them.

| Finding | Status/grade | Disposition |
|---|---|---|
| F01 logit analysis | unaudited C | M01 `mass-migration` (audit day supplies its number via producer) |
| F02 cross-family logits | rescoped C | DEAD (superseded by v3 word-level instruments) |
| F03 cross-family generation | unaudited C | BOOK (ch7) |
| F04 step analysis | unaudited C | M01 `acquisition-order` (audit pending) |
| F05 logit lens | rescoped D | DEAD |
| F06 baseline validation | unaudited C | M01 `liminal-targeting` lineage (via F40) |
| F07 training-data attribution | unaudited C | BOOK (ch1/ch2) |
| F08 displacement taxonomy | rescoped C | M01 historical source (superseded by annotation schema) |
| F09/F10 tulu ablations | unaudited C | M03 (which-data ablations; [569].3 transfer) |
| F11 contradiction + addendum | rescoped B / verified A | M02 STUB-GATED on the annotation (EXITED rate) |
| F12 fold geometry | retained-downgraded C | DEAD by RH ruling ([1051].3); later articles only |
| F13 jakobsonian axes | rescoped C | M01 clauses 6-7 + the axes apparatus (ci_paper_notes §5) |
| F14 syntagmatic baseline | rescoped C | M02 STUB (combination damage) |
| F15-F18 passage/corpus/MMD/entropy | unaudited C | BOOK (ch6/ch10) — incl. TODO.md's cloud scale-ups, which stay parked |
| F19 BOS entropy | rescoped C | M03-cited method note only |
| F20 family (drift, third-person, who-are-you) | verified A / rescoped | M02 STUB (frame battery) + BOOK ch7 |
| F21 + addendum | solid-by-design B / verified A | M03 core (C1 IN FORCE) |
| F22 circuit decomposition | unaudited C | BOOK (ch5/ch6); RH? if the paper cites |
| F23 reasoning distillation | unaudited C | BOOK (ch8) |
| F24 pretraining emergence | unaudited C | BOOK (ch2); RH? if the paper cites |
| F25 temporal signature | rescoped C | DEAD |
| F26 census | rescoped D | DEAD (superseded by the registry) |
| F27 nudging negative | unaudited C | BOOK (negative result, ch4) |
| F28 resistance trajectories | rescoped C | DEAD |
| F31 permanova | rescoped C | DEAD (superseded by circuit work) |
| F32 template-mediated | solid-by-design B | BOOK (ch4 method) |
| F33 scale effects | unaudited C | BOOK; feeds claim (B) when it wakes |
| F34 cross-linguistic displacement | unaudited C | **RH? — no meta home; the [897].2 Chinese predictions await evaluation** |
| F35 architecture independence | unaudited C | **RH? — no meta home; the boxes are computing its data NOW** |
| F36 family (capstone, violence, euphemism, ledger) | verified A / rescoped B | M01 clauses 6-7 sources + M02 three-zone |
| F37 judges | — | M03 (write-up debt; freeze on RH's call) |
| F38 pipeline/confirmation | — | BOOK (ch4/ch5); campaign memory |
| F39 preference-corpus insensitivity | verified B | M03 |
| F40 discovered vocabulary | unaudited B | M01 `liminal-targeting` |
| F41 word norms | registered — | REGISTERED pre-data ([1147]); P1 -> candidate M01 clause `arousal-descent`; P3 -> gendered work; TEMPLATE for the pointer convention |

Carried live from TODO.md (malign's log): publication figures for new
findings; the one-row-per-family summary CSV for drafting; everything
else there is done or parked.
