# M02 first look: exit markers across four designs — the scene, not the signifier

Written 2026-08-08 by the registrar seat with RH steering in session. Sources: the docket record
[4993]-[5008], the claims register (campaign block of 08-08), `scripts/exit_underscore.py`,
`exit_underscore_fc.py`, `exit_underscore_stats.py`, `exit_underscore_within.py`,
`exit_markers.py`, `exit_forced.py`, `results/*.csv`, commits 99b36e87..ca128a0e. STATUS: FIRST
LOOK THROUGHOUT — regex surface patterns over cached generations and beams, declared before
reading (TYPES copied verbatim from `y_exit_typology.py`; REFUSAL declared a priori, always
reported apart). Nothing here is a measurement to quote; the coded pass is the instrument of
record. Everything below was posted, audited across seats, and registered before this document
existed.

## 0. The question and the four designs

Does transgressive content elicit foreclosure symptoms — continuations that abandon the
fictional frame (quiz/cloze format, Q/A format, task reframes, assistant commentary, use-to-
mention collapse, editorial meta)? Four designs, in increasing order of control:

- **RAW BATTERY**: the 48 DEFAULT prompts (F01's battery), 190,261 passages at 100 tokens,
  107 checkpoints; transgressive-vs-neutral is BETWEEN-PROMPT and genre-confounded (RH's
  diagnosis) — demoted to descriptive pilot.
- **EDGES**: base -> derivative within lineage (Registry.base_of; 48 edges over 29 bases),
  matched prompts, per-edge interaction (transgressive delta − neutral delta), base-clustered.
- **TWINS**: beam_fc undisturbed, 105 stems x MARKED/UNMARKED, ~714k 10-token beams per side;
  the genre-controlled frame. Per-checkpoint delta; the checkpoint is the unit (RH ruling).
- **FORCED ARMS**: wave-3 LEXICAL forced beams only (design read from record VALUES, per
  [4996]; the earlier waves' function-word half excluded on RH's instruction) — faller vs
  riser at the same site in the same checkpoint.

## 1. The negative, closed: the cloze blank is not a transgression symptom

`___` tested in all four designs: raw battery REVERSED (neutral 3.53% vs transgressive
2.51-3.33%; Wilcoxon over checkpoint deltas p 0.0092, mean −0.82pp — a detection of the
CONTROLS' genre, not of foreclosure); twins NULL (47/66 checkpoints at exactly zero — floor
instrument); edges: post-base training suppresses it in EVERY domain including neutral
(uniform format cleanup); twin DiD null. Transgression does not elicit the blank. The marker
is a genre-and-stage variable. CLOSED.

## 2. The positive, thrice-surviving: Q/A conversion tracks the transgressive scene

**Twins (genre controlled): E-QA is higher at the MARKED member in aligned checkpoints**, and
survived three decoder rosters: +0.61pp p 0.016 (full, 26 nonzero) -> +0.56 p 0.037 (drop-8)
-> **+0.43pp p 0.035 (drop-12, fully clean)**. Positive in 6/7 domains; SEXUAL is the flat
exception — consistent with Y, where sexual sites get refusal rather than reformatting. Base
checkpoints lean the same way (+0.22, p 0.085, near-clean arm throughout). The magnitude to
quote is the clean one; the arc travels with it.

**Edges (independent design, independent population)**: the interaction puts the same types on
top — E-QA +3.99pp and E-ASSIST +0.77pp (both p 0.0002 base-clustered; survive Bonferroni x7);
E-MENTION +0.98 and E-QUIZ +2.63 at p 0.012 (do not survive). Composite shape: training
scrubs school-format from neutral prompts and adds assistant-frame commentary at transgressive
ones. GENRE-CONFOUNDED at this design — the twins are what make the direction claimable.

**Withdrawn along the way**: E-MENTION's aligned column died under the full 12-sampler filter
(p 0.148 at 8 nonzero; the base column stands, +0.015 p 0.011). The twin DiD (does alignment
WIDEN the gap?) is null everywhere AND is the one statistic with no decoder protection
([4994].3) — not claimable at this grain in either direction.

**Faces** (top cells, chosen to illustrate, rates printed in the docket record): Falcon3-3B-
Instruct generating its own `<|assistant|>` tag mid-narrative and refusing, in raw
continuation mode; MiniCPM5 converting an erotic scene into a multiple-choice comprehension
exercise about itself (0% base -> 75% aligned on that cell). The scene is not refused and not
continued; it is converted into a school exercise about the scene.

## 3. The forced arms: friction on the word, defense against the scene

The repair-work discriminator Findings W left unrun. Wave-3 lexical sites are the 210 twin
prompts (30 per domain, both members); fallers and risers are open-class verbs (said, told,
put / watched, whispered, began) — mostly the demotic-to-literary movers, so the test is the
general form "does uttering the DEMOTED SIGNIFIER expel the scene."

- **Faller vs riser: NULL on every type, every roster, and within BOTH twin members** (E-QA
  the largest at +0.31-0.48pp, p >= 0.11). Which word was forced does not matter downstream.
- **The scene effect SURVIVES forcing**: MARKED − UNMARKED on E-QA is +0.36pp p 0.0009
  undisturbed, +0.22 p 0.063 after a forced faller, +0.23 p 0.010 after a forced riser.
  Attenuated, not erased.
- **Forcing vs undisturbed: CONFOUNDED, demoted per [5026]**: the raw contrast (E-QUIZ
  +0.41pp, E-QA +0.33pp, both p < 0.0001) crosses a COMMITMENT BOUNDARY (a continuation that
  has committed to any word differs definitionally from a free one) and a ONE-TOKEN POSITION
  OFFSET (forced windows sit at sentence positions 2-11, undisturbed at 1-10). "Mildly
  destabilizing" overreads it; what it licenses is only the design consequence — the
  word-agnostic component is real enough that faller-vs-riser, which holds commitment and
  position constant, is the ONLY interpretable primary. Symmetric between arms either way, so
  it nets out of that primary.

**Read against the wave-3 damage state (register, 08-07/08)**: forcing the demoted word DOES
cost fluency — dd mean +0.0144 p 0.0043 on the rule-as-declared population, sign test 19/24,
median +0.0088 p 0.029 with bloom included (bloom convicted on the C1 conservation identity;
RH's characterization ruling mean/median/composite remains open). So the discriminator table
SPLITS: the cost channel is small-positive (the depth account's direction), the repair channel
is flat (chain-substitution's direction). The signifier carries a residual charge measurable
in probability but not in behavior — it is paid for and then absorbed, not expelled. The
sentence the day converged on: **THE FRAME APPARATUS READS THE SCENE, NOT THE SIGNIFIER** —
consistent with Findings V (the relation is local to the scene) and with Y (the apparatus
keys on the slot). The third channel — renewed displacement after the forced word — is now
the tiebreaker, unrun in data already collected.

## 4. Instrument facts bought (the reusable part)

- **The decoder arc** ([4994]-[5003]): fc's producer never pinned `do_sample`; 12 of 68
  beam_fc checkpoints beam-sampled under their own configs, 8:1 aligned at record level.
  Within-checkpoint contrasts (twins, faller-vs-riser) difference the decoder OUT; cross-arm
  contrasts (DiD, Y-style) have no protection. "A checkpoint config is only ever a hazard
  where a caller declined to decide." Per-record `decoder` field lands at malign's merge; the
  top-beam-as-mode reading in the M02 plan must not consume beam_fc before it.
- **REFUSAL at beam grain is dialogue apology**: 12/12 sampled base-arm hits at UNMARKED are
  in-scene `said, "I'm sorry, but ..." `. The pattern's beam-level rows are not refusal.
- **The visibility ladder is real**: E-ASSIST and E-META sit below 10-token reach (X.3g
  replicated); E-QA and E-QUIZ are beam-visible. fc beams cannot measure generation length
  at all ([5008]: zero early EOS in 189,000 beams — no room in ten tokens).
- **design lives in beam_fc VALUES, not keys** ([4996]); a key-level read sees one
  undifferentiated bucket.

## 5. Fences, standing

Regex typing misses paraphrased exits by construction and fires on in-scene dialogue that
quotes a question; direction of error unknown. The twin pass is 21 uncorrected tests read as
convergence with the edges, not as standalone findings. Forced sites carry the k>=2 mover
qualifier ([4817]). Ten tokens: a repair arriving at token 30 is invisible. n=34 pairs is not
34 independent lineages. Nothing here travels without "first look" attached.

## 6. What turns this into findings

1. RH settles the typology and blesses the battery plan frame -> 600-passage two-coder pilot
   -> the ~27k coded read (per-checkpoint, all 107, thin tier at own denominators).
2. **The 100-token twin generation** (210 twin prompts x FC roster x ~10 samples ~ 143k
   sequences, Y machinery, decoder pinned in the design string from birth) — the decisive
   instrument for the scene effect, the E-ASSIST question, and the repair-at-depth question
   at once.
3. The renewed-displacement channel on the forced beams already collected.
4. Archangel battery top-up (48 prompts x 8 samples x suite) whenever a fleet is up.
