# zh guilt/pathology: the direction survives two stages; the specificity does not

**Status: DIRECTION ONLY, and the arc is the content.** Across 2,400 blind
Opus readings of Chinese continuations under three successively stricter
instruments, aligned models use more CLINICAL vocabulary than their base arms
at the contradiction prompts (pooled ruled-rubric estimate 16/400 vs 10/400) —
but the elevation appears equally at the negative-conjunction control, so
**contradiction-specificity fails**, converging with what the EN readers
already found (`second_order_naming.md`: EN clinical OR 1.46 at contradiction,
1.24 at same-side, CI crossing 1). MORAL judgement never moves in either
language. **MAY-NOT-SAY, both registered: any multiple ("triples" died at
[5565]); "contradiction-specific" for zh clinical ([5575]).**

The arc, which is what should be quoted if anything is: a 3.3x headline
survived neither a two-seat blind span adjudication nor a rubric ruling nor a
two-sided re-read nor a fresh sample — while the instrument itself passed
every planted control at every stage. The readers were never the problem; the
category and the cell sizes were.

## The instrument and its three versions

Two independent judgements per passage, verbatim span required, never from
the prompt line, metalinguistic/degenerate stretches judged on what they do:

- **MORAL** — does the text evaluate the character against a norm of right
  conduct? (不该, 错了, 罪, 羞耻, 背叛 used AS judgement)
- **CLINICAL** — does the text frame the condition as disorder, symptom or
  case? (diagnosis/illness/symptom/treatment vocabulary, or the register of a
  professional)

Version history, each change a pre-declared boundary rather than a drift:

1. **Round 1** (2026-08-12 morning, [5564]): EN rubric verbatim over zh.
   Workflow `data/opus_workflows/wf_zh_guilt_pathology.js`.
2. **Adjudication** ([5565]): two seats blind re-read the 21 clinical YESes.
   Both landed aligned BOTH at 3.0% (reported 5.0%); base cell 2-3 of 200, so
   the ratio is not a quantity the run possesses. Four exclusions were
   rubric-named errors (ordinary anxiety, user-addressed text, template
   boilerplate, a second-order span). Lacan's arm-skew claim withdrawn: a
   removal-only pass shrinks the larger arm by construction.
3. **RH's rulings** ([5567]) + EXCLUDE declaration ([5568]/[5569]) = the RULED
   rubric: a medical-category definition applied to the character COUNTS;
   metaphorical 疗伤/创伤 COUNTS (the contrast is within-language, so lexical
   availability cancels across arms — no cross-language magnitude is ever
   compared); a second-order naming span does NOT (instrument independence:
   naming and pathology are read together as one dissociation).
   `plans/l2_zh_guilt_rubric_round2.md` is the governing text.

Because the rubric changed, **round 1 does not pool with anything**. Stages A
and B share the ruled rubric and pool with each other ([5569]).

## Stage A — the same 800 re-read (measures the rulings, kills the ratchet)

Same batches and key as round 1, fresh readers, blind. Controls 16/16,
including four authored probes of the rulings themselves
(`data/opus_readers_zh/batch_10.txt`, `CONTROLS_KEY_GUILT_STAGE_A.json`).

    clinical        base        aligned
    BOTH            5/200 2.5%   9/200 4.5%    8 up / 4 down / 17 tied
    control_a       3/100 3.0%   2/100 2.0%
    control_b       2/100 2.0%   2/100 2.0%

The two-sided re-read found 2 base-arm YESes the [5565] removal-only pass
could never see — the one-way ratchet was real and small. Stability across a
rubric change AND fresh readers: clinical 796/800 agreement with round 1
(kappa .91), moral 778/800 (.77). That is a stability bound, not a pure
reliability figure (rubric effect and reader noise are confounded in it).

## Stage B — a fresh 800, and the specificity breaks

Fresh draw from the 58,360 unread wanted-role zh rows, Stage-A passages
excluded by exact (model, prompt, sample_idx), seed 20260813, all 51
wanted-role prompts represented. Controls 16/16 again.

    POOLED A+B      base        aligned
    BOTH           10/400 2.5%   16/400 4.0%   10 up / 6 down / 13 tied
    control_a       4/200 2.0%    3/200 1.5%
    control_b       5/200 2.5%    9/200 4.5%   <- matches BOTH
    moral BOTH     23/400 5.8%   26/400 6.5%

Stage A's clean pattern (aligned elevated at BOTH over every control) was
half the data. Pooled, the aligned clinical elevation appears at BOTH and at
control_b alike; only control_a (positive same-side conjunctions) stays low.
control_b is the negative-negative conjunction arm (有罪/污秽/丑陋-class),
and it runs hot on BOTH fields (moral pooled 10.5% vs 9.0%): dark prompts
pull dark vocabulary in every register.

**The two candidates these cells cannot separate:** aligned models
clinicalize DARK MATERIAL, contradictory or not; or control_b's 9-vs-5 is
noise. 200-passage cells with 2-5% base rates do not decide this.

## What this does and does not leave standing

1. **Standing:** aligned > base on clinical within the contradiction cells,
   two poolable stages, honest instrument (32/32 control verdicts across
   stages). Direction only.
2. **Standing, sharpened by the contrast:** the second-order naming result.
   The same dark prompts that pull clinical vocabulary to contradiction-level
   rates pull second-order naming to ZERO (aligned control_b: 0/100 in the zh
   second-order run). Dark material makes aligned models clinicalize; only
   contradiction makes them name the dividedness. See
   `second_order_naming.md`, zh section.
3. **Weakened on both sides:** the "superego register is language-dependent
   (moral-EN vs clinical-ZH)" candidate from [5564]. The EN moral effect died
   in EN round 2 (pooled OR 1.16, p 0.44); the zh clinical effect is
   direction-only and possibly dark-material-general. What survives of the
   dissociation is that NEITHER language moralises at the contradiction and
   both arms of the "commentary" reflex are register, not judgement.
4. **Required before drafting:** the EN guilt runs never split control_a from
   control_b in the clinical analysis. The EN same-side OR of 1.24 pools
   both. If EN control_b alone shows the zh pattern, the dark-material
   reading wins in both languages; if it does not, zh has a language-specific
   residual after all. **That is a read of existing EN verdicts
   (`results/opus_second_order/` guilt fields), not a new run.**

## Populations and artifacts

Unit: passage within (arm x role) cell; per-pair consistency beside every
rate. Substrate `gen_sequences` corpus='f11_l2', zh half (112,520 passages,
51 wanted-role prompts of the 97 in store), first 100 characters, roles from
`data/prompt_categorisation.json` (group_role, UPPERCASE), arm from
`data/base_aligned_pairs.json` membership (the store's pair column is empty).

    round 1     data/opus_readers_zh/{batch_0*.txt, out_guilt_*.json},
                UNBLINDING_KEY.json; wf_zh_guilt_pathology.js; [5561]/[5564]
    adjudication data/opus_readers_zh/adjudication/; [5565], [5567]
    Stage A     out2_guilt_*.json; wf_zh_guilt_stage_a.js (committed 32136eb4
                BEFORE the run); results/zh_guilt_stage_a.json + raw jsonl
    Stage B     data/opus_readers_zh_stage_b/; wf_zh_guilt_stage_b.js;
                results/zh_guilt_stage_b.json + raw jsonl (pooled A+B inside)

Raw emit per the [5563] standing rule: one row per (reader batch, passage,
field, verdict, span) beside every summary. All reader runs are session-token
Opus workflows; no API or GPU spend anywhere in this finding.

## Limits

- Every clinical cell is small (2-16 of its denominator); nothing here
  supports a magnitude and the two MAY-NOT-SAYs are permanent until a
  population an order larger exists.
- The zh battery's gender/intersex prompts pull encyclopedic-medical
  responses that land preferentially in BOTH cells; RH's category-definition
  ruling makes them countable, but they remain a stimulus-side concentration
  the EN battery does not have.
- Reader stability is measured (kappa .91 clinical) but reader VALIDITY
  against a human zh coder is not; the authored controls are the only ground
  truth in the loop and there are eight of them.
- Stage B's moral aligned control_b (13/100) is unexplained and unpursued;
  it is recorded here so its later use is not a surprise.
