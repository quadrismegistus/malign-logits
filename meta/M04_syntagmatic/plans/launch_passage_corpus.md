# LAUNCH PLAN — the domain-general passage corpus

**Step 5 of the freeze state machine ([5527], [5533]). NOTHING IN THIS DOCUMENT SPENDS ANYTHING.** The spec is frozen; this is how the run is executed, written before a box exists so that the decisions are visible while they are still cheap to change.

    SPEC        meta/M04_syntagmatic/registrations/spec_passage_corpus_105.md
                FROZEN at e541f6a4, file sha256_16 27b1369efdd9dc0e
                (verified at this seat against both the commit and the working
                tree; identical, no drift)
    POPULATION  data/forced_arms_46reps_drmatch.json   89eb642b50d00dd9
                pairs 46      8567e0ee993b457b
                prompts 208   27484f7ade774b77
                cells 8,169   aa5389c1420c7f76   (7,309 with a matched control)
                matched       723e81b3946b6d56
    RUNNER      scripts/vllm_y_run.py --manifest ... --pair-index N
    COST        $104, 1,307,040 sequences

---

## 1. GEOMETRY, DERIVED FROM THE FROZEN TABLE RATHER THAN ASSERTED

Every number here is computed from `forced_arms_46reps_drmatch.json`, not carried from a prior post:

    pairs                        46
    checkpoints                  92        (2 per pair)
    cells                     8,169
    cells per pair          12 / 192 / 199  (min / median / max)
    sequences  = cells x 5 arms x 2 roles x 16 samples
                          1,307,040        matches §5 exactly
    generation at 33.8 seq/s   10.7 GPU-hours

**The cells-per-pair spread is the fact that shapes the shard plan.** One pair carries 12 cells and most carry ~192, a 16x range, so **balancing on pair COUNT would produce badly uneven boxes**. The partition below balances on SEQUENCES with pairs kept whole.

## 2. THE PAIR IS THE UNIT THAT CANNOT BE SPLIT

Cross-scoring happens inside one process with both checkpoints resident, so a pair cannot be divided across boxes. This is the same rule that governed the L2 and M05 fleets and it is why the shard is a set of pairs, never a set of cells.

**Shard by checkpoint, not by cell** (spec §5): the term that does not shrink with sampling depth is the 92 checkpoint downloads at 15-30 GB each. Wall clock, not dollars, is the constraint.

## 3. SHARD COUNT — THE TRADE-OFF, AND THE RECOMMENDATION

Longest-processing-time-first packing on sequence count, pairs kept whole:

    N boxes   pairs/box   ckpts/box   max seq    gen h    imbalance
        4       11-12       22-24     329,280     2.71       1.9%
        6        7-8        14-16     227,680     1.87       7.2%
        8        5-6        10-12     175,840     1.45      12.6%
       10        4-5         8-10     144,960     1.19      15.2%
       12        3-4          6-8     118,400     0.97      21.5%

**RECOMMEND N=6.** Generation falls to 1.87 h on the longest box and each box acquires 14-16 checkpoints. Past N=8 the generation term stops being the binding one — download does not shrink proportionally, imbalance climbs past 12%, and **every added box is another chance to lose one in provisioning**, which is the failure the L2 fleet actually suffered (6 of 14 lost, `project_l2_corpus`).

I am recommending, not choosing. RH's call, and N=8 is defensible if wall clock matters more than provisioning risk on the day.

## 4. PREFLIGHT — BEFORE ANY BOX LOADS ANYTHING

**`scripts/f11_l2_preflight.py --pairs <roster>` against `data/model_load_environments.json`.** Non-negotiable, and the reason is in CLAUDE.md: an environment tag is not a cause, and three known classes fail at different layers (`OLMoE`'s `histc` is MPS-only, `mpt-7b`'s repo is gone, `deepseek`/`croissant`/`Teuken` mangle the prompt in the TOKENIZER, which no card changes).

**The corpus outranks the record**: a checkpoint with a complete output file works here whatever any prior observation predicts.

Anything learned on a box goes back into `data/model_load_environments.json` and `docs/cloud_runbook.md` **in the same session**. Absence of an observation is not evidence of success.

## 5. REPORTING — CELLS WRITTEN, NEVER CELLS ATTEMPTED

**The health loop reads what was WRITTEN.** `grep '"rows": [{'` on the output, plus a record count. Three "ALL MODELS COMPLETE" messages in one day meant zero cells (`feedback_failure_that_looks_like_progress`).

    healthy      cells written since last check > 0, and rising
    SUSPECT      "complete" in 0.3 min having produced nothing — an orphaned
                 vLLM engine holding the card. THE DANGEROUS FAILURE LOOKS
                 LIKE FAST PROGRESS.
    discriminator  pgrep + a record count. "Instance running" is the rental,
                 not the work.

Retrying is correct for a RACE and never for a STATE (runbook §2.13). A box that does not respond twice gets diagnosed, not retried.

## 6. TRANSFER AND CUSTODY

- One rsync loop per box, pulling continuously, not at the end.
- Local free space stays above the **20 GB floor**; check it every cycle.
- A box is done when a RESTART offers zero cells — not when it says it is done.
- **Destroy only after byte-level verification that the data is local.** RH's standing permission covers destruction on that gate and no other.
- Record the `machine_id` at acquisition, so a failure has a durable handle after the instance dies (`feedback_durable_handle`).

## 7. WHAT LANDS, AND WHERE

Per (pair, prompt, arm, role): 16 passages at 256 tokens, t=1.0, cross-scored both directions. Ingest to the stash, then ClickHouse via `ch_read.py`, then the declared analyses in spec §6.

**The four §3 disclosures travel with every write-up:** the 36.5% control change, QUOTATION-ADJACENCY (never "dialogue rate"), the fifth arm present on 85.6% of cells, and `class_match` disagreeing on 42%.

## 8. DECLARED SECONDARY — THE DE-CURLIFICATION TEST, RIDING FREE

@registrar's [5535], adopted here rather than as a separate addendum so there is one document. **PLAN-SIDE: the frozen spec is untouched and nothing new is collected.** This is an analysis of text the run produces anyway, and it costs nothing.

**Why it is declared now.** [5477]'s quote finding was withdrawn at [5524] because its character class bundled U+2019, the apostrophe. What survived was an observation, not a result: aligned models appeared to emit less curly typography and more ASCII quoting. **This corpus is its designed substrate** — 46 pairs, domain-general, base and aligned arms at passage length — and declaring the test before the data exists is the difference between a designed test and a post-hoc read of a run bought for something else.

    PREDICTION, declared before any generation
      straight_dq   aligned > base
      curly_dq      aligned < base
      curly_sq      aligned < base
      U+2019        aligned < base
    REFUTED BY
      all four moving TOGETHER in the same direction, which would make it a
      general punctuation-frequency effect and not a normalisation
      straight_dq falling with the rest

    MEASURE   per-arm rates per 1,000 characters, SPLIT BY CHARACTER and
              NEVER POOLED (the [5522] discipline, which is the entire reason
              the parent finding fell)
    UNIT      pair; independence at the lineage, one vote each
    ARM       primary read on UNDISTURBED, which carries no injected word.
              The forced arms are reported beside it as description.

**One confound this design already controls, and it is worth saying why the pair unit does the work here.** Whether a model can emit curly quotes at all is partly a tokenizer fact, so a cross-model comparison would confound typography with vocabulary. **Base and aligned share a tokenizer within a pair**, so the within-pair delta cannot be a vocabulary artefact. That is a stronger guarantee than the parent finding ever had.

## 9. THE COOL-OFF STARTS WHEN THIS COMPLETES

RH's ruling, and it is a design constraint rather than a budget note: **what this corpus does not collect is not collected for an unbounded period.** That argument restored the fifth arm and it weakens every "revisit later" written anywhere in the spec. If something is missing from this plan, the moment to say so is before the boxes spin.
