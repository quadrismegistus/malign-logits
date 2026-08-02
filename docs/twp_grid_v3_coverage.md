# true_word_probs coverage — the 979-prompt run (`twp_grid_v3`)

    producer     scripts/ (this document was assembled by hand from the artifacts)
    producer_run malign, seat audit, 2026-08-02
    sources      data/twp_grid_v3/*.jsonl (95 files)
                 data/grid_run_manifest.json
                 data/grid_spec.json @ commit e5b42a3 (the roster the run used)
    stamp        2026-08-02T08:45 UTC

**Scope: the 2026-07-30 `twp_grid_v3_from_scratch` run on instance 46301965,
979 prompts per model. The August cloud run (2,583 prompts) is OUT OF SCOPE.**

---

## §0 HEADLINE

    ROSTER (spec @ e5b42a3)        103 models x 979 prompts = 100,837 planned
    COMPLETE at 979 rows            93
    PARTIAL                          2
    ABSENT ENTIRELY                  8
    ROWS ON DISK                91,421   =  90.7% of planned

**`data/grid_run_manifest.json` records `cells: 100837`. That is the PLAN, not
the achievement.** The arithmetic reconciles exactly: 93x979 + 144 + 230 =
91,421. **Anyone quoting 100,837 as a delivered count is quoting the roster.**

## §1 THE TWO PARTIALS, DISPOSITIONED

    allenai/Olmo-3.1-32B-Instruct-DPO     144 / 979    memory-bound 32B
    tiiuae/Falcon-H1-1.5B-Base            230 / 979    SSM / compute-bound

Both are the models `data/grid_spec_phase32b.json` and
`data/grid_spec_phasefalcon.json` were carved out for — the July split "by
BOTTLENECK not by vendor". **The phases were written; neither completed.**
`data/twp_phasefalcon/` holds one model at 70 rows; `data/twp_phase32b/` holds
two files.

## §2 THE EIGHT ABSENT, NAMED

    allenai/Olmo-3.1-32B-Instruct-SFT
    tiiuae/Falcon-H1-1.5B-Instruct
    tiiuae/Falcon-H1-7B-Base
    tiiuae/Falcon-H1-7B-Instruct
    tiiuae/Falcon3-Mamba-7B-Base
    tiiuae/Falcon3-Mamba-7B-Instruct
    tiiuae/falcon-mamba-7b
    tiiuae/falcon-mamba-7b-instruct

**Seven of eight are Falcon SSM/hybrid; one is the third 32B.** No file exists
for any of them — these are named gaps, not short files.

## §3 VALIDITY — SUCCESS IS VALID ROWS, NOT LINE COUNTS

Ten rows sampled per model (first, middle, last, plus seven seeded-random),
**950 rows across all 95 files**:

    unparseable rows                 0
    rows with a non-finite or out-of-range probability   0
    rule_version                     3 on 950/950   (uniform, no mixed-rule cells)
    conservation                     min 1.000000, max 1.000001

**No model has a single failing sampled row.** `conservation` is the producer's
own mass check and it holds to 1e-6 everywhere sampled.

**Stated as a sample, not a census:** 950 of 91,421 rows were opened (1.0%).
A defect confined to unsampled rows would not appear here.

## §4 COUNTS AT BOTH UNITS

    UNIT = model      roster 103   complete 93   partial 2   absent 8
    UNIT = lineage    roster  31   complete 27   lost 4

    lineages LOST ENTIRELY (no complete member):
        tiiuae/Falcon-H1-1.5B-Base      tiiuae/Falcon-H1-7B-Base
        tiiuae/Falcon3-Mamba-7B-Base    tiiuae/falcon-mamba-7b

    lineage PARTIALLY affected (some members complete, some absent):
        allenai/Olmo-3-1125-32B

**Every roster model resolves in the lineage map; there are no unmapped
models in this run.** The model-level loss is 10 of 103 (9.7%); the
lineage-level loss is 4 of 31 (12.9%) — **the loss is worse at the unit that
matters for paired analysis**, because the missing models cluster into whole
lineages rather than spreading across them.

## §5 WHAT THIS DOES NOT SAY

- **Nothing about WHY the eight failed.** No log from those attempts has been
  read. The July finding "Falcon needs KERNELS not a card" is a hypothesis
  carried forward, not a diagnosis performed here.
- **Nothing about the current cloud run**, which is a different prompt set
  (2,583) and a different roster order.
- **Nothing about whether 979-prompt cells are comparable to 2,583-prompt
  cells.** Same `rule_version` 3 and same `dict_sha` in both, but the prompt
  populations differ and no one has checked that the 979 are a subset.
