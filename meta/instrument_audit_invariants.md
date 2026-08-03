# Instrument audit — the invariant list

Ordered [3791].1(b). **This file enumerates and classifies; it does not test.** Testing is at the module owner's seat. Scope is RH's word: *"fixing and auditing our instruments we've already used"* — `movement.py`, `decompose()`, `cell_roles`, `contrast.py`, `cell.py`, the store read paths.

    79   | candidate invariant sentences, 4 modules |
           regex sweep for cannot/never/always/must/exactly/by-construction/
           refuses over comment and docstring lines, len>25 |
           2026-08-03
    18   | of them in movement.py (447 lines) | as above | 2026-08-03
     9   | cell.py (277) | as above | 2026-08-03
    14   | contrast.py (634) | as above | 2026-08-03
    38   | cache.py (1,204) | as above | 2026-08-03
     8   | total `assert`/`raise` statements across movement/cell/contrast |
           `grep -c "^\s*\(assert\|raise\)"` | 2026-08-03

**79 asserted invariants, 8 enforcement points.** That ratio is the audit's whole justification and it is not an indictment — most of the 79 are rationale, not claims. **Triage is the work.**

## THE CLASS THAT JUST COST US, AND IT DEFINES THE PRIORITY ORDER

The residual-as-faller defect had one signature:

> **A comment asserting a NEGATIVE BEHAVIOURAL CLAIM ("X cannot be Y"), with no runtime enforcement, where a violation is SILENT because a later strip repairs the VIEW.**

**Priority is therefore not by module and not by line count. It is: unasserted negative behavioural claims, silent on violation, first.**

## P0 — UNASSERTED NEGATIVE CLAIMS WHOSE VIOLATION IS SILENT

**P0.1 — `movement.py:194-199`. THE RISER STRIP IS STILL A STRIP-AFTER-COMPUTE.**

    # The RISER strip stays real: the bucket IS a non-faller and does carry a
    # null, so it can satisfy the riser rule -- but it is not a lexical event
    # and must not be reported as one.
    m.risers = [k for k in m.risers if k != RESIDUAL_KEY]

The faller side was fixed at candidacy ([3777]). **The riser side was not — it is the identical shape, one field over.**

> **THE QUESTION TO RUN, NOT READ: does riser selection involve any COMPETITION — top-k, ranking, a budget, a threshold computed over the candidate set — in which the residual's participation can DISPLACE a real word?** If selection is a per-word independent predicate, the strip is harmless and should still become an assert. **If there is any competition, the strip repairs the list and not the selection, and this is the same defect.**

Same treatment as the faller fix: exclude at candidacy, leave an assert where the strip was.

**P0.2 — `movement.py:175-176`. `null`, `excess` and `delta` are POPPED after computation.**

    for coll in (m.null, m.excess, m.delta): coll.pop(RESIDUAL_KEY, None)

Three more strips-after-compute. **Each needs the same question: was any quantity derived from these collections WHILE the residual was in them?** `decompose()` reads `m.excess` for `arrived`; the pop precedes that, so `arrived` is clean — **but "clean at the one call site I checked" is not the invariant.** Every reader of `m.null`/`m.excess`/`m.delta` inside `_movement` must be enumerated.

**P0.3 — `cell.py:214`. `AMBIGUOUS IS NOT "MIXED", AND allow_mixed MUST NOT COVER IT.`**

A flag covering more than its name is silent by construction: the caller passes `allow_mixed=True` believing they accepted two rules they can name, and receives a read that did not say which. **The comment calls `_check_versions` "the single choke point every compute path passes through" — a claim about REACHABILITY, testable directly and not currently asserted.**

**P0.4 — `contrast.py:269`. `A unit is a group holding EXACTLY ONE of each role.`**

Groups holding two of a role are the polysemous-field class this campaign has already been bitten by twice (`pair_role` returning 1,784 where 1,368 was registered). **Testable: construct a group with a duplicated role and confirm refusal.**

**P0.5 — `contrast.py:21`. `PAIRED IS DECLARED, NEVER INFERRED.`**

The failure mode is a pairing silently inferred from co-occurrence when `group_id` is absent — which is exactly the shape of the F21 worker/mgmt case the comment cites.

## **THE RULE THAT REPLACES "FIND EVERY STRIP-AFTER-COMPUTE" — WRITTEN AFTER P0.1 CLOSED AND P0.2 INVERTED**

P0.1 and P0.2 were listed on a resemblance: both are `pop`/filter of `RESIDUAL_KEY` after a computation, same shape as the faller defect. **Both were run. They came back opposite.**

    FALLER strip     COSMETIC over a CORRUPTED quantity.  `fallset` fed R, S
                     and `inflation`, so the strip repaired the VIEW and not
                     the number.  DEFECT -- moved to candidacy [3777].
    RISER strip      HARMLESS for selection: the predicate is per-word
                     independent, 0 of 400 cells differ.  Hygiene only.
    THE POPS         **LOAD-BEARING.** `movement():175-176` popping
                     `null`/`excess`/`delta` is THE ONLY THING keeping the
                     bucket out of `top_riser()`'s ARGMAX -- which feeds
                     `concentration = top / arrived`, a booked quantity.
                     The bucket WINS that argmax in 1.5% of cells under
                     CANONICAL and 4.0% under DRAW ([3798]).
                     **REMOVING THIS STRIP WOULD CREATE A DEFECT.**

> **THE SAME SYNTACTIC PATTERN IS A DEFECT OR A PROTECTION DEPENDING ON WHAT FED IN AND WHAT CONSUMES OUT. YOU CANNOT TELL BY LOOKING AT THE STRIP.**

    ASK BOTH SIDES:
      (in)   did the stripped key feed a GLOBAL quantity computed before the
             strip?   -> the strip is cosmetic; move the exclusion to candidacy
      (out)  does any consumer RANK or REDUCE the collection -- argmax, top-k,
             sum, sort, max?   -> the strip is protection; make it explicit
             and assert it, never remove it

**AND THE FRAMING ERROR IS THE LIST'S OWN, RECORDED SO IT DOES NOT REPEAT:** P0.1 asked *"does riser SELECTION involve competition?"* The competition was in the CONSUMER, `top_riser()`, one line away. **An invariant about a COLLECTION must be checked at every point that RANKS OR REDUCES it, not only where it is BUILT.** Producer-side questions find producer-side defects.

## P1 — POSITIVE BEHAVIOURAL CLAIMS, TESTABLE, MOSTLY CHEAP

    movement.py:300   "SUM, never overwrite"          -- two rows, one word
    movement.py:80    "1.0 (never binding)"           -- IS it ever binding on
                                                         the real corpus?  A
                                                         threshold that never
                                                         binds and one that
                                                         binds silently look
                                                         identical from inside
    cell.py:23        "THE PARTITION IS SUMMED, NOT OVERWRITTEN"
    cell.py:110       "ANSWERS, never raises"         -- ambiguous arm IS present
    cell.py:54/70     refuse-to-compute / never-refuse-to-describe, and
                      "A PREDICATE THAT GATES AN ACTION MUST REFUSE WHEN IT
                       CANNOT ANSWER"
    contrast.py:30    "always as a Counter and never silently"
    contrast.py:436   "The reference is ALWAYS the step's FULL scored population"
    cache.py:71       "true_word_probs ALWAYS emits mode"
    cache.py:241      "THE DTYPE COMES FROM THE KEY, NEVER A CONSTANT"

## P2 — ALREADY ENFORCED. VERIFY THE ENFORCEMENT, NOT THE CLAIM

    movement.py:193   assert RESIDUAL_KEY not in m.fallers        [3777]
    movement.py:274   malformed row NAMED, NEVER SKIPPED          [3770].N7
    cache.py:196      logits value is ALWAYS {file,row,dim}       TypeError
    cache.py:273      refuse on non-finite                        [3715].2(iii)

**For each: does the enforcement fire on a constructed violation?** An assert that cannot be reached is a comment with a keyword.

## WHAT THIS LIST CANNOT DO

- **It finds only invariants someone WROTE DOWN.** An invariant the code relies on and no comment states is invisible to a comment sweep, and there is no reason to think that set is empty.
- **It is a regex over prose.** It will have missed sentences phrased without its keywords; the recall is unmeasured and unmeasurable from inside.
- **It cannot rank by consequence.** P0 is ranked by SILENCE, which is a proxy. A loud violation of a load-bearing invariant may matter more than a silent violation of a cosmetic one.
- **Reading will not settle any P0 item.** The residual defect was three lines below the comment it violated and survived every reading. **Every item above is settled by RUNNING it or not at all.**
