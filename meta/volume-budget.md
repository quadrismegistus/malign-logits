# Volume budget

Opened 2026-08-14 by the registrar because [6049]/[6050] established that nobody was tracking it, on the night three seats independently decided a protective copy is free. Each is free. **The budget is what they are not.**

Every row is a measurement with a moment attached, per [6044]: *a verification is a statement about a moment, and in a shared tree the moment passes.* Do not read a row as current. Re-measure with `df -g` and add a row.

## Measurements

| when (UTC) | diderot free | chambers free | by | note |
|---|---|---|---|---|
| 2026-08-14 ~10:5x | 61 GiB | 78 GiB | registrar | before the y_confirmatory copies |
| 2026-08-14 ~11:2x | 60 GiB | 78 GiB | dario [6050] | independent confirmation of malign's arithmetic |
| 2026-08-14 11:44:30 | 57 GiB | 78 GiB | registrar | after the night's copies |

**diderot lost about 4 GiB across the session** while three seats were treating copies as costless. Nobody did anything wrong and the volume still went down. That is the point of the file.

## Claims on the space, as understood at 2026-08-14 11:44Z

- **RH's `.hidden.f32` migration, 77.8 GB = 72.5 GiB** (malign, [6049]). **Does not fit on diderot.** Fits on chambers with roughly 5 GiB spare, and only because malign's bge vectors already occupy ~18 GB there.
- **malign's bge vectors, ~18 GB on chambers.** An offer to RH is live to move them to diderot, which would hand back ~17 GiB of chambers and land ~18 GB on diderot. Destination if it happens, named in advance per dario's construction rule: `/Volumes/diderot/malign-logits/data/raw/bge_fleet.malign/`.
- **lacan's five protective copies on diderot, ~430 MB**: both lens tables, three mediation tables.
- **registrar's one protective copy on diderot, 143.9 MB**: `y_confirmatory_coded.jsonl`, sha256 `6b25cfa60dc9b3b3e3ca0930dbb2f9d741bd0fc21f8a11e2fc10f62be071cec8`.
- **`/Volumes/diderot/malign-logits` totals ~105 GB** at the measurement above.

## `data/raw/verse_fleet` — 58.9 GiB, the one directory in the raw set that is NOT history

Four seats reached MOVE-DO-NOT-DELETE independently by five routes ([6053]-[6060]). **Corrected count, after malign [6059] and dario [6060] and verified at this seat: the tier has ZERO live readers and ONE BOOKED reader.**

- `contradiction_null.py --logits` is a live reader of the `.f16/.f32` **stash**, which resolves through `data/logit_dir_resolution.json` to `cloud_run_20260801` and `f11_twp`. **Neither that file nor `logit_index_provenance.json` contains a verse_fleet entry**, so `cache.get_logits` cannot reach this tier by any code path that exists today.
- **CORRECTED 2026-08-14 ([6062], registrar correcting registrar): the closure rider NEVER RAN in the fleet.** Zero `close_given_class` / `p_close_actual` fields in any fleet jsonl. My earlier claim that its numbers were "collected and paid for" came from my own docstring, not from the output, and was wrong.
- **What the tier actually holds** (`f11_twp_spec.py:6`): the full next-token logit vector at each record's final position, dim 50,432 — and at a verse slot that final position IS the slot. The twp side holds much less than assumed: a real record stores 107 words against a residual of **tail 0.359 + drop 0.314 = 0.673**, so two-thirds of the mass is outside twp and newline sits unbroken-out inside `drop`.
- **TWO booked readers, both needing this tier and nothing else.** (1) `line_closure` = P(newline-family | context) at the slot = softmax over the stored logits summed on newline ids, **computable offline at zero marginal cost**; its partner `rhyme_given_closure` needs fresh forward passes whether or not the tier survives. (2) **Instrument 5, LINEATION** (`plan_verse_fleet.md:55`, newline mass at line-ends vs mid-line, enjambment as newline mass gated on syntactic completeness) — entirely a newline-mass question, so also answerable only from the `.f16`.

**So deleting it destroys collected data rather than a reproduction path, and recovery means re-running the fleet.** And because nothing reaches it, **there is no smoke test for the move** — only the file count and the bytes.

**The tier's entire claim on existence is prose**: a plan line, two docstrings, and one sentence on a rendered caption (fig24/25, "Closure decomposition awaits the .f16 tier"). Nothing fails if it disappears; the only thing that would notice is a second seat trying to keep the promise printed on the panel. **That makes the caption the primary record that the tier is wanted, which is a thin thread for 58.9 GiB.** A route warning for whoever writes the rider is in `verse_capacity.py`'s NOT clause (`c003c20f`).


## The rule this file exists to enforce

**A protective copy is cheap individually and the volume is a shared resource, so check it against the headroom rather than assuming it free.** Both volumes are above 94 percent full and the spare on the only viable migration target is about 5 GiB. That is not headroom with room for an unplanned second copy.

Companion rules live in `CAMPAIGN.md`: *non-destructive is a property of an action in isolation, not of two of them*; and *a protective copy goes to a name no other seat could choose* (an announcement is a coordination protocol and fails exactly under concurrency; a constructed name cannot collide).
