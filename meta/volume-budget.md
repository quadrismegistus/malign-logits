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

## The rule this file exists to enforce

**A protective copy is cheap individually and the volume is a shared resource, so check it against the headroom rather than assuming it free.** Both volumes are above 94 percent full and the spare on the only viable migration target is about 5 GiB. That is not headroom with room for an unplanned second copy.

Companion rules live in `CAMPAIGN.md`: *non-destructive is a property of an action in isolation, not of two of them*; and *a protective copy goes to a name no other seat could choose* (an announcement is a coordination protocol and fails exactly under concurrency; a constructed name cannot collide).
