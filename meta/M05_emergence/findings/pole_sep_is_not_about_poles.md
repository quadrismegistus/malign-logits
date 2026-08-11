# The pole-separation arc is not about poles, and its floor is at step 256

**Status: a NULL that discharges an owed debt, plus one positive dating result.**
The cross-group null A-R4 records as owed is run: the collapse-and-recovery arc
appears just as strongly between prompts that share no opposition at all, so it
is not a pole phenomenon. Separately, Pythia's eleven sub-1000 rungs locate the
floor at **step 256**, where OLMo's two-point segment could only assert step
1000 — and OLMo has already recovered 2x by the time its first non-zero rung is
taken.

Producer `scripts/m05_pole_sep_pythia.py`, importing `geometry()` from
`m05_pole_sep` rather than reimplementing it. Artifacts
`results/m05_pole_sep_pythia.csv` (154 rungs, 21 en f11 groups, 106,722 rows)
and `results/m05_pole_sep_crossgroup_null.csv` (13 checkpoints, 99,099 rows).
Both are re-reads of hidden sidecars already on disk; the full ladder took 16
seconds.

---

## 1. THE NULL: the arc is present where there are no poles

`pole_sep` is computed from a group's own two pole prompts and is IDENTICAL
across role — max |both − control_a| = 0.00 over 1,521 cells — so the same-side
conjunction controls give no comparison. Nothing in `m05_pole_sep.csv`
established that the trajectory was about poles.

The null pairs `pole_a` of group X with `pole_a` of group Y: same arithmetic,
same model, same layer, two prompts that are merely different rather than
opposed.

    checkpoint                     real    cross-group null
    OLMo stage1-step0             0.795          1.396
    OLMo stage1-step1000          0.227          0.526
    OLMo stage1-step16000         0.475          0.805
    Pythia step0                  0.347          0.748
    Pythia step128                0.115          0.270
    Pythia step16000              0.384          0.701

**The null collapses and recovers exactly as the real column does**, on both
lineages. Whatever the arc is, it is what training does to the distance between
any two distinct prompts.

**THE LEVEL GAP LICENSES NOTHING.** The real column sits below the null
throughout (0.34–0.64x) and that is expected from surface overlap alone:
within-group prompts share **0.750** of their words by Jaccard, cross-group
**0.273**. One word differs inside a group. The null is not matched on lexical
distance and a matched one would be a different, harder build.

The single divergence worth keeping: at trained checkpoints the real column has
more depth structure than the null (layer spread 13–18x against 4.9–10x). That
is the only surviving candidate for something pole-specific, and it inherits the
same overlap confound.

## 2. THE FLOOR IS AT STEP 256, AND THE COLLAPSE HAS AN ONSET

OLMo's first non-zero rung is step 1000, so its collapse is one line segment.
Pythia has eleven rungs below that.

    step        median pole_sep, final layer
       0        0.3470   flat --
       1        0.3470            four rungs identical to 4 dp
       2        0.3471
       4        0.3464   --
       8        0.3232   first movement
      16        0.2258
      32        0.1187
      64        0.1342
     128        0.1150
     256        0.0741   <- FLOOR over the whole 154-rung ladder
     512        0.0758
    1000        0.1543   already 2x recovered
   64000        0.4700   peak
  143000        0.3691

At the group unit, step 0 → step 256: **21 of 21 groups fall, sign p = 9.5e-07.**

**Nothing happens for four rungs, then it falls 4.7x over five doublings.** The
onset sits between step 4 and step 8 — roughly 8–16M tokens at Pythia's
2,097,152 tokens/step. **OLMo's apparent floor at step 1000 is an artefact of
where its first rung falls**: by step 1000 Pythia is already twice its floor
value, so the OLMo segment was measuring the recovery, not the collapse.

The layer-spread diagnostic tracks it: **1.7x and flat through step 4** — the
random-projection signature from [5445] — widening to 3.0x at the floor and
8.0x by step 16,000 once real depth structure exists.

## 3. CROSS-INSTRUMENT, ONE LINEAGE, DESCRIPTIVE

malign's [5430] puts an eight-fold rise in words-per-cell between steps 8 and
128. **That is the same window as this collapse** (0.323 → 0.115). Two
instruments on the same rungs of one lineage dating the same interval. Recorded
as descriptive and cross-instrument; nothing causal, and one lineage each.

## What this does to A-R4

`A_acquisition.md` Result 4 currently fences the arc: *"A cross-group null … is
owed before any pole-specific reading; until then the 'spread → collapse →
re-separation' arc is not quotable in any form."* **The null is run and the
fence should become a strike**: the arc is real, replicated on a second lineage,
now dated to an onset between steps 4 and 8 with a floor at 256 — and it is not
about poles. It belongs to whatever training does to representation distance in
general.

That is registrar's paragraph to amend, not this seat's; this document is the
add-beside.

## Limits

**The null is not lexically matched** (§1) and a matched null is the obvious
next build: pairs of prompts differing in one word but not opposed. Until then
the level comparison is uninterpretable and only the SHAPE comparison is used
here, which is unaffected by a constant offset.

**Two hidden-state stores disagree.** Recomputing the committed
`m05_pole_sep.csv` gives 5e-07 agreement under `prefer="fleet"` and 6.8e-4
under `prefer="wider"` — same model, same prompt, different bytes. Everything
here uses `fleet`. Which store a geometry column came from is a parameter and
should be recorded with it.

**308 of 3,234 Pythia cells had no hidden state** and are absent rather than
imputed.

**One lineage each**, and the two lineages are not aligned on any axis here;
§3's window agreement is within Pythia only.
