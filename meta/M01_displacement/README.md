# M01 — Displacement: what alignment does to the words it will not say

**The map.** What the campaign is asking, and which registration sits where. **No statuses and no hashes live here** — they go stale the moment work happens, and they have their own file.

    REGISTRATIONS.md   THE RECORD — every registration by letter, whether it
                       ran, what came back, which artifact holds it
    ledger.md          THE HISTORY — supersessions, withdrawals, what a number
                       may not say, and the clause list as the campaign's
                       pre-registration vocabulary
    **this file**      THE MAP — the questions, and where each piece sits

**The one-sentence claim: alignment does not delete the transgressive lexicon; it redistributes it.** Suppressed probability mass migrates to substitutes — kill → scream — and the migration has structure.

**Everything below is a summary and none of it is quotable on its own.** The registration governs its own result; the ledger governs its caveats.

---

## The two measured axes

Both axes ask the same three questions: **in general, at transgressive sites, and across languages.**

### Axis 1 — Does alignment substitute? (movement)

*Does alignment produce a statistically significant change in word probability, and does the departing mass land on other words rather than vanish?*

| | | |
|---|---|---|
| **in general** | **N** | 2,199 English stimuli × 44 edges. **Substitution confirmed**: the tail gives mass up to nameable words rather than absorbing it, in 91% of cells, with all 34 clusters agreeing. |
| **at transgressive sites** | **F**, **G** | Displacement at transgressive sites is **not more frequent** (F, rate null) **but is larger when it happens** (G, magnitude confirmed). The pair is the point: F alone would have read as "nothing here." |
| **across languages** | **O**, H1 | **Confirmed in both arms.** The redistribution is not an artifact of English — bounded to Chinese-trained models on Chinese text (see the limits). |

**The structure of the redistribution** — how concentrated the receiving set is, and whether families agree on direction — is measured but unregistered. It lives in the `m01_concentration` producer and in ledger clauses 3 and 5.

**And the two rows of this axis do not yet meet.** N answers *in general*, F and G answer *at sites* — on different statistics and different units, so no sentence of the form "N times stronger at transgressive sites" is available. **Registration Q is drafted to close exactly that; see the bridge below.**

### Axis 2 — Does alignment change word-norm semantics?

*Of the words that move, do the falling and rising ones differ in valence, arousal or dominance — and is the change targeted or general?*

| | | |
|---|---|---|
| **in general** | **C**, **E** | **Valence de-extremification is general** — aligned output moves toward affective neutrality across the corpus, not only at transgressive sites. **Dominance is dead as a general effect.** E carries the same question onto the gap stratum. |
| **at transgressive sites** | **D**, **D2**, **D3b** | **Both extremity arms confirm at sites** (D2): valence and dominance de-extremification concentrate at the transgressive member of a minimal pair. **Arousal is movement-general, not site-targeted** (D — a quotable null). **Sweetening is unsupported.** D3b brackets how much of D2's effect the pool's own extremity could account for: *"just reflects pool extremity"* is dead; mediation stays open. |
| **across languages** | **O**, H2/H3 | **Not supported.** Both arms confirm in English and come back as clean coin-flip nulls in Chinese. **O cannot separate "the affective mechanism is absent in Chinese" from "the Chinese norm join is too thin to see it."** |

**Dominance splits by frame** — dead in general, confirmed at sites. That split is itself the finding.

**The same gap sits under this axis:** C is the general corpus, D2 is the minimal pairs, and they share machinery with no common scale.

### The bridge — Registration Q (drafted, not frozen)

*Both axes ask their three questions and neither can compare its own answers.* **Q puts general and site on one instrument, one population, one scale** — N's 2,199 English stimuli × 44 edges, four measures per cell, with the partitions read out of that single pass: the marked and unmarked halves of the 684 minimal pairs, against the transgressive and neutral prompts of the general corpus.

Three declared contrasts. **`tail_excess` at sites, paired on the pair** — the campaign's own substitution claim, never once tested where it matters most. **`tail_excess` transgressive-vs-neutral in the general corpus, paired within cluster** — the bridge itself. And **the norm statistics estimated with a stated MDE and never tested**, because their minimum detectable effect sits at the size of the effects this campaign has actually measured. That last is Registration C's control arm exactly, and **the only difference is that Q knows beforehand.**

**Q is a draft posted for review. It has never been frozen and has produced no quantity.**

---

## Three axes the 2 × 3 does not hold

### Axis 3 — What *is* the substitution? (the relation)

**The question the campaign is named after — and as of 2026-08-04 it has a registration: P, frozen and pending its run.** *Displacement* names a relation, not a rate: what is `scream` to `kill`? A speech-act shift, a metonymy, an attenuation, a register change?

Four geometric instruments — WordNet similarity, contextual cosine, inverted syntagmatic JS, mass-winner percentile — **all fail to locate a relation a reader sees instantly.** That is a verified instrument-failure record and nothing more.

**Registration P is the instrument that could replace it with something positive.** Frozen 2026-08-04: 4,443 items over 685 prompts, coded by three model families one pair per call, with a *paired* primary — each risen word against a stationary decoy drawn from its own faller, so the comparison holds the faller fixed and varies only whether the word moved. Two strata carry primaries (speech-act shift in ACT slots, metonymy in REFERENT slots); everything else is descriptive. **CONFIRMED requires all three coder families; two of three is reported as a SPLIT with the dissenter named — and that sentence was written before anyone knew the answer.**

**It has not run**, and until it does the relation has no positive characterisation. **The flagship pair is not in it:** `kill → scream` clears the drawing threshold on merit at 18 edges and is excluded solely because it was spent on tuning, recorded in the deposit as an exhibit so a reader can see exactly what was given up.

P also carries the stratified test that clause 7's slot-sensitivity has been waiting on. **One unrun pass gates both.**

### Axis 4 — When is it learned? (training stage)

*Does the operation install at SFT or at DPO, and does repression precede displacement?* Two ledger clauses, **no registration, and the stronger of the two is currently unreproducible** — the SFT-installation result has never been regenerated and its cause was never found.

### Axis 5 — What does alignment do to a human author's word?

**L** and **M**, and they fit nowhere in the 2 × 3 because the comparison is not base-against-aligned at a site — it is **model against a human writer.** Given found prose, does the aligned model still hold the word the novelist actually wrote?

**It does not: alignment loses the human's word**, on all three rungs of the ladder (L). **And the mechanism is boundary proximity, not a contracting tail** (M): eviction concentrates entirely where the word was barely retained, and is exactly zero above the fifth headroom decile.

---

## The validity layer, which is not an axis

Some registered work exists to attack the campaign's own results rather than extend them, and it attaches to a claim rather than sitting beside it:

- **The renormalisation null** — is the redistribution genuine, or an artifact of the distribution renormalising after suppression? Verified once on one family, **dormant**: the exact test needs a second full campaign of logits.
- **The threshold correction** (inside N) — built to destroy N's result if it were an artifact of where the faller cut falls. It moved 0.13% of cells and left the statistic identical to six decimals.
- **The pool confound** (D3b) — could the affect result be the transgressive pool's own extremity rather than alignment's doing?
- **M adjudicating L** — a registration whose entire purpose is to decide between two readings of another registration's gradient.

---

## Two limits that bind every sentence on this page

**There is no bridge quantity between "in general" and "at sites."** C and D2 are built from the same contrast machinery, but no statistic puts them on a common scale, so **the two rows of each axis cannot currently be compared numerically.** Anything of the form "the effect is *N* times stronger at transgressive sites" is unavailable.

**"Across languages" currently means one registration and one language pair.** O's nine surviving clusters are all Chinese-origin or Chinese-heavy — the one non-Chinese lineage was removed by the competence filter — so a crosslingual confirmation licenses **"the mechanism appears in Chinese-trained models on Chinese text"** and never the general claim. And magnitudes never compare across O's arms: the two languages' valence scales differ in dispersion by a third, so "the effect is larger in Chinese" would report a lexicon difference as a model difference.

---

## What is open

- **Registration P's run** (axis 3) — frozen, its producer built and cleared at two seats, **not run**. It waits on RH's explicit word and nothing else fires it. Until it does, nothing about the relation can be said positively, and clause 7's slot-sensitivity waits on the same pass.
- **The whole of axis 4** — no registration, and the SFT-installation claim needs reproducing before it can be cited at all.
- **Registration Q** — drafted, posted for review, **never frozen**. It carries the norms half of the live-roster sweep (the movement half is discharged by N and O) *and* the general-vs-site bridge, and three questions sit at RH's seat: an arbitrary per-cluster floor, whether to re-derive G and D2 as known answers at all, and a measure listed with no hypothesis.
- **C's one blind arm** (general sweetening), never emitted.
- **D3b's negative slope** — more extreme pools appear to displace *less*, which is the opposite of what the confound predicts. A described fit awaiting a registered surrogate.
- **The bloomz question** — a model whose base is Chinese-competent and whose aligned child is not, same tokenizer, one step. **Alignment took its Chinese away.** Deposited as n=1; a ten-lineage table is authorized.
- **Four audits owed** on findings the ledger's clauses rest on, and two producer repairs whose consequence is that every Stouffer Z in N and O is a floor rather than a value.

*Where this page and `ledger.md` differ, the ledger governs; where the ledger and the docket differ, the docket governs. For whether a registration ran and what it returned, `REGISTRATIONS.md` governs both.*
