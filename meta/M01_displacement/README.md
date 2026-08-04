# M01 — Displacement: what alignment does to the words it will not say

A reader's map of the campaign, organised by question. Every claim here is a summary; the authoritative record — full statuses, caveats, supersessions, docket citations — is `ledger.md` (this folder), and the quotable shapes with their MAY-NOT-SAYs live in the TheoryMachines claims register (`notes/claims-register.md`, section A8). Frozen designs are in `registrations/`, result artifacts in `results/`, populations in `populations/`.

**READ SECTION 5 BEFORE QUOTING ANYTHING FROM SECTIONS 1–3.** This page states each finding in its strongest honest form; section 5 states how far each has been *verified*, and the two are not the same. **Not one of the ten numbered clauses is both fully verified and independently audited.** One (clause 9, the SFT-installation result) is UNREPRODUCED with its cause unlocatable, and appears below because the ledger still carries the claim — not because it stands. Registrations F, G, N, O, D, D2, D3b, L and M are a different matter: each is frozen, run, and two-seat, and their statuses are given where they appear.

The campaign's one-sentence claim: **alignment does not delete the transgressive lexicon; it redistributes it** — suppressed probability mass migrates to substitutes (kill -> scream), and the migration has structure: in degree, in affect, in site, and in training time.

---

## 1. How MUCH does alignment displace? (degree)

The instruments: `movement.py:decompose()` — the canonical decomposition of a site's probability movement into fallers, risers, and their shares — and, at its simplest, the count of fallers and risers per site.

What is established (ledger clauses 1, 2, 3, 5; registrations F and G):

- **The mass migrates rather than vanishing** (clause 1 — *PENDING its F01 audit day*), and the migration is ~92% genuine rather than renormalisation artifact (clause 2 — ***DORMANT***: *verified two-seat at its time, but the exact full-vocabulary null needs real logits at v3 coverage, which is a data block rather than a writing block; RH declined the re-verification campaign ("Let's not waste time verifying old results"), so **no producer will lift this** — and the clause **scopes honestly to one family***).
- **It concentrates**: the top recipient takes a median 0.381 of gained mass — but only modestly above a random split among qualifying receivers (1.135 vs the Dirichlet null), and the receiving set is small (median 3-20 risers by family). Both halves quote together or not at all (clause 3).
- **Direction is shared across families beyond independence at every denominator tested**, but there is no typical rate — which sites compel shared direction is itself the phenomenon (clause 5 — *measured, all spec gates passed, producer committed, and **REPRODUCED AT A THIRD SEAT in a clean environment** — every headline figure lands. What is outstanding is the **independent-arithmetic** bar, deliberately deferred to drafting need: reproduction checks that the code makes the numbers, independence checks that the numbers are the quantity*).
- **At transgressive sites specifically: displacement is NOT more frequent** (Registration F: rate null, n=33 pair-sites, p=0.148) **but IS larger when it happens** (Registration G: magnitude, d=0.748, p=0.00006).
- **SUBSTITUTION CONFIRMED AT FULL SCALE** (Registration N, 2026-08-04, frozen and run once, blind): on all 2,199 English stimuli x 44 edges — 82,775 analysed cells — **91% show negative tail excess: the tail gives mass UP to nameable words rather than absorbing it.** All 34 base clusters agree; the adversarial correction built to destroy the result if it were a threshold artifact moved 0.13% of cells and left the statistic identical to six decimals. Quote the Stouffer Z (+47.9) as a FLOOR, not a value (the producer's p-to-z conversion saturates; the p-values are exact). English only, by design; the claim is direction and consistency, never "large."

Coverage: the general corpus AND the pairs AND (as of N) the full live English roster. Clauses 1-5 ran on the general prompt rosters (959x95 and 975x93 populations); F and G ran on the 684 minimal pairs; N ran on 2,199 x 44. Degree is the best-covered question in the campaign.

**Does the pair structure give us "site-specific vs general" for free (neutral member = general)? No.** The unmarked member of a minimal pair is a matched twin of a transgressive prompt — same frame, same syntax, one word different — not a sample of language in general. The pairs answer "at the site versus one word away from the site"; the general corpus answers "in general." They are different questions, which is why the campaign carries both instruments — and the general corpus is itself broader than the pairs in kind, not just size (a third of its non-pair remainder is cross-lingual, a sixth deontic-framed; the pairs contain neither).

## 2. What happens to the SEMANTICS of what is said instead?

### 2a. Word norms: does alignment neutralise affect?

The norms are arousal, valence, dominance (well-studied psycholinguistic scales). The question: does aligned output move toward affective neutrality, and is the movement targeted at transgression or general?

The current supported picture, four registrations deep:

- **Valence de-extremification is GENERAL**: confirmed on the 959-prompt corpus, all sites (Registration C, arm H2 — A +0.025 residualised, p=0.0012, beats its benchmark).
- **And it is STRONGER at transgressive sites**: both valence-extremity and dominance-extremity de-extremification concentrate at the transgressive member of minimal pairs (Registration D2: D +0.0151 / +0.0166, both p<0.01 at split alpha).
- **Dominance splits by frame**: dead as a general effect (C, arm H3) but confirmed at sites (D2) — the site-specificity is the finding.
- **Arousal is movement-general, NOT site-targeted** (Registration D: a quotable null — the design could have seen the known general effect and saw nothing site-specific).
- **Sweetening (negative words fall, positive rise) is unsupported at sites** (D, H1); its general-corpus arm is the ONE C arm still blind (valence/signed/GENERAL, never emitted).
- **None of this is vocabulary availability**: transgressive prompts do offer more extreme valence vocabulary, but that difference is bounded ABOVE at 56% of the valence effect and at nothing on dominance (wrong sign), and the pair-level relationship runs opposite to the confound's prediction (Registration D3b, the bracket decomposition).

Two cautions the register enforces. **No quantitative "general vs site" ratio yet** — C's and D2's statistics are built from the same contrast machinery, but a bridge quantity on a common scale would need its own registration. And **C's own displacement-specificity question is OPEN, not answered**: its control arm's minimum detectable effect (0.0390) *exceeded the effect it was testing* (0.0251), so that null could never have detected an effect of the size at issue — an underpowered null is not evidence of absence. D2 addresses site-specificity by a different route; it does not retroactively power C's control arm. C's first conjunct (the signed riser term) is evaluated but **its value is sealed at RH** and is unauditable while sealed.

### 2b. The displacement relation itself: what is scream to kill?

Four geometric instruments — WordNet similarity, contextual-embedding cosine, inverted syntagmatic JS, mass-winner similarity percentile — all fail to locate the relation a reader sees instantly (ledger clause 6). So the measurement moves to blinded, decoyed human/LLM judgment: the coding instrument exists at `malign_logits/tasks/code_displacement_relation.py` (schema frozen at docket [637] as amended; two-axis vocabulary — speech-act shift vs metonymy, with intensity orthogonal). **Built, never run: the annotation pass is the campaign's largest unfinished item.** Until it runs, "the relation is interpretive, not geometric" is a verified instrument-failure record with the positive characterisation pending. **Two things the shorter statement of this item omits: the run needs a SECOND CODER (the ledger's open dependencies list it as pending), and it gates TWO claims, not one — 2b's relation here, and clause 7's slot-sensitivity positive form in section 3.** One unrun pass is the bottleneck on both.

### 2c. Does the operation cross languages? (Registration O)

The first crosslingual instrument (frozen and run 2026-08-04): 301 ratified English-Chinese translated pairs — the pair holds content fixed and varies only language — over the 9 edges whose models are both tokenizer-capable AND behaviorally competent in Chinese (a capacity filter, then a competence filter with a blind-ruled threshold). Three hypotheses, each requiring BOTH arms.

- **Substitution crosses: SUPPORTED IN BOTH ARMS.** English 86% of cells (8 of 9 clusters); Chinese 95% of cells (9 of 9 clusters) — the redistribution mechanism is not an artifact of English. **The frozen bound travels with every quotation: all nine surviving clusters are Chinese-origin or Chinese-heavy, so this licenses "the mechanism appears in Chinese-trained models on Chinese text," never the general not-a-property-of-English claim.**
- **The affective signature does not follow — or cannot be seen: valence and arousal de-extremification are NOT SUPPORTED crosslingually.** Both confirmed strongly in English and came back as clean coin-flip nulls in Chinese (splits 0.4992 and 0.5023 — null, not reversed). By the reading rule frozen before any number existed, neither may be reported as an English finding of O. **And O cannot separate "the affective mechanism is absent in Chinese" from "the Chinese norm join is too thin to see it"** (the Chinese arm yields a scoreable statistic on half its cells against English's 70%); that fork travels with the nulls.
- **Magnitudes never compare across arms for valence** (the two languages' norm scales differ in dispersion by 1.32x — a pre-run rider forbids reading the difference as a model difference); signs compare, and every registered test is a sign test.
- **An incidental finding, descriptive, n=1, deposited:** the competence filter's one exclusion is bloomz — a model whose BASE is Chinese-competent (56% of retained mass on Chinese word forms) and whose ALIGNED CHILD is not (6.7%), same tokenizer, one alignment step on an English-heavy instruction corpus. **Alignment took its Chinese away — the capacity survived, the disposition did not.** A ten-lineage base-vs-aligned table is authorized as the ground for a possible registration ("does alignment strip non-English mass across lineages"); that question could not be asked inside O, whose filter selects on exactly the quantity it would test.

## 3. WHERE does the operation live, and WHEN is it learned?

**This is the campaign's least-verified section and the statuses below are not decoration.**

- **Site structure**: families converge on the same substitute site-specifically at real suppression sites (clause 4 — *partly discharged: the flagship site **REPRODUCES at word level (25/42, against a booked 24/45 at token level — different instrument, different roster, same answer)**, so UNRETESTED-PENDING-V3 is **discharged for that half**, single-seat. The rest of the clause still awaits v3, which changes what a word is, and the anger audit is still owed*); the operation is slot-sensitive — where the grammar admits both plan and discharge, alignment chooses discharge (clause 7 — *PARTIALLY VERIFIED; the fist/voice control is blind-coded and the referent structure measured, but the full stratified test awaits the annotation run, and the **cessation-operator leg is defect-flagged and UNCITABLE**: its two carrier pairs place the manipulated word last, so the scored position reads a continuation of the manipulation rather than the manipulation. Re-measurement is a design change, not a truncation*); category-specific targeting holds at liminal sites and fails at explicit ones, where the drain is largest but undifferentiated (clause 8 — *its source finding F40 is B-graded and unaudited; the audit is scheduled behind the draft-cited findings*).
- **Training stage**: repression precedes displacement in training — the model learns what it cannot say before it learns what to say instead (clause 10 — *PENDING its F04 audit day*).
- **The SFT-installation claim does not currently stand.** Clause 9 — that the operation installs almost entirely at SFT, in Amber, the only family with all three arms stored — is **UNREPRODUCED, CAUSE UNLOCATABLE**. The attempt to regenerate it predates the cache ingest and the cause was never found. It is stated here because the ledger still carries the claim and a reader will meet it in older drafts; **it is not evidence for anything until it reproduces**, and any sentence putting the operation's installation at SFT rests on it alone.

## 4. What is open, by choice or by pendency

- **The annotation run for 2b (built, unrun) — still the campaign's largest unfinished item, and it needs a SECOND CODER.** The kill-to-scream relation has no positive characterization until it runs, **and neither does clause 7's slot-sensitivity: one unrun pass gates both.**
- **Four audits owed on already-stated clauses, none of them new work in the sense of new experiments:** the F01 audit day (clause 1), the F04 audit day (clause 10), the F40 audit (clause 8), and the anger audit (clause 4). Until they are held, four of the ten clauses on this page are stated on unaudited findings.
- **Clause 9 needs reproducing before it can be cited at all**, and its cause was unlocatable when tried. **Clause 7's cessation leg needs re-measuring** as the design change it actually requires. **Clause 4's remaining half needs v3.** **Clause 2 is not on this list**: it is DORMANT by RH's decision, blocked on data rather than on anyone's writing, and reviving it is a fresh decision rather than an owed task. **Clause 5 is not either**: it is reproduced at a third seat, and its outstanding independent-arithmetic bar was *deferred to drafting need*, not left undone.
- C's one blind arm (general sweetening) — a single-arm opportunity, not a battery.
- Wave 2 on the live roster: **the movement half is discharged** (N is claim 1 at full scale on the live English roster; O is a wave-2-shaped registration on the translated pairs). **The norms half remains open** — C-style questions re-registered on the full roster rather than the recovered 959.
- The negative slope from D3b (more extreme pools -> LESS displacement): a described fit awaiting a registered surrogate before it can be a finding.
- **The bloomz question (new, from O's filter):** does alignment strip non-English mass across lineages? The observation is deposited (n=1, descriptive), the ten-lineage table is authorized, the registration awaits RH's word — it needs its own design because O's competence filter selects on the quantity it would test.
- **The shared-predicate declaration (campaign-wide):** 1 of 22 registrations states the faller rule its quantities rest on; O now states its own (§O0), and RH's morning choice is one frozen declaration cited by all versus per-registration amendments. The pin must carry the candidacy rule and repair commit, not constants alone.
- Artifact-vs-mechanism for the pool gap, and the entanglement of "transgression" with the swapped word's own extremity: open by design; this corpus cannot decide them.
- **Two instrument repairs on the registration producers**, both one line, neither affecting a verdict: the `p`-to-`z` conversion shared by N's and O's producers saturates at |z| = 8.3265 (`0.5*(1+erf(x/√2))` underflows to exactly zero below x ≈ −8.33), so **every Stouffer Z those two registrations report is a floor**; and the same file needs a guard against a family saturating *against* its hypothesis, which the conservatism argument does not cover. Until the first is fixed, N's +47.9 and O's +20.7 / +25.0 quote as floors and never as values.

## 5. How verified is each clause?

**Sections 1–3 state the findings; this states their evidential standing.** Compressed from `ledger.md`'s status column, which governs.

| Clause | What it says | Standing |
|---|---|---|
| 1 | mass migrates rather than vanishing | **PENDING** — F01 audit day scheduled, not held |
| 2 | ~92% genuine, not renormalisation | **DORMANT** — verified-at-its-time, blocked on DATA not writing (no producer clears it), RH declined the re-verification campaign, **scopes to one family** |
| 3 | concentration, with its Dirichlet null | verified, with a re-denomination pending; tokenizer-comparability scope |
| 4 | site-specific convergence on a substitute | **PARTLY DISCHARGED** — flagship site reproduces at word level (25/42), single-seat; rest awaits v3; anger audit owed |
| 5 | shared direction beyond independence | measured, all gates passed, producer committed, **reproduced at a third seat**; the **independent-arithmetic** bar is deferred by decision |
| 6 | the relation is not geometric | verified **as an instrument-failure record only**; positive form pending the annotation run |
| 7 | slot-sensitivity | **PARTIAL**; cessation-operator leg **defect-flagged and uncitable** |
| 8 | category targeting at liminal, not explicit | source finding F40 **B-graded, unaudited** |
| 9 | the operation installs almost entirely at SFT — **IN AMBER, the only family with all three arms stored**, and nothing about any other family | **UNREPRODUCED, cause unlocatable** |
| 10 | repression precedes displacement | **PENDING** — F04 audit day scheduled, not held |

**Row 9's scope is part of the claim, not a caveat on it.** The ledger records that scope being *added* on 2026-07-30 because the clause "had stated a single-family result as general." A statement column that drops it re-commits the defect the standing column is warning about — and a reader trusting the standing column still reads the claim from the left.

**The registrations are the campaign's verified layer and they are a different standard**: F, G, C, D, D2, D3b, L, M, N and O are each frozen before the numbers existed, run once, and two-seat — designs registered in advance, producers audited by a seat that did not write them, artifacts sized-verified after. Where this page carries a strong claim, it is usually a registration carrying it; where it carries a numbered clause, read the row above.

**What this table is not**: I read these statuses out of the ledger's own status column rather than re-deriving any of them. **That makes this a faithful summary of a record, not an independent check of the work** — the same distinction the campaign draws between custody and verification.

**One known staleness, flagged rather than silently corrected here:** `ledger.md`'s Registration D section still reads "frozen, awaiting its battery." D ran — `result_d_stage1/stage2.json` and its D2 and D3b siblings are on disk and two-seat. The ledger governs this page, so the correction belongs there and not in a fix applied downstream of it.

---

*Every clause above compresses a ledger entry that carries its own caveats; where this page and `ledger.md` differ, the ledger governs, and where the ledger and the docket differ, the docket governs. Nothing here is quotable without its ledger caveats attached.*
