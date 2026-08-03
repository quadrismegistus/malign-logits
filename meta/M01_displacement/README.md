# M01 — Displacement: what alignment does to the words it will not say

A reader's map of the campaign, organised by question. Every claim here is a summary; the authoritative record — full statuses, caveats, supersessions, docket citations — is `ledger.md` (this folder), and the quotable shapes with their MAY-NOT-SAYs live in the TheoryMachines claims register (`notes/claims-register.md`, section A8). Frozen designs are in `registrations/`, result artifacts in `results/`, populations in `populations/`.

The campaign's one-sentence claim: **alignment does not delete the transgressive lexicon; it redistributes it** — suppressed probability mass migrates to substitutes (kill -> scream), and the migration has structure: in degree, in affect, in site, and in training time.

---

## 1. How MUCH does alignment displace? (degree)

The instruments: `movement.py:decompose()` — the canonical decomposition of a site's probability movement into fallers, risers, and their shares — and, at its simplest, the count of fallers and risers per site.

What is established (ledger clauses 1, 2, 3, 5; registrations F and G):

- **The mass migrates rather than vanishing** (clause 1), and the migration is ~92% genuine rather than renormalisation artifact (clause 2).
- **It concentrates**: the top recipient takes a median 0.381 of gained mass — but only modestly above a random split among qualifying receivers (1.135 vs the Dirichlet null), and the receiving set is small (median 3-20 risers by family). Both halves quote together or not at all (clause 3).
- **Direction is shared across families beyond independence at every denominator tested**, but there is no typical rate — which sites compel shared direction is itself the phenomenon (clause 5).
- **At transgressive sites specifically: displacement is NOT more frequent** (Registration F: rate null, n=33 pair-sites, p=0.148) **but IS larger when it happens** (Registration G: magnitude, d=0.748, p=0.00006).

Coverage: the general corpus AND the pairs. Clauses 1-5 ran on the general prompt rosters (959x95 and 975x93 populations); F and G ran on the 684 minimal pairs. Degree is the best-covered question in the campaign.

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

Caution the register enforces: no quantitative "general vs site" ratio yet — C's and D2's statistics are built from the same contrast machinery but a bridge quantity on a common scale would need its own registration.

### 2b. The displacement relation itself: what is scream to kill?

Four geometric instruments — WordNet similarity, contextual-embedding cosine, inverted syntagmatic JS, mass-winner similarity percentile — all fail to locate the relation a reader sees instantly (ledger clause 6). So the measurement moves to blinded, decoyed human/LLM judgment: the coding instrument exists at `malign_logits/tasks/code_displacement_relation.py` (schema frozen at docket [637] as amended; two-axis vocabulary — speech-act shift vs metonymy, with intensity orthogonal). **Built, never run: the annotation pass is the campaign's largest unfinished item.** Until it runs, "the relation is interpretive, not geometric" is a verified instrument-failure record with the positive characterisation pending.

## 3. WHERE does the operation live, and WHEN is it learned?

- **Site structure**: families converge on the same substitute site-specifically at real suppression sites (clause 4); the operation is slot-sensitive — where the grammar admits both plan and discharge, alignment chooses discharge (clause 7); category-specific targeting holds at liminal sites and fails at explicit ones, where the drain is largest but undifferentiated (clause 8).
- **Training stage**: the operation installs almost entirely at SFT — in Amber, the only family with all three arms stored (clause 9); and repression precedes displacement in training — the model learns what it cannot say before it learns what to say instead (clause 10).

## 4. What is open, by choice or by pendency

- The annotation run for 2b (built, unrun).
- C's one blind arm (general sweetening) — a single-arm opportunity, not a battery.
- Wave 2 on the live 2,579-prompt roster: new registrations by the campaign's own population rules (B/C's registered population is recovered, deposited, and disjoint from the pairs — but C's instrument is read; nothing re-opens).
- The negative slope from D3b (more extreme pools -> LESS displacement): a described fit awaiting a registered surrogate before it can be a finding.
- Artifact-vs-mechanism for the pool gap, and the entanglement of "transgression" with the swapped word's own extremity: open by design; this corpus cannot decide them.

---

*Every clause above compresses a ledger entry that carries its own caveats; where this page and `ledger.md` differ, the ledger governs, and where the ledger and the docket differ, the docket governs. Nothing here is quotable without its ledger caveats attached.*
