---
status: current
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-13
role: finding
topics: [surface-accounting, clause-architecture]
description: "Plans A and B at verdict grade: ALIGNED PROSE IS LESS LEXICALLY DIVERSE (A.H2 REVERSED, p .003, survives its own conditioning table) and packs MORE dependent clauses per 1,000 words into SHORTER clauses (p .002 / p .028) while every per-sentence ratio sits flat — compressed subordination, exactly the surface the per-sentence denominators cannot see. Sentence length itself: not supported. Single run, no cross-seat audit."
---
# Findings A/B: the style of alignment is compression, not simplification

**Status: FIRST FULL-RUN VERDICTS, single analysis pass, no cross-seat
audit yet. One registered hypothesis REVERSED significantly and survives
its own decision rule; one registered pair resolves as "one distribution,
two surfaces" with the moving surface identified; sentence length is a
non-finding.**

Population: the passage corpus's undisturbed arm (plan A Amendment 2),
238,381 passages over 42 pairs after row dedup; hardened stratum = prose
AND non-degenerate AND English (Amendments 4/5; screens computed from
stored text: top_word_share >= .20 | non_ascii_alpha_share >= .20 ->
degenerate; english_nltkwords_share >= .60 -> English), 197,186 passages
(all Ns synced to the corrected verdicts JSON after the [5705]/[5706]
flags-dedup fix; malign's counterfactual check: all 13 verdicts hold).
Unit: (pair, prompt) cell mean per arm; aligned minus base; pair medians;
sign test over pairs, Wilcoxon beside (results/m06_verdicts.json). Pooled
prose reads beside every stratified one. Producer m06_verdicts.py at the
commit of record; measures from m06_style.py (OSP instrument, commit
56f2562, [5698]-verified twp-free path irrelevant here — Stanza parses).

## 1. A.H2 REVERSED: aligned prose is LESS lexically diverse

Registered: aligned HIGHER windowed TTR. Found: LOWER, significantly, and
robust to the registered decision rule.

    ttr_mattr_w100   Δmed −0.0263   10 up / 29 dn of 39   p_sign 0.0034
    ttr_mattr_w50    Δmed −0.0169   12 up / 27 dn of 39   p_sign 0.024
    pooled prose     Δmed −0.0258   10 up / 31 dn of 41   p_sign 0.0015
    within sents-per-window tertiles (Amendment 1 rule):
      t1 −0.0230 (11/28) p 0.0095 · t2 −0.0250 (10/29) p 0.0034 · t3 −0.0222 (11/28) p 0.0095

The Amendment-1 coupling worry (windowed TTR moved by sentence length,
not diversity) is answered by its own table: the contrast holds INSIDE
every tertile of sentences-per-window. Window-fit rates by arm are in the
JSON and did not decide anything (A.H1 being null, the differential-
missingness scenario never engaged). **DE-DIVERSIFICATION: alignment
narrows the working vocabulary of generated prose.** (Anti-conflation
fence stands: this is TTR over text — never read against next-token
concentration, drift, or entropy without a measured bridge, [5670].)

## 2. A.H1 NOT SUPPORTED: sentence length is a non-finding

    sent_len_words_mean  Δmed −0.859   17/24 of 41   p_sign 0.35
    len_words            Δmed −1.799   18/23 of 41   p_sign 0.53

Direction as registered, nowhere near significance. The pilot's weak
signal did not consolidate. (Passage length itself was never registered —
the 256-token cap binds both arms; the tail asymmetry lives in the strata
descriptions, not here.)

## 3. B RESOLVES AS COMPRESSED SUBORDINATION — one distribution, two
surfaces, and the moving surface is the per-word one

As registered (per-sentence ratios), both hypotheses are flat:

    parataxis_indep_clauses_per_sent  Δmed −0.0112  13/26 of 41  p 0.053
    hypotaxis_dep_clauses_per_sent    Δmed −0.0216  20/20 of 41  p 1.0
    dep_clause_share (the mix)        Δmed +0.0014  23/18 of 41  p 0.53

Per Amendment 1's rule, neither B.H1 nor B.H2 is reportable as a finding
from those rows. The denominator-free reads move instead, and together:

    dep_clauses_per_1000w    Δmed +3.67   29/11 of 41   p_sign 0.0064
    clause_len_words_mean    Δmed −0.224  13/28 of 41   p_sign 0.028
      (pooled: −0.299, 11/31 of 42, p 0.0029)

**Aligned prose packs MORE dependent clauses per 1,000 words into
SHORTER clauses**, with per-sentence ratios flat because the sentence
denominator shrinks in step — exactly the picture plan B Amendment 2
predicted from the pilot feature diff (the non-finite cluster: acl,
advcl, xcomp, VBG/VBN/TO all aligned-higher) before this run existed.
The Amendment-2 joint table lands in its second cell (A.H1 null, clause
shortening real): compression WITHOUT a per-sentence subordination
shift. Modal density: flat (hedging candidate stays dead). Max clause
depth: flat.

## 4. Verdict-grade descriptions (per-arm rates, N = 238,381)

    list-formatted lines   aligned 4.8% vs base 2.1%   (the format
      attractor at passage grain, 2.3x, direction never registered)
    prose passages         aligned 88.9% vs base 94.5%
    degenerate             aligned 8.1% vs base 5.7%   <- NOTE: aligned
      degenerates MORE by this screen; unexpected, unexamined, and worth
      its own look before anyone repeats "degeneration is a base-model
      behaviour" from the pilot close reading
    English                aligned 97.8% vs base 95.3% (base drifts out
      of English more — Teuken-base and kin)

## Deepseek exposure ([5770] defect, checked same day)

The deepseek pair's stored passage texts are undetokenized (spaceless,
byte markers); its flags were computed pre-defect on spaced text, so
2,481 of its passages sat INSIDE the hardened stratum carrying garbage
measures. Exclusion rerun, every verdict INSENSITIVE: both TTR rows are
bit-identical (the w100 window never fits spaceless text — deepseek was
structurally absent from A.H2 all along, one of the three missing pairs
behind n=39); dep_clauses_per_1000w unchanged at 29/11; clause_len
slightly STRONGER without it (12/28, p .018 vs .028); every null stays
null. The per-arm description rates stand — the flags' spaced-text
source is the right instrument for "did the model degenerate" (the
store lost the decode, not the model). The pair is fenced from
text-grain analyses until the store is repaired ([5770]: malign's to
repair or fence).

## Limits

Single analysis pass, no cross-seat reproduction of any number here yet
([5503] discipline: nothing above is audit-grade until a second seat
reproduces it from the artifacts). Sign tests quoted; Wilcoxon values in
results/m06_verdicts.json. The degeneracy/English screens' thresholds
were declared before the verdict pass but after the pilot's close
reading (Amendments 4/5 record the sequence). n_pairs is 41 on most
stratified rows (one pair contributes no stratum cells; per-pair
denominators in the JSON). The forced-arms secondary table has not run.
The style profile is fiction-continuation register; the legacy-corpus
replication (TODO) is the register-generality check.
