# Positive-control marker slate — DECLARED BEFORE TESTING

Registered 2026-07-26, before any `D` was computed for these pairs. Required by
`docs/preference_corpus_spec.md`: seven pairs, named on external grounds, all
seven tested, all seven results disclosed whichever way they run.

**External grounds.** RLHF preference data rewards hedged, qualified and
de-escalated phrasing over blunt, absolute or pejorative phrasing. This is the
documented shape of helpfulness/harmlessness annotation (Bai et al. 2022;
Ouyang et al. 2022; Stiennon et al. 2020) and is independent of the reroute
chains, which concern transgressive content rather than register.

**Frequency comparability.** The spec requires both members within a factor of 3
of the median chain-word frequency (2,642 combined in hh → band 881–7,926). The
slate is chosen for *attestation first*; where a member falls outside the band
that is recorded here rather than used to reselect, because reselecting on
frequency after seeing which pairs are attested would reintroduce the shopping
the seven-candidate rule exists to prevent.

| # | blunt | preferred | ground | hh counts (c/r) | in band |
|---|---|---|---|---|---|
| 1 | `must` | `should` | softened obligation; hedged directive | 3703/3552 → 31279/30241 | blunt yes, pref **no (high)** |
| 2 | `never` | `rarely` | absolute → qualified quantifier | 3729/3961 → 246/234 | blunt yes, pref **no (low)** |
| 3 | `always` | `often` | absolute → qualified quantifier | 8972/8769 → 19346/18544 | **both no (high)** |
| 4 | `wrong` | `incorrect` | pejorative → neutral register | 2826/2752 → 316/330 | blunt yes, pref **no (low)** |
| 5 | `stupid` | `unclear` | pejorative → neutral attribution | 531/652 → 455/464 | **both no (low)** |
| 6 | `no` | `unfortunately` | bare refusal → softened refusal | 17454/17622 → 1430/1490 | blunt **no (high)**, pref yes |
| 7 | `obviously` | `perhaps` | confident assertion → epistemic hedge | 398/408 → 4675/4746 | blunt **no (low)**, pref yes |

**Excluded before testing:** `cant`→`cannot` (blunt member has 5/5 occurrences,
below the 20-occurrence floor — a stimulus property, checked before any `D`).
`sorry`→`unfortunately` remains disqualified per the spec: it was computed at
spec-writing time, runs the wrong way in both corpora, and is burned as a blind
control.

**Firing rule.** A pair fires if `D > 0` AND `|D|` exceeds the p75 decoy floor
for that corpus (hh 0.0903, pku 0.2657). The gate passes at **3 of 7**. Fewer
than 3 books the INSTRUMENT-INSENSITIVITY finding, not a verdict on convention.

**Note on the band.** Only one pair (`stupid`/`unclear`) has both members inside
the comparability band, and no pair has both members comfortably inside it.
That is a real weakness of the slate and is registered here rather than
discovered afterwards: register markers are either much commoner or much rarer
than the mid-frequency content words the chains are built from, so
frequency-matching and external attestation pull against each other. The gate
result must be read with that stated.
