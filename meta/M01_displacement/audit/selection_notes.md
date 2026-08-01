# Selection notes on the round-2 survivors

Recorded per [1876].3. **These are properties of the surviving pool that no later
analysis will surface on its own**, because a filter applied for one reason
conditions everything subsequently measured on what it leaves behind ([1597].3).

## ANIMAL (subdomain `cruelty`): 50 stand, and they are a NON-RANDOM SUBSET

RH ruled 50 stands rather than rebuilding 120. The reasoning is sound — the
subdomain is not load-bearing and 50 clean pairs beat 120 contested. **But the 50
are the pairs where that drafter happened NOT to add a transgressive instrument
alongside the transgressive verb, and that is a selection on sentence
construction.**

**The hypothesis was that they may be "systematically simpler and shorter." It is
MEASURED here rather than recorded, from `r2_audit.json`:**

    animal, single-span (survivors)      n= 47   MARKED length mean  9.15  median  9.0
    animal, multi-span (failures)        n= 73                      10.16  median 10.0
                                                 difference -1.00,  MW p = 0.0011

    animal survivors                     n= 47   mean  9.15
    ALL OTHER subdomains' survivors      n=437   mean 11.95
                                                 difference -2.81,  MW p = 6.8e-06

**CONFIRMED AND LARGER THAN THE INTERNAL GAP.** Animal's survivors are ~2.8 tokens
shorter than every other subdomain's survivors — nearly three times the 1.0-token
gap between animal's own survivors and failures. So the effect is **not only** the
filter removing longer pairs within animal; **animal's pairs were shorter to begin
with, and the filter sharpened it.**

**Counts here use the aligned diff's own criterion (`n_spans > 1`) and give 47/73;
lacan's 50 also reflects hand rulings on the second pass, so the two are close but
not identical. The length comparison is unaffected — it is between long and short,
not between 47 and 50.**

### WHAT THIS MEANS FOR ANYTHING MEASURED ON THE POOL

**Sentence length is not a nuisance in this project — it is upstream of the
outcome.** Departed mass and every movement statistic depend on the continuation
distribution, and a shorter prompt with fewer preceding content words is a
different distributional object from a longer one. **Any per-subdomain comparison
involving animal is confounded with length**, and any pooled analysis carries
animal as a systematically shorter stratum.

**Not a reason to drop animal. A reason that `subdomain` and length must both be
available to anyone who compares across subdomains, and that a subdomain-level
difference involving animal is not interpretable as a content difference without
the length control.**

## THE GENERAL FORM

**A FILTER APPLIED FOR ONE REASON CONDITIONS EVERYTHING LATER MEASURED ON ITS
SURVIVORS.** The 2(e) filter was applied for span-count reasons and has delivered
a length-selected subdomain. Nothing detects that downstream; it has to be
recorded at the point of selection or it is invisible.

**The same question is open and unmeasured for the other subdomains' 211 failures
— this note measures animal because RH's ruling made animal's survivors a standing
population. The others were re-drafted or are pending, so their survivors are not
yet a fixed set.**
