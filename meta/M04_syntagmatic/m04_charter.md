# M04: The syntagmatic campaign — charter

**STATUS: FREEZE CANDIDATE, commission [2246](b). Awaits RH's ratification of the
route in §2.**

## 0. Provenance, and what does not carry over

M04 opens at [2156] on RH's ruling of a ground-up redesign after F14's audit
([2144]). **His constraint binds the whole campaign: pre-campaign work was sloppy,
SO FAILURE TO REPRODUCE IS NOT A RELIABLE NULL. F14's death answers nothing about
the phenomenon; the question is OPEN, not answered negative, and nobody quotes
that audit as evidence of absence.**

**Does not carry over: any F14 number, in either direction** — not the deltas as
effects and not their non-reproduction as a null. F13's direction result stands on
its own audit and is M04's **prior**, not its property.

**The design-side reason RH's constraint is right, on the face:** F14's
per-category MDE was 0.152 at 2 prompts/category against a largest booked delta of
0.106 ([2162]). **It could not have resolved its own claims at any data
completeness.** An instrument that cannot detect an effect it claims tells you
nothing when the claim disappears.

## 1. The founding question, and the COARSE target

**Does alignment damage combination, where, and at which stage** — the FACT half
of the anti-poetic-function argument, whose STRUCTURE half (F13) is regime-audited
and anchored on direction.

**RH's N2 ruling ([2245].2): the confirmatory target is the ~0.10-grade claim —
alignment's substitutions damage combination, concentrated in the transgressive
regime.** The 225-prompt per-category gradient design is **explicitly not
commissioned** and reopens only on his word.

## 2. THE TWO ROUTES, PRICED — this is the section RH ratifies

### Route A — the existing beam stashes (RECOMMENDED)

    substrate      data/raw/cache/beams, 19,105 keys, LIVE
    lineages       28 present, ALL PAIRED base/aligned; 25 carry a paired
                   M04 cell in at least one category
    coverage       9 of 9 categories clear the 4-prompt floor,
                   23-25 lineages paired in each
    depth          max_tokens = 10, invariable, no deeper beam exists
    generation     NONE
    n              25

### Route B — purpose-built corpus

    corpus         36 prompts, 4/category, designed from zero
    lineages       34 (full roster)
    depth          chosen
    generation     full, GPU
    n              34

### The arithmetic that decides it

    sign test, one-sided .05     n=25 clears at 18/25 (p=0.0216)
                                 n=34 clears at 23/34 (p=0.0288)

    MDE, lineage unit, 80% power, two-sided .05
    t-based (df = n-1), which is the correct form at these n --
    a normal approximation understates by 3-4% here ([2260], adopted)
      sd     n=25    n=34
      0.05   0.029   0.025
      0.077  0.045   0.038     <- measured prior, [2162]
      0.15   0.088   0.074     <- pessimistic branch

**EVERY ROW RESOLVES A ~0.10 EFFECT AT BOTH n. The six absent lineages are not
worth generating for AT THE COARSE TARGET**, and that is the whole of the case for
Route A: the substrate exists, the target clears, and **its confirmatory status is
intact because [2245].3's freeze landed before anyone looked.**

**RECOMMENDATION: ROUTE A.** Route B's only advantage is chosen depth, and §4.2
makes depth a measured property rather than an assumption — so Route B's
justification, if it ever comes, arrives with evidence attached.

## 3. Population

**RH's standing scope rule ([2198]), verbatim: EVERY CANONICAL FINDING IN A
meta/ CAMPAIGN RUNS ON ALL FAMILIES WE HAVE — never the 5, 11, or other subsets
the F-series used.**

Operationalised here: the population is the **full roster admissible under a
declared rule** — all lineages with both arms present on a prompt in the category
— never a hand-picked list. **Counts stated at BOTH units** (family labels /
independent lineages, `data/lineage_map_models.json`). **The unit is the
independent pretraining lineage. 103 models is never the n; 48 family labels are
never the n.**

**ID NORMALISATION, from [2251].3's false zero:** any paired or bidirectional
design over the stash resolves identifiers **through the lineage map, never by
string**. `source` is a normalised label and `model` is a HuggingFace ID; comparing
them directly returns a clean, confident, false zero.

**Gaps are ENUMERATED AND NAMED**, never silently dropped: the 6 absent lineages,
the 2 models absent from the map, the 16 uncategorised prompts, and the 8,266 keys
carrying no `source`.

## 4. The instrument

### 4.1 Bidirectional crosswise teacher forcing

Per RH's design core ([2149].1): score each arm's substitutions under **every**
judge — base judges aligned's pairs, aligned judges base's, and each judges its
own — across stages where arms permit.

**This is the difference-in-differences that kills the two-models-differ confound
BY DESIGN, and it is the direct answer to [2144]'s inversion.** The noise account
(two models simply differ) predicts **SYMMETRIC** cross-model divergence; the
damage account predicts **ASYMMETRY** — aligned substitutions jar the chain more,
under either judge. **The discriminating observation is designed in rather than
argued after.**

### 4.2 THE DEPTH PROFILE — declared readout, not an assumption

`max_tokens = 10` is invariable in the stash. **Whether ten tokens is a long
enough syntagm is converted from a design argument into a measurement.**

**Verified implementable ([2257]): `base_token_probs` is length 10, one value per
position, and it is the cross-forced quantity. The profile costs no generation and
no re-forcing.**

**Report the effect at depths 1..10. The three branches, frozen before any read:**

    PRESENT and PLATEAUED or FALLING by ~token 5-7
      -> ten is sufficient, the ceiling is not binding,
         the claim stands as measured

    PRESENT and STILL RISING at token 10
      -> AMBIGUOUS AS DRAFTED. See the divergence confound below; the
         ceiling-is-binding reading is licensed ONLY after the control.

    ABSENT at every depth
      -> a null on this substrate, and the profile itself certifies it is
         not a truncation artifact

### 4.2a THE DIVERGENCE CONFOUND — ruled, per [2258]/[2261]

**A RISING DEPTH PROFILE HAS TWO READINGS AND ONLY ONE OF THEM IS OURS:**

    (i) DAMAGE ACCUMULATING -- the substitution's disruption compounds along
        the chain. This is the construct.
    (ii) FORCED-PATH DIVERGENCE -- the forced sequence simply becomes less
        native to the forcing model with every position. THIS HAPPENS TO ANY
        two models, with or without a substitution, and is not about
        combination at all.

**Unamended, the rising branch licenses (i) from evidence that equally supports
(ii), which is [2144]'s inversion in a new place: taking a quantity the null
predicts and reading it as the null's refutation.**

**RULING: CONTROL, NOT CAVEAT.** The control is available from the stored rows and
therefore costs nothing, and a caveat where a control is affordable is the
[2185] failure.

**THE CONTROL: the depth profile is computed on the DIFFERENCE between the
substituted and unsubstituted paths through the SAME judge, never on a raw path
probability.** Both paths are equally non-native to the judge in every respect
except the substitution, so generic per-position divergence differences out and
what remains is substitution-specific. `path_prob` / `log_prob` / the
per-position `base_token_probs` support this from the same rows ([2257]).

**AND THE CONTROL HAS A POSITIVE CHECK, which is what makes it a control rather
than an assertion: the RAW paths must show the divergence (both falling with
depth) while the DIFFERENCE does not.** If the raw paths are flat, forcing is not
behaving as teacher forcing and the instrument is reported as not validated,
before any arm contrast.

**WHERE THE CONTROL CANNOT BE BUILT** — cells with no recoverable unsubstituted
counterpart — **those cells carry the caveat, are counted on the face, and are
excluded from the ceiling-is-binding reading.** The caveat is a per-cell fallback,
never the design.

**Construct scoping, on the face:** combination damage is **LOCAL** — if a
substitution damages the syntagm it damages it at the words immediately following,
because that is what collocation is. **Damage appearing only at token 40 is
discourse drift, a different construct with a different name.** And note the
scoping and the confound point the same way: **a profile that only rises late is
the shape forced-path divergence predicts, and the shape our construct does
not.**

### 4.3 Conditioning on where resistance bites

Per [2149].3, gates and high-entropy positions enter as **DECLARED STRATA, never
post-hoc cuts.** The stratum definition is frozen with this charter or the
stratified analysis is exploratory.

## 5. Readout

**UNIT: the independent pretraining lineage.** n = 25 (Route A) or 34 (Route B).

**PRIMARY:** the asymmetry statistic of §4.1 — aligned-substitution damage exceeds
base-substitution damage under a common judge — as a sign test across lineages,
one-sided, clearing at **18/25** or **23/34**.

**SECONDARY:** concentration in the transgressive regime — the asymmetry is larger
at transgressive categories than at neutral. **Reported with the neutral category's
own base rate beside it**, per [2144]'s lesson: a nonzero neutral difference is the
noise account's own PREDICTION, not its refutation.

**REQUIRED WITH EVERY RESULT:** the depth profile (§4.2); base rates for every
rate; the design effect on every p; counts at both units; the enumerated gaps.

## 6. MEI-first, and re-powering is a REQUIRED step

**[1989].1 is M04's first clause and applies from birth.**

**The day-zero MDE is §2's table.** Its variance prior — paired-difference
sd 0.077 — is **measured but FOREIGN**: it comes from between-PROMPT differences
on the taxonomy corpus, a corpus M04 does not use, and the unit here is the
LINEAGE.

**So the registration carries re-powering against M04's own realized sd, from its
first cells, as a REQUIRED step and not an optional check.** The [325] precedent is
explicit: an assumed 0.02 turned out to be 0.15–0.19, an order of magnitude, and
only running it could show that.

## 7. Confirmatory and exploratory

**CONFIRMATORY** — declared here, before any arm-level read: the primary asymmetry
test, the transgressive-concentration secondary, the depth profile's three
branches, the declared strata of §4.3.

**EXPLORATORY, wearing its own name:** per-category profiles; the stage
decomposition; any relation between damage and model scale; anything the strata
were not declared for.

**BARRED:** any comparison to an F14 quantity, in either direction.

## 8. Scope sentence, pre-written

Whatever M04 concludes, it concludes about teacher-forced continuation
probabilities at up to ten positions, over the pairs and categories its declared
population covers, at the lineage unit. **It does not establish that a model
"cannot combine", and it does not speak to generated text.** Per RH's founding
constraint, a null here is a null about this instrument on this substrate — **and
this campaign, unlike its predecessor, is powered to say so.**
