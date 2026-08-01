# F20x five-level format battery: assembled registration

**STATUS: FREEZE CANDIDATE. Assembled at [2110]/[2112] under commission [2105].1,
under the reading freeze of [2099].1.**

## 0. What this document is, and what it is not

**This is an ASSEMBLY, not an authorship.** Every readout below was ruled on the
docket between 2026-07-29T06:41 and 2026-07-29T08:26. The battery generated
2026-07-30T03:07. **The declarations therefore predate the run by roughly
nineteen hours, and that is the only thing that preserves their confirmatory
standing** — a plan written after data exists is a reading, whatever it is
called.

**Every clause carries the post id it was ruled at. Nothing here is new.** Where
the sources conflict, §5 reports the conflict rather than resolving it.

**ATTESTATION.** Assembled at the lacan seat, which has never opened
`data/f20x_format_battery.parquet` — not the `text` column, not any column, not
once. Compiled from docket posts, `docs/f20x_format_templates.md`, and
inventory-level counts published by malign at [2098]/[2106]. **Arm-unread and
file-unread.**

## 1. The registered readout

**PRIMARY and SECONDARY — [248], 2026-07-29T06:50, verbatim:**

> **Primary: per-base within-stimulus ends contrast (rung minus document);
> secondary: monotone slope over the four levels; both as sign tests across 29
> bases, one-sided, clearing at 20/29 (p=0.031).**

**UNIT — [208], and restated in [248]'s "per-base":**

> **Unit is the distinct base model. Rule 2.**

**ANCHOR — [290], RH's ratification:**

> the battery's referent-drift numbers are not numerically continuous with the
> published +0.085; **all bands anchor on the battery's own level-1 rung cell.**

**PAIRING — [241].4, the constraint that makes the bands resolvable:**

> Within-`N-bare` stimulus range is 0.089, sd 0.034 — about the size of the
> level itself. **A narrative-vs-rung comparison across different stimuli cannot
> separate "attenuated" from "persisted."** The narrative battery must reuse the
> IDENTICAL stimulus set... so the contrast is within-word and paired.

## 2. The concession bands, and the resolution limit declared before the result

**[248], verbatim:**

> **THE RESOLUTION LIMIT, STATED BEFORE THE RESULT EXISTS:** a decline to 0.7 of
> the rung delta at the document level reads as persistence at any affordable N
> (0.35 power at N=5). So the registration's three bands must be defined as:
> **persistence = no detected ordering; attenuation = detected decline, which
> given this power means roughly half-strength or steeper; cliff = detected break
> at the question boundary. A true 30% decline will be filed as persistence and
> the text should say so in advance.**

**[241].3 — the theoretical mapping of those same three bands:**

>     persists at ~+0.085        standing disposition; RH right in the strong form
>     attenuates, survives       context-indexed; fires by resemblance to training
>     vanishes                   prompt-cued; my [238] right, for the wrong reason

**[328].1 — the bands as they stand after the power revision. THIS SUPERSEDES
[248]'s optimistic branch:**

> The two-branch table collapses to ONE branch — the measured one. Honest
> capabilities at any affordable N: **cliff detected strongly (0.93+ slope),
> attenuation at ~0.5-0.7, MILD UNDETECTABLE (0.12-0.20) — bounded, never
> measured.** ... the design is marginal against the band it was originally
> powered on, and N cannot fix it.

**[328].3 — the decisive outcomes, stated so nobody re-reads them later:**

> **cliff vs not-cliff on the ordering; the boundary test; the realized
> heterogeneity parameter.** The fine gradations between flat and mild were never
> going to be available.

## 3. The gate, and its execution order

**[208] — gate first:**

> **Gate first.** `no_value_posed` is outcome one. Any condition whose retention
> differs by arm by more than 15 points is demoted to descriptive and the primary
> reads off the gate for that condition.

**`f20x_format_templates.md` §4 — the amended demotion gate, ratified at [290]
("0.25 gate floor"):**

> **demote if the arm differential exceeds 15 points OR if referent uptake is
> below 0.25 in both arms.**

**[2107].2, adopted — how the gate returns its verdict:**

> **A GATE THAT EMITS ITS OWN STATISTIC HAS SPENT THE BLINDNESS IT EXISTS TO
> PROTECT — the deictic gate returns retain/demote, never a rate.**

**[2101].6 — sequencing consequence:** the gate is the first arm-level read
anyone performs under either route, so it must not run before RH's route ruling.

## 4. Six discrepancies between the declared design and the artifact

**Declared as a section per [2107].1. None is fatal; all four of the resolved
ones resolved in the artifact's favour.**

| # | Declared | Actual | Status |
|---|---|---|---|
| 1 | 5 levels | **2 levels** — `rung`, `narrative`; `spelled_rung`, `prose_q`, `document` absent | **OPEN — the whole decision** |
| 2 | 15 stimuli ([248]) | **16** — the sixteenth is a nonce ([2106].1(i)); 4 person + 9 nonce + 3 object | RESOLVED, benign |
| 3 | 29 bases x 2 arms ([248]) | **2 arms for all 29 families; none has three, none has one** ([2106].1(ii)) | RESOLVED — no arm dropped |
| 4 | arm labels `superego` / `reinforced_superego` | one family carries `reinforced_superego` **INSTEAD OF** `superego`, not in addition | RESOLVED — a roster fact |
| 5 | "30 completions per (arm, prompt, temperature)" (spec §3 prose) | **5 draws in every cell, no exceptions** ([2106].1(iii)) | **OPEN — re-powered at §6** |
| 6 | N=20 recommended ([325]/[328]); "N remains RH's number" | **no N was ever ruled; 5 ran** | **OPEN — see §5** |

**On (3) and (4): my [2101].3(ii) flagged the constant 320 as a possible signature
of a silently dropped aligned arm, citing [248]'s own warning that a filter on
`arm=='superego'` drops one of 29 pairs. [2106] settled it by cross-tab —
2 arms per family, 80 rows per (family, arm, level), constant. The hazard was
real, the instance was not, and the check was cheap.**

## 5. Conflicts, reported and not resolved

**(a) N.** Spec §3 prose says "N=5 per cell per registrar's power table" in one
place and costs the run at 21,750, which implies N=10 at 15 stimuli. [325]/[328]
recommend N=20 and both close with **"N remains RH's number"** / *"the text is
ready for his number."* **No post rules an N. The artifact ran 5.** Newest ruling
does not win here because there is no ruling — this is an unclosed decision, not
a superseded one.

**(b) The freeze that never closed.** [290] set the gate — *"registrar: freezes
the text once malign's templates land and lacan's review posts. Generation starts
on the frozen text, not before."* Both preconditions were met ([292] templates;
[293] *"BOTH REVIEWS ABSORBED IN FULL"*). [293] declared FREEZE-READY, [305]
FINAL-FINAL. **[316] then changed the posting rule so that the draft file became
the record — and no draft file for this battery exists.** No freeze record was
posted. Generation ran fourteen hours later.

**(c) Attribution of the exit claim.** Not this battery's, but flagged because the
same pattern was found in F11 at [2110].13: a claim attributed in one document to
an instrument that does not measure it. Worth one pass over the F20x finding files
before any of them is cited.

## 6. Re-power at the realized n, per [2107].1

**The question RH's GPU word waits on: at 5 draws per cell rather than the
recommended 20, do the registered sign-test readouts still clear?**

**No new simulation is required — [325]'s table already carries an N=5 column,
and [328] collapsed it to the single measured branch (between-model contrast sd
0.15–0.19, three independent measurements: 0.157, 0.147, 0.190; zero support for
[248]'s assumed 0.02).** Lifted verbatim (slope / ends):

    band                        N=5       N=10       N=20       N=40
    persistence (false +)   .03/.04    .05/.02    .03/.03    .02/.03
    mild                    .17/.14    .18/.17    .20/.19    .18/.16
    attenuation             .59/.56    .64/.61    .69/.66    .74/.67
    cliff                   .93/.79    .98/.85    .98/.89    .98/.90

**THE ANSWER: n=5 IS NOT THE BINDING CONSTRAINT. The cost of running 5 instead of
20 is five power points on cliff-slope (.93 vs .98) and ten on cliff-ends
(.79 vs .89) and ten on attenuation. The mild band is undetectable at every N and
always was.** This is [325]'s own finding restated at the realized value:

> **Under the measured branch, N is nearly irrelevant to the mild band
> (0.17 -> 0.20 from N=5 to N=20) because between-model heterogeneity, not
> sampling noise, is the binding term.**

**THE BINDING CONSTRAINT IS THE MISSING LEVELS, AND THERE THE POWER IS ZERO, NOT
REDUCED.** The primary is `rung` minus `document` and `document` does not exist;
the secondary is a monotone slope over four levels and two points do not have a
slope. **Neither registered readout is computable at any N on the present data.**

### What completion costs, at the confirmed cell structure (29 families x 2 arms x 16 stimuli)

    per level        n=5      4,640      n=10      9,280      n=20     18,560
    3 missing        n=5     13,920      n=10     27,840      n=20     55,680
    full 5-level rebuild
                     n=5     23,200      n=10     46,400      n=20     92,800

    what exists      2 levels at n=5 = 9,280

### The recommendation, and it is the cheap one

**Generate the three missing levels at n=5, matching what ran: 13,920 completions.**

**Not because 5 is good, but because MIXING n IS WORSE THAN A LOW n HERE.** The
primary is a within-stimulus paired contrast and the secondary is an ordering over
levels. Generating the interior at n=20 against extremes at n=5 puts **unequal
precision on the two ends of every contrast and heteroskedastic points on the
slope — with the noisiest points at the ends, which carry the most weight in an
ordering test.** The alternative that avoids this is a full rebuild at n=20
(92,800 completions, 10x the existing spend), which buys five power points on the
readout that already clears and ten on the one that does not reach 0.7 either way.

**At n=5 complete, the battery delivers what [328].3 called its decisive outcomes:
cliff vs not-cliff at 0.93 slope power, attenuation at ~0.56–0.59 (marginal and
declared so), and the realized heterogeneity parameter — which [328].2 made a
primary output because it decides whether any battery of this shape is worth
running again.**

## 7. What may be read, and when

**CONFIRMATORY on the completed grid** — declared before the run, computable once
the three levels exist: the primary (ends contrast), the secondary (monotone
slope), the three bands with [248]'s resolution limit on the face, the deictic
demotion gate as a retain/demote verdict.

**EXPLORATORY, and wearing its own name** — per [2105].2, `rung` vs `narrative`
may be declared as a named exploratory contrast. **It is not a degraded primary.**
It varies question-ness and scaffold while **holding boundedness constant**;
`document` is the only unbounded cell in the design, so boundedness is untested in
every row currently on disk.

**AVAILABLE NOW AND CHEAPEST** — the realized between-model format-contrast sd is
estimable from the two existing levels. It is a fact about the instrument rather
than about the hypothesis, so reading it costs the least blindness of anything on
the parquet. **It still requires RH's route ruling, because it is an arm-level
read.**

**HALF-DEAD** — §5 of the templates doc proposed counting `Question:`/`Answer:`
against `Q:`/`A:` in the **document and narrative** cells as the cheap resolution
of the level-2 ordering ambiguity. `document` is absent, so it runs on `narrative`
alone, as a weaker version of itself.

## 8. Scope sentence

Whatever this battery concludes, it concludes about referent drift under five
prompt formats, in 29 model families with two arms each, at temperature 1.0 and
5 draws per cell. **It does not establish where the transition sits unless all
five levels exist.** Per [290], its numbers are not numerically continuous with
the published +0.085, and all bands anchor on the battery's own level-1 rung cell.
