# Instrument commitments

**Read this before writing any code that scores or compares distributions in this repo,
including code that deliberately does not use `malign_logits`.**

These are not style preferences. Each is a choice that changes published numbers, each was
learned from a defect, and each **binds independent implementations as much as this one**.
An independent scorer that silently differs on any of them will produce a number that
looks like a non-replication and is not.

That is not hypothetical. On 2026-07-31 two seats ran the same comparison and got
**7 of 10 families against 3 of 10**. Neither implementation was wrong. They differed on
two of the commitments below, both undeclared, and reconciling took a reproduction to the
decimal place. The commitments were written down — in a class docstring, where a person
writing their own implementation had no reason to look.

---

## 1. The residual is a BIN, not a renormalisation

`word_probs()` returns a truncated head plus `residual`, and they sum to 1.0. When you
compute any divergence, **carry the residual as an extra bin.** Do not renormalise over
the visible words.

Renormalising reports a redistribution among survivors and hides the mass that left the
scored set — **0.05 to 0.21 of the distribution** on this instrument, a fifth in places.

```
olmo, institutional vs neutral, same population:
  residual kept     z = 4.10
  residual dropped  z = 0.48
```

## 2. Language populations do not pool without a declared commensurability check

English and Chinese behave **incompatibly** on this instrument. Measured, not assumed:
`js_total` gives opposite significant answers across families (amber 275/386 en>zh at
3.6e-17; yi 79/386 at 7.8e-33), `tail_share` flips with it, and the one metric that looked
unanimous across five models failed the `base->sft` control.

**English-only, declared, is the default for cross-family strata analyses.** Pooling is
allowed and must be stated, with a reason.

```
tulu, institutional vs neutral:
  English only  z = +3.29
  pooled en+zh  z = +1.06
```

## 3. `true_word_probs` rows are a PARTITION — sum them, never overwrite

One row per `(word, FIRST TOKEN)`. **Do not write `{r["word"]: r["p"] for r in rows}`** —
it keeps the last token path and drops the rest.

```
payloads scanned 300     containing a duplicated surface: 60 (20%)
worst observed single-cell loss: 99.85%
```

Damage is **anti-correlated with salience**: the median cell loses exactly 0.000%, so every
spot check passes and the catastrophic losses land on the smallest cells. Three separate
consumers shipped this.

## 4. Mixed `rule_version` is an instrument change, not a result

v3 changed what a word is (contractions, mojibake, Chinese). A v1 arm against a v3 arm
books an instrument change as training movement. `Cell` raises; an independent
implementation must check the field and refuse, or state a reason for proceeding.

## 5. Risers are tested against the renormalisation null; fallers are not

```
faller  iff  P >= 0.003  AND  Q < 0.5 * P
R = 1 - sum_fallers Q          S = sum_non-fallers P
null = P * (R / S)
riser   iff  not faller  AND  max(P,Q) > 0.003  AND  (Q - P) > 0.003  AND  Q > null
```

Without the last clause a riser is any word that went up, and **every** word goes up when a
faller's mass is removed. The asymmetry is deliberate and preserved from the original, so
nothing downstream may describe fallers as "beyond renormalisation."

Two rules ship, named: `CANONICAL` applies the null, `DRAW` does not. `DRAW` fed the
annotation item draw, so work rests on each. Name the rule at the call site.

## 6. Excess is ZERO-SUM across survivors

`sum_non-fallers null == R == sum_non-fallers Q`, so the excesses cancel over the union
support. Consequences: **`arrived` is not a share of `departed`** (the two have no ordering
relation and their ratio routinely exceeds 1), and any per-word excess must be computed on
the **union** of pre and post keys — iterating the pre keys alone skips post-only words and
breaks the identity.

## 7. Sharpening is measured per family, never assumed

Two peaked distributions over different supports have high JS almost mechanically, so a
step that concentrates its distributions raises pairwise divergence for free. **Every
distributional claim reports its own families' measured sharpening beside the effect.**

`malign_logits.sharpening` computes it. **A family the operation barely touches is the
roster's natural null** — if a claimed effect is absent there too, it is a candidate for
reduction. On the current roster `archangel` and `ct-llm` are flat.

An effect significant at 1e-06 to 1e-27 across five families was found to be entirely this.

**This entry carried a MAGNITUDE until 2026-07-31 — "0.2–0.6 bits, five of six families" —
and the magnitude was wrong.** Measured on sixteen families the median is −0.034 bits, the
range runs −0.643 to +0.378, and one family de-sharpens. The figure came from six families
that happened to include the heaviest sharpeners. **The rule now carries no numbers at all,
deliberately: writing a population-bound figure into a standing rule is the original defect,
not its fix.** A gate's error propagates retroactively through everything it filters, so
**a gate carries a higher population bar than the claims it gates.**

## 8. Every posted table names its population, its residual handling, and its test sidedness

Three lines. They turn a non-replication into a five-minute reconciliation — and on
2026-07-31 the absence of each one in turn moved a count: residual handling (7 of 10 vs
3 of 10), sidedness (11 vs 14 of 16), population (a duplicated row, then seven retired
ones). **Not one of them moved the direction.**

And the corollary, learned the same day: **a declared population is not a checked
population. The diff is the check.** Two seats each stated an `n`; neither read the
other's, and the difference sat in both header lines for an hour.

## 9. The catalogue holds retired rows, and a text can carry more than one

`data/prompt_categorisation.json` is **not** a list of live prompts. It holds 1,189 rows of
which 198 are `RETIRED` and 4 `DISPUTED`, and **the same prompt text can appear on several
rows**, sometimes under different findings, sometimes with different domains:

```
texts on more than one row      all statuses   64      ACTIVE only   12
                          English only   61      ACTIVE + English    9
```

Each of those four is the answer to a different question, and `61` — the figure quoted in
`object_layer.md` and across the docket — is the **all-status English-only** one. State
which you mean; the gap between 64 and 9 is the gap between "this file has duplicates" and
"a live lookup will hit one."

```python
{r["prompt"]: r["domain"] for r in doc["prompts"]}          # WRONG, twice over
```

That comprehension lets a **retired** row win a key by file order, and it collapses a
one-to-many relation to one value. Both happened: six prompts were reassigned from their
live `neutral` classification to a `violence` label carried only on a retired F13 row, which
silently removed them from a stratum and shifted a rank-sum across the significance line in
three families.

**Filter status before building any lookup, and expect a text to carry more than one row.**
`Prompts.where()` defaults to `status="ACTIVE"` and `Prompt.find()` applies a ranked pick —
the package guards this by construction, which is exactly why an independent implementation
that (correctly) does not import the package needs it written here.

**Still undeclared, and small:** one English text is `neutral` on one ACTIVE row and
`contradiction` on another. It currently enters both strata because nobody has ruled
otherwise. If you build a stratum, say which rule you used.

## 10. The transport directory and the stash are different objects

`data/twp_grid_v3/` is the **rsync transport** — one JSONL line per scored prompt, written
by the cloud grid so a kill is survivable and a sync is incremental. The **stash**
(`true_word_probs`, via `CacheManager`) is what `Step.cell()` measures. `twp_ingest.py`
bridges one to the other, and says so in its own first line: *"THE CLOUD FILES ARE A
TRANSPORT FORMAT, NOT THE STORE."*

**They diverge whenever ingest lags.** On 2026-07-31 the transport held 90,350 lines
against the stash's 80,336 payloads — a gap of 10,014, because the repair pass was writing
faster than it was being ingested. Neither number is wrong; they count different objects.

```
count the STASH   when you mean "what was measured"       <- pins, provenance, coverage
count the FILES   when you mean "what has arrived"        <- run progress, ingest lag
```

**Three separate errors came from this in one day**, all of them at the pin-and-provenance
layer rather than in any measurement:

- a producer's `STORE` constant pointed at the transport, so its "store pinned 90,077
  payloads" was a line count — and the figure reached a certified clause before anyone
  checked what the constant counted;
- that same producer then reported the table as **stable across five store sizes**. It was
  five reads of a store that never moved, while the transport filled beside it.
  *A re-run is evidence only if the input could have changed;*
- and the transport is **gitignored**, with zero tracked files, so any path into it raises
  `FileNotFoundError` on a fresh clone — for exactly the independent implementer this
  document exists for.

**A live instance, currently harmless, kept here because harmless-today is the point.**
`Checkpoint.landed` globs the transport directory:

```python
def _landed():
    d = os.path.join(PATH_DATA, "twp_grid_v3")      # transport, not stash
```

Checked 2026-07-31: transport and stash agree exactly, 95 models each, zero divergence in
either direction. **So `landed` is right — by coincidence.** It reads a proxy that happens
to match the fact today and would part from it the moment transport is pruned after ingest,
or a model is ingested from anywhere else. Every family enumeration in this project filters
on `landed and prompts`, and only the second of those reads the stash.

**The general form, which is the shape of five defects found on 2026-07-31 across two
seats: AN ARTIFACT MUST ANSWER FROM THE FACT, NOT FROM A CAUSE OR A PROXY OF IT.** A grid
spec is an execution plan, not a roster — *a plan that shrinks is not a roster that shrank*.
An exclusion keyed on weights format answers "why might this fail" when the question was
"do we have the data". A test pinning a roster count asserts a state as though it were a
property. And a transport line count answers "what has arrived" when the clause asked "what
was measured".

---

## The rule behind the rules

**Two seats running one implementation is one seat.** Reconciliation by adopting the other
seat's script is not verification. Had either seat done that on 2026-07-31, they would have
agreed at 7 of 10 and been wrong together.

So independent implementations are wanted, which is exactly why the commitments have to be
findable from outside the package.
