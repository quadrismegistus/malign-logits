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

## 7. Any distributional claim must exceed the sharpening baseline

Preference training reduces next-word entropy by **0.2–0.6 bits** and raises top-1 mass by
**2–10 points** in five of six families. Two peaked distributions over different supports
have high JS almost mechanically, so a great many "effects" reduce to this.

`malign_logits.sharpening` computes it. **`archangel` does not sharpen and is the roster's
natural null** — if a claimed effect is absent there too, it is a candidate for reduction.

An effect significant at 1e-06 to 1e-27 across five families was found to be entirely this.

## 8. Every posted table names its language population and its residual handling

Two lines. They turn a non-replication into a five-minute reconciliation.

---

## The rule behind the rules

**Two seats running one implementation is one seat.** Reconciliation by adopting the other
seat's script is not verification. Had either seat done that on 2026-07-31, they would have
agreed at 7 of 10 and been wrong together.

So independent implementations are wanted, which is exactly why the commitments have to be
findable from outside the package.
