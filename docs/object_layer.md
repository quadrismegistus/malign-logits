# The object layer

Six classes, one file each, over artifacts that already exist. Nothing here adds a fact
or re-implements a traversal — the layer exists so that questions this project asks
constantly are one object instead of four lookups.

```
malign_logits/prompts.py      Prompt, PromptGroup, Prompts
malign_logits/checkpoint.py   Checkpoint
malign_logits/family.py       Family
malign_logits/step.py         Step
malign_logits/cell.py         Cell
malign_logits/movement.py     movement(), movers(), word_probs(), CANONICAL, DRAW
```

The flagship exhibit, end to end:

```python
from malign_logits.checkpoint import Checkpoint
from malign_logits.step import Step
from malign_logits.movement import CANONICAL

s = Step(Checkpoint("LLM360/Amber"), Checkpoint("LLM360/AmberChat"))
m = s.cell("She was so angry she wanted to").movement(CANONICAL)
m.top_riser()        # 'scream'
len(m.fallers), len(m.risers)      # 26, 12
```

---

## Why it exists

Every class here replaces something that was being done by hand, and every hand-rolled
version had produced a defect by the time the class was written.

| Class | Replaces | The defect it ends |
|---|---|---|
| `Prompt` | ad-hoc lookups into `prompt_categorisation.json` | a text-keyed dict reported **48 group disagreements** where the true figure was **1** |
| `Checkpoint` | four separate lookups across five files | a hand-assembled roster claim — *"every remaining family is a transformer"* — while two pure-SSM families sat in the spec |
| `Family` | reading `MODEL_FAMILIES` slots directly | `position` cannot distinguish preference methods, so the separator cell was unaddressable |
| `Step` | a family key plus an edge label | a label-based design was dead on **62 of 103** models while the registry cache was stale |
| `Cell` | per-script riser/faller derivation | fourteen scripts, two incompatible rules, **1,650 cells against 3,366** on the same question |
| `movement` | `Psyche.displacement_map()` | a `cosine >= 0.15` filter buried 200 lines inside a method, invisible at the call site |

---

## The design rules, and what each cost to learn

### A method must not hide an analytic choice

`displacement_map()` reads like a fact and was a filter. So no method here computes a
movement statistic without naming its rule in the call:

```python
cell.movement(CANONICAL)     # the rule is visible at every call site
cell.movement(DRAW)          # and so is the difference
```

`CANONICAL` tests risers against the renormalisation null; `DRAW` does not. **Both ship,
named**, because work exists under each and silently unifying them would invalidate it.
`DRAW` is what fed the annotation item draw, so M01's clauses 5–6 rest on it.

### Objects for nouns, functions for choices

`Prompt`, `Checkpoint`, `Family`, `Step`, `Cell` are nouns and carry state. `CANONICAL`
and `DRAW` are nouns too — a rule is a thing. The metrics are functions over a cell.
The failure of the old god-object was never that it had methods; it was that its methods
made invisible commitments.

### Identity is the unambiguous key, never the readable one

`Prompt` is keyed by `prompt_id`, never by text: **61 texts carry more than one row**
(one prompt serving two designs), and a text-keyed dict takes whichever came last.
`Prompt.find(text)` gives the ranked pick — ACTIVE over DISPUTED over RETIRED, then
grouped, then role-bearing — and `.duplicates` makes the ambiguity visible where it bites.

*(That 61 is **all statuses, English only**. The same question has four answers — 64 all
statuses all languages, 12 ACTIVE all languages, 9 ACTIVE English — and this doc quoted it
without its population until 2026-07-31, when a hand-rolled lookup elsewhere let RETIRED
rows win keys and moved a rank-sum across the significance line in three families. The
ranked pick is what makes the ACTIVE figure the operative one; see
`instrument_commitments.md` §9 for the version an implementation outside this package
needs.)*

`Step` is identified by its **pair of checkpoints**, not by a family and a label. A label
needs a lookup table; the pair needs nothing and works for comparisons nobody named.

### Refuse, or stamp — but never average

```python
cell.movement(CANONICAL)                      # raises on a mixed rule_version
cell.movement(CANONICAL, allow_mixed=True)    # with a stated reason
```

A v1 pre-arm against a v3 post-arm books an **instrument change as training movement** —
v3 changed what a word is, so words appear, merge and vanish for reasons that have nothing
to do with the model.

Where refusing would block real work, the layer stamps instead. Teacher-forcing
base→sft→base is a legitimate experiment, so `Step` does not reject a reverse pair; it
records `direction` on every result, and a pooled analysis cannot silently mix.

### Four directions, because two would erase a distinction

```
forward    base -> sft -> preference -> rlvr
reverse    the same, backwards
lateral    kto vs dpo — alternatives at ONE point, not a sequence
unknown    either stage is unknown, and says so
```

Calling `kto → dpo` "reverse" would invent an ordering the training never had.

Direction is read from **stage order**, not from the edge list. The registry's relations
*are* now chained — `path(base, kto_arm)` returns `sft_of` then `kto_of` — but a traversal
only answers where an edge exists, and a `Step` is defined for pairs that have none: two
checkpoints from different families, a base against another family's aligned arm, any
comparison nobody declared. Stage order answers for all of them.

*(An earlier draft of this section said the edges were star-shaped from the base and that
`path()` therefore could not sequence them. That was true when the layer was written and
was fixed in `92ebc95`, seven minutes later. The design does not change — `Step` takes a
pair precisely so that it does not depend on how the edges are shaped — but the reason
given for it was stale, and a doc organised by defect should not carry one.)*

### Two vocabularies, kept apart on purpose

```python
f = Family("archangel-kto")
f["superego"]    # by POSITION — the taxonomy's slot, complete for every family
f.dpo            # None — by STAGE, and honestly so
f.preference     # the KTO arm — by method, whatever the method is
```

**`position` cannot distinguish preference methods**; all four archangel arms are
`superego`. **`stage` can** — and that is the whole reason those four checkpoints form a
separator cell, holding SFT constant while the method varies. Asking by stage when you
mean method, and by position when you mean slot, is the point of having both.

### The gap is a gap, never a default

`architecture` returns what the registry says, including `None`. An earlier draft returned
`"transformer"` for anything unmatched — the majority-class fallback that mis-described a
roster containing two pure-SSM families. That draft's own docstring said *"defaults to
None, never to the majority class"* while its code did the opposite.

`cell.prompt` is `None` for text the catalogue does not carry. The grid scores census
strings too, and a default domain would quietly pool unclassified prompts into `neutral`.

### Stratification is one attribute away, deliberately

```python
cell.domain, cell.language        # from the catalogue row
cell.record()                     # a flat dict WITH its strata attached
```

*Stratify before the statistic, not after* cost a round on the docket to establish. This
makes it the path of least resistance rather than a rule someone remembers.

---

## Reading `true_word_probs`

**Do not write `{r["word"]: r["p"] for r in payload["rows"]}`.**

The payload is one row per `(word, FIRST TOKEN)` and the rows are a **partition** —
summed with the residual they come to exactly 1.0. A dict comprehension keeps the last
token path and drops the rest.

```
payloads scanned 300    containing a duplicated surface: 60  (20.0%)
max rows sharing one surface: 3
worst observed single-cell loss: 99.85%
```

Three separate consumers shipped this bug, one of them losing **4.6% of the distribution
over 979 prompts**. `word_probs()` folds the partition correctly and reports `.collapsed`
so a caller can see when it happened. The damage is **anti-correlated with salience**:
the median cell loses exactly 0.000%, so every spot check passes, and the catastrophic
losses land on the smallest cells — the ones nobody would open, whose smallness the loss
itself produced.

The general form: **for any aggregation over a partition, assert the partition sums.**

---

## The renormalisation null

```
faller  iff  P >= 0.003  AND  Q < 0.5 * P
R = 1 - sum_fallers Q        S = sum_non-fallers P
null = P * (R / S)           what each survivor gets from PURE renormalisation
riser   iff  not faller  AND  max(P,Q) > 0.003
             AND  (Q - P) > 0.003
             AND  Q > null                     <- more than renormalisation explains
```

Without the last line a riser is any word that went up, and **every** word goes up a
little when a faller's mass is removed.

**Declared asymmetry, preserved from the original:** risers are tested against the null,
**fallers are not**. A faller is a bare ratio rule, so nothing downstream may describe
fallers as "beyond renormalisation."

**On `true_word_probs` the null is approximate and says so.** The support is truncated at
theta, so the residual is carried as an explicit non-faller mass — dropping it inflates
every survivor's null, renormalising deletes the mass that left the scored set (about a
quarter of the distribution). `diagnostics["exact_null"]` is `False` on that path;
`residual_share` reports what the approximation rests on. The residual can never be a
faller: an undifferentiated bucket has no word to fall.

---

## What is deliberately absent

**No graph.** `Registry` already is one — models as nodes, typed relations as edges, with
`parent_of`, `children_of`, `path`, `variants_of`. Its gap was coverage, and that was a
stale-cache bug rather than missing structure. A `GraphStash` wrapper would have inherited
the same 41 rows with a nicer query language.

**No `Lineage` — but the blocker is gone.** "Every olmo" is a real question: does the
effect scale across 1B → 7B → 32B on one recipe. Building it *was* blocked because the
grouping would have had to be inferred from the key prefix, and name-pattern inference
had already produced two defects. **`smaller_version_of` now carries 18 edges**, so the
grouping is a declared traversal and the class can be built on evidence whenever an
analysis wants it.

**No persisted derived movement.** The cache plus `word_probs()` gives recomputation for
free, and persisting derived numbers is the staleness that retired M01/stage-share.

**No aggregation helpers yet.** `step.records()` produces dataframe rows; anything above
that is better shaped by the first real analysis than guessed in advance.

---

## Provenance

Written 2026-07-30 against `data/prompt_categorisation.json` (frozen at `9fd292b`,
987 active rows, 30 assertions) and `data/model_registry.json` (112 models, 155 typed
edges). 204 tests pass; each test in this layer was watched to fail before it was kept.
