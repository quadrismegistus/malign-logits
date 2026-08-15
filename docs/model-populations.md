# The model populations: which "pairs", and where each one comes from

**Written 2026-08-14 because every seat re-derives this every session, and the
sources disagree.** RH: *"how does one get the representative model pairs?"*

There is no single answer, and that is the actual state rather than a gap in
the documentation. **Four defensible numbers exist, they are computed from
three different places, and two of them are live while two are frozen.** This
file names each one and says which to use.

## The short answer

```python
from malign_logits import ch
ch.query("SELECT base, aligned FROM {db}.movement_edges "
         "WHERE is_model_pair AND is_representative")
```

**52 pairs over 47 bases**, registry-derived and live. Drop
`AI-Sweden-Models/gpt-sw3-6.7b-v2`, which is gated and has zero rows on both
sides, and you have **51 measurable pairs over 46 bases**.

Use this unless you specifically need the frozen battery membership.

## The four numbers

| route | pairs | lineages | live? | what it actually is |
|---|---|---|---|---|
| CH `is_model_pair AND is_representative` | **52** | 47 | live | base→superego, canonical, base is its lineage's rep |
| CH `is_model_pair` alone | 58 | 47 | live | the same minus the representative restriction |
| `data/lineage_representative_pairs.txt` | **46** | 46 | FROZEN | what the forced-arms battery happens to span |
| `data/base_aligned_pairs.json` | 54 | — | FROZEN | the forced-arms builders' own output |

**52 pairs over 47 bases is not a contradiction**: some bases carry several
aligned arms. `EleutherAI/pythia-2.8b` alone contributes four archangel arms.

## How the two flags are computed

`scripts/build_movement_edges.py` reads two artifacts and one Python constant:

```
data/model_registry.json        -> position (base / ego / superego)
                                   *** GENERATED. See the warning below. ***
data/lineage_map_models.json    -> model_to_lineage, lineage_to_representative
MODEL_FAMILIES                  -> which arms are canonical
```

and then:

```python
is_representative = l2r.get(m2l.get(base)) == base
is_model_pair     = position(base)=="base" and position(aligned)=="superego"
                    and kind == "canonical"
```

## Where `position` actually lives, and why editing the JSON does nothing

**`data/model_registry.json` IS GENERATED.** `scripts/build_model_registry.py`
joins the DECLARATION in `MODEL_FAMILIES` (`malign_logits/__init__.py`) to
measured artifacts, each naming its own producer in `_provenance`, and
separates `declared` / `derived` / `measured` on purpose — the builder's own
docstring records why: *an architecture claim was asserted from a spec read
and was wrong; a stage field says `dpo` for all four archangel arms because
it was derived from a family label rather than from the name that carries
the method.*

**So `position` is a DECLARED fact and a fix goes in `__init__.py`. Editing
the JSON is overwritten on the next build.** Established by malign at
[6265]; **an earlier version of this file named the JSON as the source of
position and did not say it was downstream**, which would send a reader to
edit the file that gets regenerated. `CLAUDE.md` calls `MODEL_FAMILIES` the
single source of truth without saying the registry is generated from it.

**14 CHECKPOINTS ARE IN THE REGISTRY AND IN NO TAXONOMY** (malign, [6265];
verified at the registrar's seat: **145 ids declared in `MODEL_FAMILIES`, 159
registry rows, 14 with no `position`, and those 14 are exactly the ones
absent from the declaration** — 145 + 14 = 159).

    AI-Sweden-Models/gpt-sw3-6.7b        microsoft/phi-4
    NousResearch/Hermes-3-Llama-3.1-8B   microsoft/phi-4-reasoning
    NousResearch/Nous-Hermes-2-Mistral-7B-DPO   mosaicml/mpt-7b
    THUDM/glm-4-9b-hf                    mosaicml/mpt-7b-instruct
    cognitivecomputations/Dolphin3.0-Llama3.1-8B  openchat/openchat-3.5-0106
    cognitivecomputations/dolphin-2.6-mistral-7b-dpo
    teknium/OpenHermes-2.5-Mistral-7B    tiiuae/falcon-7b
                                          tiiuae/falcon-7b-instruct

**No position, so no relation, so no group, so no roster, so no
denominator.** This is `build_fleet.py`'s defect in the registry rather than
in a shard — that file's docstring: *a model absent from the plan is absent
from the denominator too.* **They are invisible to every relation-based plan
and nothing reports them missing.** RH has malign researching all 14; **the
declaration is the registrar's and the results come here first.**

**`is_model_pair` is POSITION-DEFINED, per RH's rule of 2026-08-12 that a model
pair is base→superego.** That is why it holds whether the contrast is a
declared edge or a derived one — and why it excludes one pair the frozen file
contains (below).

The producers behind those inputs:

```
scripts/build_model_registry.py   -> data/model_registry.json
scripts/build_lineage_map.py      -> data/lineage_map_models.json
scripts/build_movement_edges.py   -> CH malign_logits.movement_edges
scripts/lineage_representative_pairs.py --write
                                  -> data/lineage_representative_pairs.txt
```

## Where CH and the frozen file disagree, and who is right

Measured 2026-08-14: **7 pairs in CH and not in the file, 1 in the file and not
in CH.**

**The file's extra one is a rule violation and CH is right to drop it.**
`BAAI/Aquila2-7B > BAAI/AquilaChat2-7B` is base→**ego**, not base→superego:
AquilaChat2 is SFT-only, with no preference-optimisation stage (arXiv
2408.07410, read by RH 2026-08-12; the registry's `position=ego` was correct).
So **one of the file's 46 is not a model pair under the stated rule**, and any
count taken off that roster and called "base/aligned pairs" is off by one in a
way no arithmetic on the file will reveal.

**CH's extra seven are models the battery did not run**, and six of them have
full data:

```
gpt-sw3-6.7b-v2      0 rows both sides   <- gated; the only unmeasurable one
pythia-2.8b x4       365,571 / ~330,000  <- battery carries pythia-6.9b instead
Olmo-3-1025-7B       497,440 / 384,486
Llama-3.1-8B         334,667 / 252,861
```

**The file is frozen to `data/forced_arms_105_v3.json`'s membership, which is
the battery's population and not the campaign's current one.**

## The counting trap this exists to prevent

The roster has been reported as 37, 42, 21 and 32 in one evening, and as "52
pairs" and "46 lineages" in consecutive sentences with neither labelled. Both
were correct and they are not the same population.

**And there are two 46s.** The file's 46 is a property of a frozen artifact.
The registry answers the same question live and gives 47 declared, 46 with
data — agreeing with the file on 45 and differing by one member each way.

**A cross-lineage test wants LINEAGES, not pairs.** Anything reporting 52 rows
as independent observations is counting Falcon3-1B, -3B and -7B as three things
when the vendor's own card calls two of them compressions of the third.

## Why this was hard to find

All of the above was already correct and already written down — in the
docstring of `scripts/lineage_representative_pairs.py`, which by this
campaign's own channel ordering is read by *whoever runs or edits that
producer*. Ten consumers reach the population by hand-rolling
`os.path.join(ROOT, "data", "lineage_representative_pairs.txt")` and parsing
`base>aligned` themselves; none of them reads that docstring.
