# Two sources for "base > aligned pair", and what separates them

Settled 13 Aug 2026 at the malign seat, on lacan's [5846] ask. Two producers asked one question, got 29 and 26, and **both seats' explanations of the gap were wrong** — lacan attributed it to the pairing source, malign to the declared floors. It is neither.

## The two sources

    data/base_aligned_pairs.json    54 pairs. CURATED and adjudicated: one aligned
                                    endpoint per base, carrying `stage`, `ambiguous`,
                                    `ruled`, `candidates`, `warn_sft_as_aligned`.
    data/model_registry.json        203 relations. The transitive walk below yields 64.

## The transitive recipe (the part worth reusing)

**`dpo_of` names the SFT, not the base.** A direct read of `dpo_of` finds 16 pairs in `f11_l2`; walking `sft_of` upward to the root reaches 29.

```python
DERIV = {'dpo_of','ppo_of','kto_of','slic_of','rlvr_of','sft_of'}
parent = {r['child']: (r['parent'], r['relation'])
          for r in reg['relations'] if r['relation'] in DERIV}

def root(m):                      # cycle-safe
    seen = set()
    while m in parent and m not in seen:
        seen.add(m); m = parent[m][0]
    return m

pairs = {(root(c), c) for c, (p, t) in parent.items()
         if t != 'sft_of' and root(c) != c}
```

## Where the two genuinely disagree, and it IS definitional

- **The registry walk emits every aligned descendant** — `archangel_sft-kto/ppo/slic`, `OLMo-2-Instruct`, `Olmo-3-7B-Think`, `Think-DPO`. 11 pairs the JSON does not carry.
- **The JSON carries one the walk drops: `BAAI/Aquila2-7B > BAAI/AquilaChat2-7B`.** The relation exists in the registry as **`sft_of`**, and the recipe above skips SFT-terminal endpoints. The JSON includes it with `stage: 'sft'` and an explicit `warn_sft_as_aligned` flag — **its curators considered the case and ruled the other way.**

**So the open question is: IS AN SFT-ONLY CHECKPOINT AN ALIGNED ENDPOINT?** For a family whose only post-training release is an SFT, the JSON says yes. The recipe says no. Neither is obviously right and **a plan must declare which it means**, because for some families it changes n.

## The gap that started this was NEITHER of those

**Within `f11_l2` coverage the two sources agree EXACTLY: 29 and 29, zero symmetric difference.** The 29-vs-26 gap is entirely in the **coverage predicate** — malign's "≥1 row in both languages in `gen_sequences`" against lacan's passage-level test. Not the source, not the floors.

**The lesson is the shape, not the number: two seats each proposed a plausible cause for a discrepancy, both causes were real differences between the sources, and neither was THE cause.** A real difference in the right place is not evidence that it produced the observed gap — decompose before attributing.

## Custody note on `f11_l2`

**`pair` and `role` are EMPTY** on all 228,520 rows — one distinct blank value each, against `passage`'s 42 and 2. **A join on either returns one group and no error.** Pairing must come from model names through one of the two sources above.
