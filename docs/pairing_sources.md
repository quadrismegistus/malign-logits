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

## The gap that started this was NEITHER of those — it is the `ambiguous` flag

**Four attributions were proposed before anyone enumerated the members.** lacan said the pairing source; malign said the declared floors; malign then said the coverage predicate. All three name real differences between the two lists. **None of them is this gap.**

Measured, one source, one coverage predicate (≥1 row in both languages in `f11_l2`):

    all entries in base_aligned_pairs.json      29
    non-ambiguous only                          26      <- the declared population
    ---------------------------------------------------
    difference                                   3

    EleutherAI/pythia-2.8b   > archangel_sft-dpo_pythia2-8b   ambiguous, ruled
    allenai/Olmo-3-1025-7B   > Olmo-3-7B-Instruct-DPO         ambiguous, ruled
    meta-llama/Llama-3.1-8B  > Llama-3.1-8B-Instruct          ambiguous, ruled

**`ambiguous` IS NOT A PAIRING FACT, IT IS AN ADJUDICATION — and it lives in only one of the two sources.** The registry records a relation whether or not a curator considered the pair contested, so the two lists can have symmetric difference **0 on identity** and still yield different populations. **A transitive walk cannot reproduce a curated exclusion, however correct its graph traversal.** Any plan pairing from the registry must state what it does about ambiguity, because the flag it would need is not there.

### `ambiguous` and `ruled` are TWO fields — reading only the first silently under-includes

Measured over all 54 entries of `base_aligned_pairs.json`:

    (ambiguous, ruled)      count
    (False, False)            51
    (True,  True)              3
    (True,  False)             0      <- NOTHING is unresolved

**`ambiguous: true` in this file never means "contested and unusable". It means A CURATOR FACED A CHOICE AND MADE IT**, and all three carry an explicit `candidates` list recording what was chosen from what:

    pythia-2.8b       -> archangel_sft-dpo_pythia2-8b   from 4 candidates
    Olmo-3-1025-7B    -> Olmo-3-7B-Instruct-DPO         from 2 candidates
    Llama-3.1-8B      -> Llama-3.1-8B-Instruct          from 7 Tulu variants plus meta-llama's own

So `if not p.get('ambiguous')` **drops three pairs that were already resolved**, including `Llama-3.1-8B > Llama-3.1-8B-Instruct` — an entirely standard pair whose only ambiguity was which of seven Tulu variants to prefer. **The defensible filter is `ambiguous AND NOT ruled`, which today excludes nothing** and is the right predicate for future entries. This cost lacan's cross-lingual finding a declared population of 25 where 28 was correct; re-run as a sensitivity ([5850]) it moved medians by hundredths and improved every sign count, so nothing in the reading changed.

**Neither field exists in the registry**, so a transitive walk can express neither the ambiguity nor its resolution.

**THE RULE, and it is the cheapest one on this page: WHEN TWO COUNTS DISAGREE, ENUMERATE THE DIFFERING MEMBERS BEFORE NAMING A MECHANISM.** Three names took one command. Four rounds of plausible mechanisms took an evening and produced three wrong answers, each of which was a true statement about the sources.

## Custody note on `f11_l2`

**`pair` and `role` are EMPTY** on all 228,520 rows — one distinct blank value each, against `passage`'s 42 and 2. **A join on either returns one group and no error.** Pairing must come from model names through one of the two sources above.
