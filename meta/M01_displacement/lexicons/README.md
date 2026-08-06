# External semantic lexicons for M01

Three JSONs. The first two are the raw resources; the third is the one you probably want.

    general_inquirer.json          8,269 words -> 182 Harvard GI categories
    wordnet_verb_supersenses.json  11,529 verb lemmas -> 15 WordNet supersenses
    m01_token_lexicon.json         the 685 M01 tokens, resolved against both, with counts

## general_inquirer.json

Built from `inquireraugmented.xls`. **`inquirerbasic.xls` is identical in content**: zero differing cells over 186 columns and 11,788 entries. The augmented file adds only a totals row and spells two column names differently (`Othrtags`, `SklTOT`). Neither is fuller than the other.

**It is lemma-only for REGULAR inflections and not for irregular ones**, which is the distinction that matters and which I got wrong at first. None of ten probe verbs has an `-ed`, `-s` or `-ing` form and there are no plurals (`knife` yes, `knives` no). But 180 entries carry an `ED` tag in `Othrtags`, covering 162 words: `SAID#1` is `DAV SUPV ED PFREQ`, `FOUND#1` is `IAV SUPV ED`, and so are `SAW#1` and `LEFT#3`. GI lists irregular pasts as their own senses and marks them.

**So look the token up first and prefer its `ED`/`SUPV` sense; fall back to the lemma only when the surface form is absent.** Unioning all senses of a surface form is what goes wrong: `found` unions the past of *find* with *to establish* and the noun. Resolving through the `ED` tag gives `saw` -> `Perceiv` and `left` -> `Travel`, where the union gave the cutting tool and the direction. 91 of the 685 M01 tokens change under this rule. The `senses` block in the JSON carries every sense separately with its tags, so you can apply your own rule.

Sense-tagged entries (`ABANDON#1`, 4,750 of them) are split on `#`, lower-cased, and their categories unioned. `n_senses` flags the 1,465 words where that union is coarse.

**Known gap, and it is not random.** 56 lexical verbs in the M01 vocabulary are absent, including `raped`, `desecrated`, `handcuffed`, `stomped`, and the modern/digital verbs `scanned`, `deleted`, `captioned`, `scrolled`. GI comes from the 1960s Lasswell tradition and its coverage of explicit violence is thin. Anything relying on GI alone will silently drop the transgressive end of the vocabulary.

## wordnet_verb_supersenses.json

The 15 verb lexicographer files, which are far more usable here than raw synsets: no word-sense disambiguation problem, and the granularity matches the question.

`first` is `wn.synsets(lemma)[0]`, WordNet's own sense-frequency order. An earlier build took it from `wn.all_synsets()`, which iterates in FILE order and gave a different, meaningless answer for half the vocabulary: `throw` came out `cognition` rather than `contact`. Use `senses_in_order` if you need more than the first.

**It is too coarse for the distinction M01 cares about.** `whispered`, `shouted`, `said` and `told` all share `communication`, and in this corpus the first two rise while the last two fall. The supersense cannot see that, so it reports `communication` as slightly falling overall.

What it does give, and it is worth having: `contact` runs 33.5% of faller slots against 18.5% of riser slots, spread across `put`, `threw`, `kissed`, `sat`, `smashed` rather than carried by one word.

## m01_token_lexicon.json

All 685 faller/riser types from `data/r_population_k2.parquet` (684 stems, 1,361 cells, 5,976 pairs), each with `as_faller`, `as_riser`, `lean`, the CLAWS tag, and its resolution into both lexicons.

**The two lexicons need OPPOSITE lookup policies, because only one of them tags its inflections.**

GI: token first, preferring the `ED`/`SUPV` sense for verbs, lemma only as fallback. WordNet: verb lemma first for `vv*` tokens, because WordNet has no inflection marking at all and the surface forms `found`, `felt` and `saw` are separate lemmas meaning *establish*, the fabric, and the cutting tool. Each resource is queried through its own metadata rather than through one policy imposed on both.

The 48 types where the policies diverge are 15.8% of token slots. For most of them WordNet gives the same supersense either way, because the irregular past is a lemma of the same synset (`went`->motion, `began`->change, `told`->communication). Only `found`, `saw` and `felt` actually flip.

**Historical note on why the lemma-first rule exists at all.** This matters more than it sounds. `found`, `felt`, `said` and `left` are all valid lemmas in their own right (to found an institution, to felt wool, the said document), so a surface-form-first lookup returns the wrong entry precisely on the commonest past-tense tokens. Before this fix `found` resolved to `social`, `felt` to `contact` and `said` to GI's adjective entry. `found` is the single most frequent riser in the corpus at 321.

Coverage of lexical verbs: GI 88% of types and 97% of slots, WordNet 100% of types.

## Use them together, not separately

Neither is adequate alone. GI has no `raped`; WordNet cannot distinguish `whispered` from `said`. Where a word is missing from one the other still has it, and where two independently built schemes agree the category is not an artifact of either. Both are also external and pre-existing, so unlike a taxonomy induced from this data they cannot have been fitted to the answer.
