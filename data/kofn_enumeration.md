# The bounded k-of-N enumeration

Commission [2181]. Mechanical; **no adjudication**. Produced 474a7ed.

THE BOUND: `data/lineage_map_models.json` holds **112 models, 48 family labels, 34 independent pretraining lineages**. Collapsing is monotone, so **any N above 34 over-counts by construction** — no roster reconstruction needed. N at or below 34 is *not* thereby cleared; it may still collapse.

Scanned `findings/*.md` + `README.md`: **66 claims** carrying a model-ish unit within 110 chars.

## Claims whose N exceeds the lineage bound

| file:line | claim | unit word | statistic | context |
|---|---|---|---|---|
| `findings/F20_addendum.md:393` | **24/42** | famil | descriptive | answering, 53 percent containing "from". The sweep stopped at 24 of 42 families |
| `findings/F11_addendum.md:41` | **14/35** | base | descriptive | | Frame-exit | 14/35 | 40% | Contradiction dampens in-frame mass below the single-pole baseline | |

## Claims within the bound (not thereby cleared)

| file:line | claim | unit word | statistic |
|---|---|---|---|
| `README.md:733` | 30/30 | famil | descriptive |
| `findings/F13_jakobsonian_axes.md:165` | 30/30 | famil | descriptive |
| `findings/F36_violence.md:102` | 16/30 | base | descriptive |
| `findings/F20_generation_drift.md:88` | 28/29 | base | descriptive |
| `findings/F20_generation_drift.md:290` | 28/29 | model | descriptive |
| `findings/F20_generation_drift.md:290` | 26/29 | model | descriptive |
| `findings/F20_generation_drift.md:290` | 25/29 | model | descriptive |
| `findings/F20_third_person.md:242` | 28/29 | model | descriptive |
| `findings/F20_third_person.md:242` | 26/29 | model | descriptive |
| `findings/F20_third_person.md:242` | 25/29 | model | descriptive |
| `README.md:81` | 26/27 | pair | INFERENTIAL |
| `findings/F02_cross_family_logits.md:9` | 26/27 | pair | INFERENTIAL |
| `findings/F20_addendum.md:315` | 20/27 | famil | INFERENTIAL |
| `findings/F36_capstone.md:103` | 22/27 | pair | INFERENTIAL |
| `findings/F36_ledger.md:38` | 22/27 | pair | INFERENTIAL |
| `README.md:1680` | 8/25 | model | descriptive |
| `findings/F20_generation_drift.md:7` | 24/25 | base | INFERENTIAL |
| `findings/F20_generation_drift.md:86` | 24/25 | base | INFERENTIAL |
| `findings/F23_reasoning_distillation.md:184` | 8/25 | model | descriptive |
| `README.md:1195` | 21/22 | base | INFERENTIAL |
| `findings/F20_addendum.md:244` | 19/22 | famil | descriptive |
| `findings/F20_addendum.md:244` | 15/22 | famil | descriptive |
| `findings/F20_addendum.md:246` | 17/22 | base | descriptive |
| `findings/F20_addendum.md:246` | 15/22 | base | descriptive |
| `findings/F20_who_are_you.md:11` | 21/22 | base | INFERENTIAL |
| `findings/F20_addendum.md:373` | 8/20 | famil | INFERENTIAL |
| `README.md:2235` | 13/19 | famil | descriptive |
| `README.md:2302` | 9/19 | famil | INFERENTIAL |
| `README.md:2319` | 4/19 | famil | descriptive |
| `findings/F28_resistance_trajectories.md:124` | 13/19 | famil | descriptive |
| `findings/F28_resistance_trajectories.md:191` | 9/19 | famil | INFERENTIAL |
| `findings/F28_resistance_trajectories.md:208` | 4/19 | famil | descriptive |
| `README.md:165` | 13/17 | famil | descriptive |
| `README.md:201` | 13/17 | famil | descriptive |
| `findings/F05_logit_lens.md:11` | 13/17 | famil | descriptive |
| `findings/F05_logit_lens.md:48` | 13/17 | famil | descriptive |
| `README.md:2015` | 10/14 | famil | descriptive |
| `README.md:2015` | 4/14 | famil | descriptive |
| `findings/F26_census.md:53` | 10/14 | famil | descriptive |
| `findings/F26_census.md:53` | 4/14 | famil | descriptive |
| `README.md:2339` | 5/12 | famil | descriptive |
| `findings/F20_third_person.md:96` | 12/12 | famil | descriptive |
| `findings/F28_resistance_trajectories.md:228` | 5/12 | famil | descriptive |
| `README.md:999` | 9/10 | base | descriptive |
| `README.md:1226` | 7/10 | base | descriptive |
| `README.md:1226` | 1/10 | base | descriptive |
| `findings/F18_shannon_entropy.md:57` | 9/10 | base | descriptive |
| `findings/F20_who_are_you.md:44` | 7/10 | base | descriptive |
| `findings/F20_who_are_you.md:44` | 1/10 | base | descriptive |
| `README.md:81` | 8/8 | pair | INFERENTIAL |
| `findings/F02_cross_family_logits.md:9` | 8/8 | pair | INFERENTIAL |
| `findings/F36_ledger.md:26` | 6/7 | pair | descriptive |
| `findings/F13_jakobsonian_axes.md:325` | 6/6 | famil | descriptive |
| `findings/F36_capstone.md:35` | 4/6 | famil | INFERENTIAL |
| `findings/F36_capstone.md:52` | 4/6 | famil | descriptive |
| `findings/F36_capstone.md:52` | 5/6 | famil | descriptive |
| `findings/F36_ledger.md:39` | 4/6 | famil | descriptive |
| `README.md:1464` | 4/5 | base | descriptive |
| `findings/F22_circuit_decomposition.md:92` | 4/5 | base | descriptive |
| `findings/F13_jakobsonian_axes.md:325` | 4/4 | famil | descriptive |
| `README.md:1235` | 3/3 | model | INFERENTIAL |
| `README.md:1235` | 3/3 | model | INFERENTIAL |
| `findings/F20_who_are_you.md:53` | 3/3 | model | INFERENTIAL |
| `findings/F20_who_are_you.md:53` | 3/3 | model | INFERENTIAL |

## Limits, stated

- **Digits only.** "across 39 families", "nine families", "all 47 prompts" are invisible to this regex. The residue is a reading task.
- **`unit word` is the nearest match in the window, not the claim's actual denominator.** A claim can say "model" and count prompts.
- **`statistic` is keyword-detected.** F20's headline lived in a YAML `description:` field with `p<0.0001` inside it; that is caught here, but a p reported one line away is not.
