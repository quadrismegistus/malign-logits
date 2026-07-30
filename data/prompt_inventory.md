# Prompt inventory

`data/prompt_inventory.csv` — every prompt scored into `true_word_probs`, with its registered category and slot where one exists.

**Population: 604 distinct prompts, 13,940 cells, read 2026-07-30 10:23 local.**

**THE STORE IS WRITTEN WHILE THE ROSTER RUNS.** Two reads twenty minutes apart gave 603/13,693 and 604/13,782. A count over a growing store carries its read time or it is a claim about nothing datable — rebuild rather than quote these.

## Columns

| column | meaning |
|---|---|
| `prompt` | exact string; the join key everywhere |
| `source` | `DEFAULT` / `INSTITUTIONAL` / `CHINESE` / `UNMAPPED` |
| `prompt_id` | registry key (`sexual_explicit_3`); empty if unmapped |
| `category` | id with trailing index stripped; empty if unmapped |
| `slot` | hand grammar assignment: ACT/NARR/REF/UTTER/RESULT/SENSE |
| `n_models`, `n_cells` | coverage (equal: one cell per model) |
| `n_words`, `script` | shape; `script` is CJK vs latin |

## Mapped — 146 prompts, 8,474 cells

Three parallel batteries over the same nine content categories plus eleven institutional roles.

| source | category | prompts | cells |
|---|---|---|---|
| CHINESE | sexual_liminal | 7 | 98 |
| CHINESE | neutral | 7 | 98 |
| CHINESE | violence_liminal | 5 | 70 |
| CHINESE | substance | 5 | 70 |
| CHINESE | sexual_explicit | 5 | 70 |
| CHINESE | violence_explicit | 5 | 70 |
| CHINESE | death | 5 | 70 |
| CHINESE | profanity | 5 | 70 |
| CHINESE | power | 5 | 70 |
| CHINESE | institutional_labor_worker | 5 | 70 |
| CHINESE | institutional_labor_mgmt | 5 | 70 |
| CHINESE | institutional_govt_agency | 2 | 28 |
| CHINESE | institutional_govt_citizen | 2 | 28 |
| CHINESE | institutional_housing_tenant | 2 | 28 |
| CHINESE | institutional_housing_landlord | 2 | 28 |
| CHINESE | institutional_police_officer | 1 | 14 |
| CHINESE | institutional_medical_doctor | 1 | 14 |
| CHINESE | institutional_political_citizen | 1 | 14 |
| CHINESE | institutional_political_party | 1 | 14 |
| CHINESE | institutional_medical_patient | 1 | 14 |
| CHINESE | institutional_police_citizen | 1 | 14 |
| DEFAULT | sexual_liminal | 7 | 714 |
| DEFAULT | neutral | 7 | 714 |
| DEFAULT | sexual_explicit | 5 | 512 |
| DEFAULT | violence_liminal | 5 | 512 |
| DEFAULT | profanity | 5 | 510 |
| DEFAULT | substance | 5 | 510 |
| DEFAULT | death | 5 | 510 |
| DEFAULT | violence_explicit | 5 | 510 |
| DEFAULT | power | 5 | 510 |
| INSTITUTIONAL | institutional_labor_worker | 5 | 512 |
| INSTITUTIONAL | institutional_labor_mgmt | 5 | 510 |
| INSTITUTIONAL | institutional_govt_agency | 2 | 204 |
| INSTITUTIONAL | institutional_govt_citizen | 2 | 204 |
| INSTITUTIONAL | institutional_housing_tenant | 2 | 204 |
| INSTITUTIONAL | institutional_housing_landlord | 2 | 204 |
| INSTITUTIONAL | institutional_police_officer | 1 | 102 |
| INSTITUTIONAL | institutional_medical_doctor | 1 | 102 |
| INSTITUTIONAL | institutional_political_party | 1 | 102 |
| INSTITUTIONAL | institutional_political_citizen | 1 | 102 |
| INSTITUTIONAL | institutional_medical_patient | 1 | 102 |
| INSTITUTIONAL | institutional_police_citizen | 1 | 102 |

## Slot grammar

Every mapped prompt carries one; **no unmapped prompt does.**

| slot | prompts |
|---|---|
| ACT | 68 |
| REF | 24 |
| NARR | 22 |
| UTTER | 18 |
| SENSE | 8 |
| RESULT | 6 |

## UNMAPPED — 458 prompts, 5,466 cells

**39% of all cells scored, with no category — so every category-stratified analysis silently drops them, including the entire minimal-pair design, whose only purpose is a contrast.**

| shape | prompts | cells | what it is |
|---|---|---|---|
| 7-15 words | 303 | 4,591 | Set D narrative variants — the bulk |
| 3-6 words | 55 | 675 | minimal pairs (captive/free, desire/…) — the F36 line |
| 16+ words | 100 | 200 | literary passages; 2 models only, a side experiment |

**Every Chinese prompt IS mapped** (`CHINESE_PROMPTS` covers all of them); it is the Set D and F36 material that is not.

Rebuild: `scripts/build_prompt_inventory.py`.
