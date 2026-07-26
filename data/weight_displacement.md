# Weight displacement across alignment stages

Companion to `data/weight_displacement.csv`. Produced 2026-07-25 for the F38
containment join (desktop-side). Read this before using the numbers.

## What is measured

For a stage pair (checkpoint A -> checkpoint B), the relative Frobenius
displacement

    ||W_B - W_A||_F / ||W_A||_F

summed over all tensors present in both checkpoints with matching shape.
Embedding and unembedding tensors (`embed`, `wte`, `wpe`, `lm_head`) are
excluded, because vocabulary-sized matrices otherwise dominate the norm and
their scale differs by tokenizer.

Computed directly from the local HuggingFace cache (safetensors, with a
`torch.load` fallback for checkpoints released as `.bin`). No GPU, no
inference, no downloads.

## Columns

| column | meaning |
|---|---|
| `family` | model family key |
| `arch_family` | architecture lineage (llama, mistral, neox, olmo2, olmo3, olmoe, qwen2) |
| `stage_pair` | the two checkpoints, as stage labels |
| `stage_shape` | what the stage actually is: sft / dpo / ppo / kto / slic / rlvr / sft+dpo / instruct_compound / sft_component_X |
| `cross_lab_comparable` | whether this row may be compared across labs (see below) |
| `model_a`, `model_b` | HuggingFace IDs |
| `rel_frobenius` | the measurement |
| `n_tensors` | tensors entering the sum |
| `max_tensor_rel`, `max_tensor` | the single most-displaced tensor |
| `flag` | anomalies and scope notes |

## What this is a proxy for, and what it is not

It is a crude proxy for **training intensity**: how far the optimizer moved the
weights. It conflates learning rate, number of steps, number of epochs, and
optimizer. It cannot distinguish many small steps from few large ones, and it
says nothing about *where* in function space the model went.

It is not a measure of alignment strength, behavioral effect, or distributional
displacement. Those are measured elsewhere (JS divergence at the logit level,
rated dimensions at the passage level). The point of this table is precisely to
sit *beside* those measures so that intensity can be separated from effect.

## Comparability rules (important)

**`cross_lab_comparable = yes`** — terminal alignment stages taken from a
released SFT checkpoint (dpo, ppo, kto, slic, rlvr, sft+dpo). These are the
rows to compare across families.

**`cross_lab_comparable = no`** — any `base -> X` row. Labs differ in how much
mid-training and annealing sits inside the artifact they label "base," so these
numbers measure the gap between two release decisions as much as a training
run. Two illustrations in the table: `olmo3_7b base->sft` is 0.263, five times
Amber's, and `olmo2_1b base->sft` comes out at 1.096 — larger than the norm of
the weights themselves, which means the released 1B base is almost certainly
not the checkpoint the instruct line started from. That row is flagged and
should be excluded, not interpreted.

**`cross_lab_comparable = within_family_only`** — the Tulu leave-one-out SFT
rows. Same base, same recipe, one data component removed; comparable to each
other, not to anything else.

**Cross-architecture caveat.** Weight scales differ by architecture, so the
metric is strictly commensurable only within an `arch_family`. Llama-lineage
comparisons (Amber, Beaver, Tulu, Llama 3.1, CT-LLM, neo, SmolLM2, DeepSeek)
are the trustworthy ones. Comparisons between, say, a GPT-NeoX model and an
OLMo model are indicative at best.

## Recipe provenance where it is documented

Two families have published hyperparameters that explain their position:

- **AmberChat** (LLM360 card): WizardLM-evol-instruct-V2 (143k) + ShareGPT
  (90k) = 233k rows, learning rate 2e-5, **three epochs**, batch 2 with
  gradient accumulation 16, max length 2048, full fine-tune.
- **AmberSafe** (LLM360 card): initialized from AmberChat; PKU-SafeRLHF
  filtered to pairs where the two safety labels disagree; procedure is
  supervised fine-tuning on that data **followed by** DPO. Hence
  `stage_shape = sft+dpo`, not `dpo`.
- **Archangel** (KTO paper, arXiv 2402.01306): Anthropic-HH + SHP + OASST,
  one epoch, RMSProp, DPO learning rate 5e-7, effective batch 32, full
  fine-tune.

Amber's SFT stage therefore runs at roughly forty times the learning rate of a
standard preference-optimization run, for three times the epochs. Its position
at the top of the table is documented, not mysterious.

## The comparison this table was built for

Amber's two stages are nearly identical in intensity and differ in curriculum:

| stage | displacement | curriculum |
|---|---|---|
| base -> AmberChat | 0.0472 | WizardLM + ShareGPT, no safety data |
| AmberChat -> AmberSafe | 0.0499 | PKU-SafeRLHF, SFT + DPO |

A ratio of 1.06. Any difference in what the two stages install cannot be
attributed to how hard they were trained. This is the only design in the
census where intensity is held fixed and curriculum varies, and it is
within-family, so it survives the cross-lab and cross-architecture caveats
above.

## The two results the table produces on its own

**1. The intensity range, and Amber's position in it.** Sorting the
`cross_lab_comparable = yes` rows (terminal alignment stages only):

| stage | displacement |
|---|---|
| archangel dpo / ppo / slic / kto | 0.00037 - 0.00058 |
| tulu3 dpo | 0.00085 |
| pythia_hh dpo | 0.00110 |
| olmoe rlvr / zephyr dpo / ctllm dpo / olmo3 dpo | 0.00132 - 0.00151 |
| olmoe dpo / olmo3 rlvr / tulu3 rlvr / olmo2 rlvr | 0.00176 - 0.00240 |
| olmo2_1b dpo | 0.00368 |
| beaver ppo | 0.00437 |
| **amber sft+dpo** | **0.04987** |

Preference optimization as normally practiced is a 0.001 to 0.004
intervention, and this holds across seven labs, four architectures, and
five algorithms (DPO, PPO, KTO, SLiC, RLVR). Amber's safety stage is
0.050: eleven times the next largest and roughly thirty times the
cluster. Amber is not at one end of a continuum of safety training; it is
off the scale of ordinary preference optimization, which is what its
published recipe (lr 2e-5, three epochs, and an SFT pass before the DPO)
predicts. Any claim resting on Amber alone should say this.

**2. The safety component is not special in size.** The Tulu leave-one-out
rows, same base and recipe with one data component removed:

| component removed | displacement from full SFT |
|---|---|
| safety (WildGuardMix + WildJailbreak, ~100k) | 0.0159 |
| persona (~285k) | 0.0175 |
| math (NuminaMath-TIR, 64k) | 0.0178 |
| wildchat (~100k) | 0.0210 |

Removing the safety data perturbs the SFT checkpoint slightly *less* than
removing math, persona, or WildChat. In weight-space magnitude the safety
curriculum is unremarkable: it is one data component among several, and
the smallest of the four. Whatever is distinctive about it is therefore in
*what* it installs, not in how far it moves the model. This is a magnitude
result only; it says nothing about behavioral effect, and the
corresponding passage-level containment isolation is null at current n.

## What this table cannot do

It cannot support a regression of behavioral effect on intensity across
families. As of the 2026-07-25 audit the passage-level containment effect is
robust in one family, so such a fit would have a single informative point and
a cloud of nulls. The table is for the within-family comparison above, for
stating the two-order-of-magnitude intensity range across published alignment
runs, and for sizing the leave-one-out safety component. Not for a curve.
