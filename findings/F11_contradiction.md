---
status: rescoped
grade: B
date: 2026-07-20
role: finding
superseded_by: F11_addendum
instruments: [logit-mass, intervention]
families: [olmo, amber, llama, qwen, tulu, zephyr, olmo-tiny, deepseek-7b, pythia, qwen-tiny, smol]
chapters: [ch03, ch11]
data: [contradiction_cross_family.csv, contradiction_detail.csv]
scripts: [f11_cross_family.py]
---
> **PROMOTION CONDITION ([2198], RH's standing scope rule).** A canonical claim
> in a `meta/` campaign runs on **all families we have** under a declared
> admissibility rule — never a hand-picked subset. This finding measures
> **11 family labels = 9 independent lineages** (`llama`+`tulu` on Llama-3.1-8B; `qwen`+`qwen-tiny` on the Qwen2.5 size ladder — the first collapse is visible by eye, the second only through the map), against a roster of
> **49 labels / 34 lineages** ([[data/lineage_map_models.json]]). It is
> therefore an **F-series observation** and cites its subset. **If the ratio or mechanism tables is
> promoted to M-canonical, the promotion INCLUDES the full-roster re-run** —
> promotion is re-measurement, not relabelling. Priced when the cloud grid
> frees.
>
> **AND THE PROMOTION HAS TWO CONDITIONS, NOT ONE** ([2199].3, ruled [2201].2).
> Alongside the full-roster run: the [2111].2 **substrate repairs** — the
> intervention has no named substrate and the coherence confound (r = −0.032)
> is computable from no named file. **A full-roster run produces new numbers at
> scale and leaves those two exactly as unsourced**, so the roster run must not
> be mistaken for the whole clearance.

# F11: Contradiction Tolerance — Cross-Family Replication

## The question

Freud: the primary process has no principle of non-contradiction. Lacan: the unconscious is "structured like a language" — it has its own logic. Fazi: the LLM is a Kantian/Hegelian synthesis, "a unity that unifies." We test whether alignment imposes exclusive disjunction (Deleuze's XOR) on a system that defaults to inclusive disjunction.

## Instrument

For each contradiction pair (e.g. "She loved him deeply and wanted to" / "She hated him deeply and wanted to"), compute the logit distribution for the combined prompt ("She loved him and hated him and wanted to") and compare against: (a) the average of the two individual distributions (superposition), and (b) each individual distribution (resolution).

**Ratio** = JS(AB, mean) / min(JS(AB, A), JS(AB, B)).
- Ratio < 1: model treats contradictions additively (superposition / inclusive disjunction)
- Ratio > 1: model resolves toward one pole (exclusive disjunction)

**Instrument limitation:** the ratio measures distance from the blend vs distance from each pole. It CANNOT distinguish pole-picking (resolving to one side) from frame-exit (abandoning the contradictory scene). The original F11 result that alignment "exits the frame rather than picking a pole" comes from the causal intervention (nnsight), not the ratio. The ratio captures the default operating point; the intervention captures the geometric structure. Both instruments are needed.

## Cross-family replication (11 families, 11 pairs)

**The gradient holds universally.** All base models tolerate contradiction (ratio < 1). Alignment shifts toward resolution in all substantially-aligned families.

| Family | Base | Aligned | Δ | Note |
|---|---|---|---|---|
| OLMo-tiny | 0.61 | 0.89 | +0.28 | Strongest shift |
| OLMo | 0.70 | 0.91 | +0.22 | |
| Zephyr | 0.82 | 1.01 | +0.18 | Crosses threshold — no safety data |
| Amber | 0.87 | 0.95 | +0.08 | |
| Qwen | 0.89 | 0.96 | +0.08 | |
| Tulu | 0.87 | 0.94 | +0.07 | |
| Llama | 0.87 | 0.92 | +0.05 | |
| DeepSeek | 0.76 | 0.77 | +0.003 | Null |
| SmolLM | 0.74 | 0.74 | +0.002 | Null |
| Pythia | 0.72 | 0.72 | −0.003 | Null |
| Qwen-tiny | 0.77 | 0.77 | −0.003 | Null |

**Coherence confound: dead.** Delta-ratio vs delta-coherence: r = −0.032 (p = 0.93). OLMo has the 2nd-largest resolution shift (+0.22) while its coherence DROPS (−0.10). Amber has the largest coherence gain (+1.22) but only moderate resolution (+0.08). Base-ratio vs base-coherence: r = 0.07 (p = 0.84) — low base tolerance is not flatness-from-incoherence.

## Stage decomposition

| Family | Base | SFT | DPO | RLVR | Δ(b→s) | Δ(s→d) |
|---|---|---|---|---|---|---|
| OLMo-tiny | 0.61 | 0.86 | 0.89 | 0.89 | +0.25 | +0.03 |
| OLMo | 0.70 | 0.86 | 0.91 | 0.91 | +0.16 | +0.05 |
| Zephyr | 0.82 | 0.91 | 1.01 | — | +0.08 | +0.10 |
| Amber | 0.87 | 1.01 | 0.95 | — | +0.14 | −0.06 |
| Tulu | 0.87 | 0.85 | 0.94 | 0.92 | −0.01 | +0.08 |

**SFT drives the resolution shift** for OLMo (+0.16) and OLMo-tiny (+0.25). DPO adds incrementally. Zephyr: both stages contribute (SFT +0.08, DPO +0.10). Amber's SFT crosses the threshold (1.01) then DPO pulls back (0.95) — a curio.

**F11 joins the ego-constitution cluster:** instruction-following installs commitment/resolution, alongside deference (F21) and mild proceduralization. This is the first major finding organized by neither coherence nor safety-data style — it tracks instruction-tuning as such.

## Per-pair observations (flagged, not claims — n=3 per cell)

Some contradictions get MORE tolerated by alignment:
- Llama pleasure/pain: 1.24 → 0.78 (alignment increases superposition)
- Llama human/animal: 1.57 → 0.71 (alignment increases superposition)

Some contradictions get strongly resolved:
- Llama trust/fear: 0.35 → 1.59 (deepest superposition → strongest resolution)
- OLMo pleasure/pain: 0.35 → 0.94 (toward resolution)
- Zephyr beautiful/disgusting: 1.05 → 1.53 (already resolved, alignment strengthens)

The aggregate gradient hides pair-level heterogeneity. Content-structured disjunction is a hypothesis pending multi-prompt-per-pair replication.

## OLMo 3 7B stage-level detail (original F11)

| Model | Mean ratio | Interpretation |
|---|---|---|
| BASE | 0.69 | Strong superposition |
| SFT | 0.81 | Less superposition |
| DPO | 0.88 | Near resolution threshold |
| RLVR | 0.88 | Same as DPO |

## Causal intervention (nnsight, OLMo 3 7B)

The contradiction axis is equally linearly decomposable in base, SFT, and DPO. Intervention range at layer 28: base 0.734, SFT 0.714, DPO 0.707. Pushing the "loved and hated" representation toward hate boosts "kill" (+0.16), "hate" (+0.08), "murder" (+0.03) and suppresses "be" (−0.13), "marry" (−0.02), "love" (−0.02).

**Alignment changes the default operating point, not the axis itself.** The base model has the geometric capacity for contradiction resolution but defaults to superposition. Alignment shifts where the model sits on this axis without changing the axis.

## For the paper (Fazi refutation)

No base model synthesizes contradictions — the Hegelian reading is refuted across 11 families. The computational evidence supports Lacan: the base model's representation space is already structured with clean contradiction axes. What alignment adds is not logical structure but a preference for deploying it — installed by instruction-tuning (SFT), not safety data. Zephyr's threshold-crossing with zero safety data is the cleanest proof.

## Open (held for Ryan's gate)

- **Address-check:** raw vs template mode — does the contradiction ratio change under the chat template? The F36 three-system decomposition predicts different behavior.
- **Multi-prompt replication** of per-pair observations before content-structured claims.

## Data

- `contradiction_cross_family.csv` — 341 rows (11 families × 11 pairs × all stages)
- `contradiction_detail.csv` — 44 rows (OLMo original, 11 pairs × 4 stages)
- Scripts: `f11_cross_family.py`, `contradiction_all.py`, `contradiction_test.py`
- Notebook: `07_contradiction_intervention.ipynb`, `F11_contradiction.ipynb`
