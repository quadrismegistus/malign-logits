# M01 — the registrations, by letter

**The record: what exists, what it says, whether it ran, what came back.** No interpretation — which question each registration answers, and how they fit together, is the README's job. Caveats, supersessions and what a number may not say are `ledger.md`'s.

**Why this file has to exist.** A registration is frozen before its numbers exist and locked at the freeze. **The run happens afterwards, so the document can never record its own result.** That is why every registration below except N and O still says "draft; not in force as declared" in its own status line, including six that have run with artifacts on disk. It is not neglect: **a locked file cannot update itself, so the run state can only live outside it.** This is that outside.

Letters run B–G then L–O. **There is no A, H, I, J or K** — the letters were never assigned, not retired.

| | Subject | Frozen at | Ran | Result | Artifact |
|---|---|---|---|---|---|
| **B** | the high-mass decomposition (movement within the high-mass set and within the tail, separately) | `06186c42f9ff46e0` (v13) | **no** | — | none |
| **C** | valence/dominance de-extremification, general corpus | `06f0272d7f21b901` (v6) | yes | **H2 valence CONFIRMED** general (+0.025 residualised, p 0.0012); **H3 dominance dead**; H1 general (sweetening) **still blind, never emitted** | **none machine-readable** — two `.txt` transcripts only |
| **D** | the displacement-site test, three dimensions, minimal pairs | `8375ff4c8335d979` (v6) + Amendment A `03e43f65085488a0` | yes | **H1 signed NOT SUPPORTED** (D +0.01272, p 0.8896); **arousal NOT SUPPORTED and the null is QUOTABLE** (D −0.02841, p 0.9967) | `result_d_stage1` `15b529a7d9261c8b`, `stage2` `519ef7f9877fa9f7` |
| **D2** | the two extremity arms at sites | `d9fedcba3857d100` | yes | **BOTH CONFIRMED** — valence-extremity D +0.01525 (p 0.0076), dominance-extremity D +0.01624 (p 0.0114) | `result_d2_stage1` `b5599a359fbf5265`, `stage2` `756eba00a0cfff4a` |
| **D3b** | decomposing D2's effect against pool extremity | `f02f59d403906503` + Amendment A `b69b3e7d3e5edf68` + **Amendment B `6c21db65ce1d2ae2`** | yes | a **bracket, no significance test by design**: pool-associated share bounded above (valence +0.5609; dominance −1.1293, opposite sign); pool-independent bounded above (valence +1.3372; dominance +0.7567). **"Just reflects pool extremity" is dead; mediation stays open** | `result_d3b_stage1` `3a7219afddb02569`, `stage2` `b95c7974cc65c1c0` |
| **E** | C's H2 on the GAP stratum | `b6198c89bdc6cd7b` (v3) | yes | 19 of 25 lineages, p 0.0073 on the blind gap arm | **none** — the producer prints and writes nothing |
| **F** | within-pair displacement RATE at transgressive sites | `4cb511ce320d90de` + Amendment A `0eab73a95be5d33e` | yes | **RATE NULL** (n=33 pair-sites, p 0.148) | `result_f_within_pair` `2adef31e8375ce50`, `result_f_collapsed` `50c3c4f21650a185` — **both cite registration `8ff56206deac048e`, the pre-re-freeze hash** |
| **G** | the same by MASS rather than sign | `0ca80e6bc2bf8323` | yes | **MAGNITUDE CONFIRMED** (d 0.748, p 0.00006) | `result_g_magnitude` `3c418f0c1be4453c` — **cites `efbab15841eae4c2`, the pre-re-freeze hash** |
| **L** | movement and fit-to-human on found prose | `72e4b4a94d7c467e` | yes | **no verdict by design (§L9).** Three rungs, z positive = alignment loses the human's word: argmax Z +2.6372, top20 +4.5220, retained +5.5820 (**retained tested on 31 clusters, not 34**) | `result_l_found_prose` `18d1b6c9ad2a37af`, `_primary` `693cf135ab473b8b`, `_tail_column` `e5a527acfd65e85e` |
| **M** | the perturbation null adjudicating L's gradient | `3506032d552438e4` | yes | **BOUNDARY BLUR, NOT TAIL CONTRACTION.** Overshoot Z −13.3170; escapes **declared UNDERPOWERED**; eviction rate falls 0.157 → 0.045 → 0.020 → 0.008 → 0.003 and is **exactly zero above the fifth headroom decile** | `result_m_column` `daf11fc743456f42`, `_primary` `e333bd58e5e1feb9` |
| **N** | mass-migration at full scale, English | `9fb5e13fd1c3b1c8`, commit `c7a101de` | yes | **SUBSTITUTION CONFIRMED.** 2,199 stimuli × 44 edges, 82,775 cells, 91% negative, **34/34 clusters agree**. Stouffer Z is a **FLOOR** | `result_n_primary` `8a2ce3fdf4950ff2`, commit `61c66090` |
| **O** | the same content in two languages | `cb8518528077f7d0`, commit `aa03cc82` | yes | **H1 SUPPORTED IN BOTH ARMS** (en 2277/365, zh 2463/141) under §O6's Chinese-origin bound. **H2 and H3 NOT SUPPORTED** — asymmetry, en confirming, zh clean coin-flip nulls (648/650, 652/646). Z is a **FLOOR** | `result_o_primary` `9b99725e8e76057b`, commit `66ecba4d` |

## Four things this table makes visible that no other document did

**1. Ten of the twelve registrations' own status lines are wrong about themselves**, and structurally so — D2, D3b, F, G, L and M all say "draft; not in force as declared" and have all run. B, C, D and E carry no status line at all. Only N and O say FROZEN, because they were drafted after the lesson.

**2. Two registrations have no machine-readable artifact.** C's producer emits two `.txt` transcripts; **E's writes nothing at all** — no `json.dump`, no output path. Their numbers live in prose, which is why a figure of C's can drift with nothing to diff it against.

**3. F and G's artifacts name registration hashes that are not on disk.** The 2026-08-03 re-freeze changed six registrations' dated status lines (`8ff56206…` → `4cb511ce…` for F, `efbab158…` → `0ca80e6b…` for G) without changing their specs. **The artifact-to-registration link is broken by hash and only a prose note in `ledger.md` explains it.** Both hashes are carried above so the link is followable from here.

**4. B has never run.** It is frozen and has no artifact.

## Maintenance

**Everything in the first six columns is derivable** — filenames, hashes, the presence of an artifact. **The Result column is not**, and is written by hand from the artifact or the frozen text. When this file and an artifact disagree, **the artifact governs.**
