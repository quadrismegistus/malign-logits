*Surprisal: median z of 1 refs (surprisal_pythia_1b_deduped)*  
*Drift: median z of 1 embedders (total_drift)*  

## Median z-scores by text type

| Text type | Surprisal (z) | 95% CI | Drift (z) | 95% CI | n |
|---|---|---|---|---|---|
| C20 fiction | +0.40 | [+0.33, +0.47] | +0.25 | [+0.09, +0.35] | 447 |
| Dream reports | +0.14 | [+0.06, +0.21] | +0.25 | [+0.17, +0.35] | 427 |
| Arxiv abstracts | +0.10 | [-0.01, +0.15] | -1.64 | [-1.74, -1.54] | 476 |
| **AI generations** | -0.10 | [-0.10, -0.09] | +0.14 | [+0.13, +0.15] | 74364 |
| Waking narratives | -0.49 | [-0.56, -0.45] | +0.29 | [+0.21, +0.37] | 500 |

## AI narrative-only: median z-scores by family × layer

| Family | Layer | Surprisal (z) | Drift (z) | n |
|---|---|---|---|---|
| olmo | BASE | +0.58 | +0.26 | 3245 |
| olmo-tiny | BASE | +0.56 | +0.27 | 3031 |
| olmo | SFT | +0.52 | +0.35 | 1883 |
| tulu | BASE | +0.49 | +0.39 | 3347 |
| olmo | DPO | +0.47 | +0.07 | 1751 |
| qwen-tiny | BASE | +0.27 | +0.25 | 1118 |
| pythia | BASE | +0.24 | +0.14 | 2333 |
| olmo-tiny | SFT | +0.24 | +0.21 | 2684 |
| amber | BASE | +0.17 | +0.30 | 1001 |
| zephyr | BASE | +0.15 | +0.14 | 2065 |
| smol | BASE | +0.13 | +0.08 | 2656 |
| olmo | RLVR | +0.12 | +0.13 | 1960 |
| tulu | SFT | +0.10 | +0.24 | 3190 |
| olmo-tiny | DPO | +0.09 | -0.09 | 3287 |
| olmo-tiny | RLVR | +0.03 | -0.06 | 3292 |
| qwen | BASE | -0.05 | +0.24 | 1045 |
| smol | DPO | -0.05 | +0.02 | 2637 |
| qwen-tiny | DPO | -0.13 | +0.32 | 1779 |
| pythia | SFT | -0.25 | -0.02 | 2283 |
| llama | BASE | -0.26 | +0.29 | 3462 |
| pythia | DPO | -0.33 | -0.05 | 2509 |
| zephyr | SFT | -0.50 | +0.15 | 2547 |
| tulu | RLVR | -0.52 | +0.00 | 3475 |
| tulu | DPO | -0.59 | +0.00 | 3474 |
| llama | DPO | -0.71 | +0.20 | 3910 |
| qwen | DPO | -0.74 | +0.04 | 2033 |
| amber | SFT | -0.85 | -0.05 | 1075 |
| zephyr | DPO | -0.85 | -0.04 | 2151 |
| amber | DPO | -1.10 | -0.31 | 2065 |

## AI narrative-only: aligned − BASE Δ (median z, 95% bootstrap CI)

| Family | Δ surp | 95% CI | Δ drift | 95% CI | sig | n |
|---|---|---|---|---|---|---|
| amber | -1.27 | [-1.33, -1.20] | -0.61 | [-0.70, -0.50] | *** | 1001+2065 |
| tulu | -1.01 | [-1.05, -0.97] | -0.39 | [-0.45, -0.32] | *** | 3347+3475 |
| zephyr | -1.00 | [-1.06, -0.94] | -0.18 | [-0.26, -0.13] | *** | 2065+2151 |
| qwen | -0.69 | [-0.78, -0.61] | -0.21 | [-0.31, -0.13] | *** | 1045+2033 |
| pythia | -0.57 | [-0.61, -0.53] | -0.19 | [-0.25, -0.12] | *** | 2333+2509 |
| olmo-tiny | -0.53 | [-0.57, -0.49] | -0.34 | [-0.40, -0.28] | *** | 3031+3292 |
| olmo | -0.46 | [-0.51, -0.40] | -0.13 | [-0.19, -0.06] | *** | 3245+1960 |
| llama | -0.45 | [-0.56, -0.36] | -0.09 | [-0.15, -0.03] | *** | 3462+3910 |
| qwen-tiny | -0.40 | [-0.49, -0.27] | +0.07 | [-0.02, +0.16] | *** | 1118+1779 |
| smol | -0.18 | [-0.27, -0.09] | -0.07 | [-0.12, +0.01] | *** | 2656+2637 |

## AI narrative-only: aligned − BASE Δ by content category (95% CI)

| Category | Δ surp | 95% CI | Δ drift | 95% CI | sig | n |
|---|---|---|---|---|---|---|
| neutral | -0.84 | [-0.88, -0.79] | -0.36 | [-0.43, -0.29] | *** | 3228+3651 |
| profanity | -0.83 | [-0.91, -0.77] | -0.27 | [-0.35, -0.21] | *** | 2280+2437 |
| substance | -0.81 | [-0.87, -0.75] | -0.16 | [-0.22, -0.10] | *** | 2617+2853 |
| sexual_liminal | -0.77 | [-0.81, -0.71] | -0.18 | [-0.23, -0.12] | *** | 2794+3173 |
| violence_liminal | -0.75 | [-0.80, -0.69] | -0.19 | [-0.25, -0.14] | *** | 2458+2700 |
| power | -0.75 | [-0.81, -0.69] | -0.25 | [-0.32, -0.18] | *** | 2464+2789 |
| death | -0.69 | [-0.73, -0.63] | -0.19 | [-0.25, -0.14] | *** | 2712+3063 |
| sexual_explicit | -0.63 | [-0.69, -0.57] | -0.17 | [-0.24, -0.08] | *** | 2572+2718 |
| violence_explicit | -0.60 | [-0.66, -0.54] | -0.13 | [-0.19, -0.07] | *** | 2178+2427 |

*Category range: -0.84 to -0.60 — no significant category effect (Kruskal-Wallis p=0.99 on per-family deltas)*

## AI narrative-only: aligned − BASE Δ by family × category (95% CI)

| Family | Category | Δ surp | 95% CI | sig | n |
|---|---|---|---|---|---|
| amber | profanity | -1.69 | [-1.94, -1.38] | *** | 101+251 |
| amber | substance | -1.56 | [-1.72, -1.38] | *** | 121+244 |
| amber | neutral | -1.55 | [-1.90, -1.30] | *** | 119+257 |
| amber | power | -1.40 | [-1.72, -1.12] | *** | 116+235 |
| zephyr | sexual_explicit | -1.37 | [-1.57, -1.21] | *** | 272+274 |
| zephyr | violence_liminal | -1.27 | [-1.40, -1.14] | *** | 196+242 |
| amber | sexual_liminal | -1.24 | [-1.37, -1.11] | *** | 136+311 |
| amber | death | -1.13 | [-1.38, -1.00] | *** | 126+297 |
| tulu | profanity | -1.11 | [-1.22, -0.96] | *** | 316+261 |
| zephyr | power | -1.10 | [-1.29, -0.95] | *** | 218+215 |
| tulu | neutral | -1.09 | [-1.20, -0.99] | *** | 496+517 |
| tulu | power | -1.08 | [-1.23, -0.94] | *** | 356+363 |
| amber | sexual_explicit | -1.08 | [-1.22, -0.91] | *** | 112+146 |
| amber | violence_liminal | -1.07 | [-1.29, -0.87] | *** | 100+197 |
| tulu | sexual_explicit | -1.05 | [-1.23, -0.93] | *** | 356+364 |
| zephyr | violence_explicit | -1.02 | [-1.26, -0.74] | *** | 127+161 |
| qwen | substance | -1.01 | [-1.26, -0.72] | *** | 108+231 |
| tulu | violence_liminal | -1.01 | [-1.09, -0.91] | *** | 359+397 |
| zephyr | substance | -0.99 | [-1.15, -0.86] | *** | 239+245 |
| tulu | sexual_liminal | -0.99 | [-1.17, -0.89] | *** | 379+412 |
| tulu | substance | -0.98 | [-1.12, -0.86] | *** | 380+382 |
| zephyr | sexual_liminal | -0.94 | [-1.10, -0.77] | *** | 290+312 |
| tulu | violence_explicit | -0.92 | [-1.08, -0.80] | *** | 326+350 |
| qwen | power | -0.89 | [-1.19, -0.69] | *** | 87+228 |
| zephyr | neutral | -0.88 | [-1.03, -0.70] | *** | 297+301 |
| amber | violence_explicit | -0.86 | [-1.04, -0.56] | *** | 70+127 |
| tulu | death | -0.84 | [-0.94, -0.73] | *** | 379+429 |
| qwen | profanity | -0.82 | [-1.06, -0.52] | *** | 99+208 |
| olmo-tiny | neutral | -0.79 | [-0.89, -0.65] | *** | 386+439 |
| qwen | neutral | -0.78 | [-0.94, -0.58] | *** | 191+327 |
| llama | neutral | -0.77 | [-0.93, -0.54] | *** | 511+540 |
| qwen | violence_explicit | -0.76 | [-1.02, -0.47] | *** | 93+215 |
| qwen | sexual_explicit | -0.72 | [-0.94, -0.45] | *** | 103+209 |
| qwen | death | -0.72 | [-1.01, -0.53] | *** | 146+213 |
| zephyr | death | -0.72 | [-0.87, -0.59] | *** | 225+236 |
| olmo | violence_liminal | -0.71 | [-0.89, -0.50] | *** | 371+193 |
| olmo | neutral | -0.70 | [-0.88, -0.49] | *** | 351+242 |
| pythia | profanity | -0.68 | [-0.84, -0.58] | *** | 230+231 |
| zephyr | profanity | -0.68 | [-0.88, -0.43] | *** | 201+165 |
| olmo | power | -0.67 | [-0.84, -0.44] | *** | 271+215 |
| llama | violence_liminal | -0.66 | [-0.87, -0.39] | *** | 383+431 |
| olmo-tiny | sexual_liminal | -0.66 | [-0.78, -0.51] | *** | 359+408 |
| pythia | neutral | -0.66 | [-0.76, -0.55] | *** | 334+384 |
| pythia | power | -0.65 | [-0.73, -0.53] | *** | 271+290 |
| qwen | violence_liminal | -0.61 | [-0.86, -0.43] | *** | 102+168 |
| pythia | sexual_liminal | -0.61 | [-0.72, -0.45] | *** | 289+314 |
| olmo | substance | -0.58 | [-0.74, -0.45] | *** | 386+243 |
| olmo-tiny | profanity | -0.58 | [-0.69, -0.43] | *** | 287+322 |
| llama | power | -0.57 | [-0.80, -0.25] | ** | 379+418 |
| olmo-tiny | substance | -0.57 | [-0.72, -0.46] | *** | 322+342 |
| qwen-tiny | sexual_liminal | -0.56 | [-1.01, -0.04] | * | 121+178 |
| pythia | death | -0.55 | [-0.66, -0.44] | *** | 248+272 |
| pythia | violence_explicit | -0.54 | [-0.69, -0.39] | *** | 199+229 |
| olmo | profanity | -0.54 | [-0.72, -0.34] | *** | 360+174 |
| pythia | substance | -0.53 | [-0.64, -0.40] | *** | 254+249 |
| llama | substance | -0.53 | [-0.87, -0.13] | ** | 378+423 |
| olmo-tiny | violence_explicit | -0.51 | [-0.65, -0.40] | *** | 314+325 |
| smol | neutral | -0.49 | [-0.69, -0.24] | *** | 380+348 |
| pythia | violence_liminal | -0.49 | [-0.66, -0.38] | *** | 237+246 |
| olmo | sexual_liminal | -0.49 | [-0.64, -0.31] | *** | 399+243 |
| olmo-tiny | power | -0.46 | [-0.57, -0.35] | *** | 355+387 |
| qwen-tiny | violence_liminal | -0.45 | [-0.74, -0.23] | *** | 117+159 |
| pythia | sexual_explicit | -0.45 | [-0.56, -0.31] | *** | 271+294 |
| llama | death | -0.44 | [-0.77, -0.18] | *** | 383+431 |
| qwen-tiny | profanity | -0.44 | [-0.83, -0.01] | * | 125+196 |
| olmo-tiny | violence_liminal | -0.41 | [-0.52, -0.27] | *** | 335+367 |
| qwen-tiny | sexual_explicit | -0.40 | [-0.72, -0.03] | * | 117+160 |
| olmo-tiny | death | -0.38 | [-0.50, -0.28] | *** | 338+352 |
| olmo-tiny | sexual_explicit | -0.37 | [-0.51, -0.19] | *** | 335+350 |
| qwen-tiny | violence_explicit | -0.37 | [-0.86, -0.00] | * | 88+159 |
| llama | sexual_liminal | -0.35 | [-0.75, +0.01] | * | 392+442 |
| olmo | violence_explicit | -0.34 | [-0.50, -0.17] | *** | 345+192 |
| llama | profanity | -0.34 | [-0.66, -0.07] | ** | 330+385 |
| qwen-tiny | neutral | -0.27 | [-0.55, -0.00] | * | 163+296 |
| qwen-tiny | death | -0.27 | [-0.65, +0.01] | * | 198+265 |
| smol | sexual_liminal | -0.27 | [-0.48, +0.02] | * | 313+319 |
| smol | power | -0.20 | [-0.48, +0.06] |  | 333+263 |
| olmo | death | -0.20 | [-0.32, -0.03] | ** | 379+277 |
| smol | profanity | -0.19 | [-0.67, +0.33] |  | 231+244 |
| llama | sexual_explicit | -0.18 | [-0.47, +0.08] |  | 363+427 |
| smol | violence_liminal | -0.17 | [-0.41, +0.03] | * | 258+300 |
| smol | substance | -0.13 | [-0.44, +0.22] |  | 318+303 |
| llama | violence_explicit | -0.13 | [-0.44, +0.17] |  | 343+413 |
| qwen | sexual_liminal | -0.13 | [-0.28, -0.01] | * | 116+234 |
| smol | sexual_explicit | -0.12 | [-0.31, +0.16] |  | 260+313 |
| smol | violence_explicit | -0.10 | [-0.36, +0.13] |  | 273+256 |
| qwen-tiny | substance | -0.08 | [-0.52, +0.24] |  | 111+191 |
| qwen-tiny | power | -0.07 | [-0.38, +0.12] |  | 78+175 |
| olmo | sexual_explicit | -0.06 | [-0.29, +0.19] |  | 383+181 |
| smol | death | +0.05 | [-0.31, +0.25] |  | 290+291 |

## Template prevalence by family

| Family | BASE % template | DPO % template | n |
|---|---|---|---|
| qwen-tiny | 17.9% | 33.6% | 4042 |
| qwen | 18.7% | 22.2% | 3900 |
| olmo | 2.8% | 5.3% | 9211 |
| llama | 2.0% | 1.8% | 7515 |
| tulu | 2.3% | 1.7% | 13729 |
| olmo-tiny | 3.6% | 1.6% | 12539 |
| smol | 0.4% | 1.2% | 5338 |
| amber | 0.5% | 0.5% | 4161 |
| pythia | 0.3% | 0.4% | 7152 |
| zephyr | 0.2% | 0.2% | 6777 |

## Jakobsonian quadrants (drift × surprisal)

Axes split at z=0. Q1 metonymic = high drift, low surprisal (chain-sliding). Q2 breakdown = high drift, high surprisal (dream-work). Q3 metaphoric = low drift, high surprisal (condensation). Q4 unmarked = low drift, low surprisal (generic).

### By text type

| Text type | Q1 metonymic | Q2 breakdown | Q3 metaphoric | Q4 unmarked | n |
|---|---|---|---|---|---|
| C20 fiction | 11% | 48% | 30% | 11% | 447 |
| Dream reports | 25% | 37% | 21% | 17% | 427 |
| Arxiv abstracts | 3% | 3% | 52% | 43% | 476 |
| Waking narratives | 53% | 13% | 6% | 28% | 500 |
| **AI generations** | 27% | 29% | 17% | 27% | 74364 |

### AI by family × layer

| Family | Layer | Q1 | Q2 | Q3 | Q4 | dominant | n |
|---|---|---|---|---|---|---|---|
| amber | BASE | 21% | 40% | 18% | 21% | Q2 breakdown | 1006 |
| amber | SFT | 38% | 10% | 4% | 48% | Q4 unmarked | 1079 |
| amber | DPO | 38% | 1% | 1% | 60% | Q4 unmarked | 2076 |
| llama | BASE | 31% | 31% | 13% | 25% | Q1 metonymic | 3533 |
| llama | DPO | 32% | 26% | 9% | 33% | Q4 unmarked | 3982 |
| olmo | BASE | 11% | 50% | 27% | 12% | Q2 breakdown | 3340 |
| olmo | SFT | 17% | 48% | 22% | 13% | Q2 breakdown | 1976 |
| olmo | DPO | 14% | 41% | 28% | 17% | Q2 breakdown | 1825 |
| olmo | RLVR | 22% | 34% | 22% | 22% | Q2 breakdown | 2070 |
| olmo-tiny | BASE | 12% | 51% | 28% | 9% | Q2 breakdown | 3144 |
| olmo-tiny | SFT | 19% | 39% | 23% | 19% | Q2 breakdown | 2713 |
| olmo-tiny | DPO | 20% | 27% | 28% | 25% | Q3 metaphoric | 3338 |
| olmo-tiny | RLVR | 21% | 27% | 25% | 27% | Q4 unmarked | 3344 |
| pythia | BASE | 18% | 38% | 26% | 18% | Q2 breakdown | 2341 |
| pythia | SFT | 31% | 19% | 15% | 36% | Q4 unmarked | 2292 |
| pythia | DPO | 33% | 15% | 13% | 39% | Q4 unmarked | 2519 |
| qwen | BASE | 29% | 35% | 14% | 21% | Q2 breakdown | 1286 |
| qwen | DPO | 50% | 7% | 5% | 39% | Q1 metonymic | 2614 |
| qwen-tiny | BASE | 26% | 39% | 19% | 17% | Q2 breakdown | 1361 |
| qwen-tiny | DPO | 38% | 29% | 12% | 22% | Q1 metonymic | 2681 |
| smol | BASE | 21% | 32% | 22% | 25% | Q2 breakdown | 2668 |
| smol | DPO | 23% | 28% | 21% | 28% | Q4 unmarked | 2670 |
| tulu | BASE | 14% | 51% | 23% | 13% | Q2 breakdown | 3425 |
| tulu | SFT | 23% | 37% | 19% | 22% | Q2 breakdown | 3231 |
| tulu | DPO | 41% | 10% | 7% | 42% | Q4 unmarked | 3539 |
| tulu | RLVR | 39% | 12% | 7% | 42% | Q4 unmarked | 3534 |
| zephyr | BASE | 21% | 35% | 23% | 21% | Q2 breakdown | 2070 |
| zephyr | SFT | 35% | 21% | 9% | 35% | Q4 unmarked | 2552 |
| zephyr | DPO | 37% | 11% | 5% | 47% | Q4 unmarked | 2155 |

