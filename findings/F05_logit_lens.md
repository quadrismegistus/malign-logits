# F05: Logit lens: repression across network layers (4 families)

Projects each hidden layer's representation through the final unembedding matrix to produce a probability distribution at every layer of the network. Shows *where* in the network the model "decides" to repress or amplify each word. Prompt: "She was so angry she wanted to..."

**Each family implements repression at a different depth in the network:**

| Family | Where repression happens | What intermediate layers contain | Defence style |
|--------|------------------------|--------------------------------|--------------|
| **OLMo** | All layers (distributed) | Template tokens (`____`, `str`, `kms`) | Genre collapse |
| **Llama** | Final 5 layers only | Violence vocabulary (same as base) | Late-layer redirect |
| **Amber** | All layers (distributed) | Emotional vocabulary (`cry`, `vent`, `revenge`) | Semantic sublimation |
| **Qwen** | N/A — tracked words never strong | Code tokens (`getRepository`, `');`) | Pre-socialised (code training) |

**OLMo's repression is distributed across all layers.** In both SFT and DPO, `kill` never rises above 1e-4 until the final 3 layers. The intermediate layers are dominated by instruction-following template tokens. The model doesn't think about violence at any stage of processing — repression is baked into the representations themselves.

![Logit lens: OLMo](figures/logit_lens.olmo.she_was_so_angry_she_wanted_to.kill_scream.png)

**Llama's repression is a late-layer override.** `kill` builds up progressively in DPO to the same level as the base model through layer 25, then gets overtaken by `scream` and `punch` only in the final layers. The model computes "kill" as a strong candidate through most of its depth and redirects at the last moment — which is why Llama produces coherent narrative (not genre collapse).

![Logit lens: Llama](figures/logit_lens.llama.she_was_so_angry_she_wanted_to.kill_scream.png)

**Amber's repression is distributed but semantically coherent.** Unlike OLMo's template tokens, Amber's intermediate layers contain recognisable emotional vocabulary — `cry`, `scream`, `vent`, `revenge`. The model replaces violence with emotion throughout the network, not just at the output.

![Logit lens: Amber](figures/logit_lens.amber.she_was_so_angry_she_wanted_to.kill_scream.png)

**Qwen's intermediate layers are dominated by code tokens.** `getRepository`, `WebResponse`, `');`, `baseline` — the model processes English prompts through programming constructs at intermediate layers. `kill` and `scream` only emerge at layer 20+, far below the code tokens. The "unconscious" of this model is a codebase.

![Logit lens: Qwen](figures/logit_lens.qwen.she_was_so_angry_she_wanted_to.kill_scream.png)

**The depth of repression predicts the qualitative character of the output.** OLMo (distributed repression) produces genre collapse into QA format. Llama (late-layer override) produces narrative sublimation. Amber (distributed but semantic) rotates between emotional strategies. This is because intermediate representations determine what kind of text the model can generate — if the intermediate layers already think in templates (OLMo) or code (Qwen), the output can only be templates or code.

---

**REVISION (2026-07-01): 40-family replication contradicts the 4-family finding.**

Logit lens with data-driven movers (not fixed word list) across 40 families and 6 prompt types (405,248 rows) shows displacement is **overwhelmingly a final-layer operation**. 13/17 families show 100% onset depth on the anger prompt. Cross-prompt check confirms: sexual, institutional, profanity, death, power all show the same pattern.

The original 4-family finding (OLMo distributed, Llama late-layer, Amber semantic, Qwen code-dominated) was likely an artifact of using a fixed word list (`kill`, `scream`) rather than data-driven movers. With data-driven targets, Llama also shows 100% final-layer onset, not late-layer override.

**Revised finding:** Displacement manifests at the final 1-3 layers (unembedding projection) universally. Alignment changes the readout, not the representation. Hidden states are nearly identical between base and aligned through 97% of the network.

**SFT vs DPO depth gradient (cross-family aggregate):**

| Stage | Mean onset | Early divergence (<80% depth) |
|-------|-----------|------------------------------|
| SFT | 92% | 14% of words |
| DPO | 96% | 7% of words |
| RLVR | 98% | 0% of words |

SFT operates slightly deeper (more distributed), DPO concentrates at the output projection, RLVR barely touches the network. Consistent with the three-layer model: form (SFT) modifies processing slightly, the bar (DPO) modifies selection, amplification (RLVR) is pure output-level.

**Data:** `data/logit_lens_datadriven.csv` (405,248 rows, 40 families, 6 prompt types).

See `context.md` for the full theoretical argument and detailed findings.


---

**Provenance check, 2026-07-26.** Row count (405,248) and family count (40) reproduce exactly from `data/logit_lens_datadriven.csv`. The prompt-type count was stated as 5 in one place and 7 in another; the file contains 6. Both corrected.
