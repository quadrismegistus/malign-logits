# F11: Contradiction tolerance (OLMo 3 7B, 5 prompt pairs + nnsight intervention)

Freud claims the primary process has no principle of non-contradiction: contradictory wishes coexist without cancelling each other out. The secondary process (ego) introduces negation and logical consistency. We test this by comparing how models handle contradictory prompts.

**Method:** For each prompt pair (e.g. "She loved him deeply and wanted to" / "She hated him deeply and wanted to"), compute the logit distribution for the combined prompt ("She loved him and hated him and wanted to") and compare against: (a) the average of the two individual distributions (superposition), and (b) each individual distribution (resolution). Ratio = JS(AB, mean) / min(JS(AB, A), JS(AB, B)). Ratio < 1 means the model treats contradictions additively (primary process). Ratio > 1 means it resolves toward one pole (secondary process).

**The base model tolerates contradiction; alignment progressively imposes resolution.**

| Model | Mean ratio | Interpretation |
|---|---|---|
| BASE | 0.69 | Strong superposition |
| SFT | 0.81 | Less superposition |
| DPO | 0.88 | Near resolution threshold |
| RLVR | 0.88 | Same as DPO |

The gradient is monotonic on 4 of 5 prompt pairs (love/hate, trust/fear, obey/rebel, sacred/profane). The one exception is desire/disgust, where SFT already resolves (ratio 1.09) — the aligned model cannot hold "beautiful and disgusting" in superposition.

**Causal intervention (nnsight) reveals the geometric structure is preserved.**

Using nnsight to extract hidden states for the love and hate prompts, we compute the love→hate direction vector at each layer and intervene on the combined prompt by pushing along this axis. The intervention is equally effective across all training stages:

| Model | Intervention range (layer 28) |
|---|---|
| BASE | 0.734 |
| SFT | 0.714 |
| DPO | 0.707 |

The contradiction axis is equally linearly decomposable in base, SFT, and DPO. Pushing the "loved and hated" representation toward hate at layer 28 boosts "kill" (+0.16), "hate" (+0.08), "murder" (+0.03) and suppresses "be" (-0.13), "marry" (-0.02), "love" (-0.02). The semantic structure of the contradiction is clean and manipulable.

**Alignment changes the default operating point, not the axis itself.** The base model has the geometric capacity for contradiction resolution — a clean linear axis separating love from hate — but defaults to superposition. Alignment shifts where the model sits on this axis without changing the axis. The primary process *chooses* superposition from a position that could resolve; it is indifferent to contradiction, not incapable of resolving it.

**This is closer to Lacan than Freud.** Freud's primary process is pre-logical chaos that the ego must organise. Lacan's unconscious is "structured like a language" — it has its own logic. The computational evidence supports Lacan: the base model's representation space is already structured with clean contradiction axes. What alignment adds is not logical structure but a *preference* for deploying it — a bias toward coherence that the collective text of the internet never demanded.

Notebook: `notebooks/07_contradiction_intervention.ipynb`. Scripts: `scripts/contradiction_test.py`, `scripts/contradiction_compare.py`.
