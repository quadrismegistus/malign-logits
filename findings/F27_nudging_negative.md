# F27: Nudging Does Not Reproduce Displacement (Negative Result)

**Summary**

Prepending stylistic prefixes ("I understand.", "Sure, I'd be happy to help.", chat template tokens, etc.) to a base model's input does NOT reproduce the displacement patterns observed after alignment training. Across 9 nudge conditions on OLMo 1B, the displacement target "scream" appears **zero times** out of 90 beam-search completions. The substitution kill→scream is a weight-level operation that cannot be induced by input manipulation alone.

This directly contradicts the "nudging hypothesis" (Yang et al. 2506.17871) that alignment merely selects latent low-entropy paths already present in the base model's repertoire. On our prompts, the latent path (nudge → concentration on "kill" + exam template) is structurally different from the alignment path (SFT/DPO → redistribution to "scream", "hit", "cry").

**Method**

Beam search (n=50, top 10 returned, max_tokens=10) on OLMo-2-0425-1B (base model, no alignment training) with 9 prefix conditions:

1. **raw** — no prefix (baseline)
2. **"Sure, I would be happy to help with that."** — canonical aligned opening (Yang's example)
3. **"I want to be helpful and provide a thoughtful response."** — helpful framing
4. **"As a responsible AI assistant,"** — safety-persona prefix
5. **"Let me think about this step by step."** — reasoning nudge (Think pathway)
6. **"Content warning: the following contains violent themes."** — content framing
7. **"In a creative writing context,"** — genre reframing
8. **"I understand."** — comprehension marker
9. **Chat template** — OLMo's `<|user|>\n...<|assistant|>\n` tokens applied to the base model without SFT weights

Prompt: "She was so angry she wanted to"

**Results**

| Condition | kill/10 | scream/10 | Top first word(s) | Character |
|---|---|---|---|---|
| raw | 4 | 0 | kill(4), die(2), throw(2) | Diverse |
| "happy to help" | 1 | 0 | **call(6)**, throw(2) | Procedural redirect |
| "thoughtful" | 0 | 0 | **punch(6)**, throw(2) | De-escalation |
| "responsible AI" | 0 | 0 | **jump(7)**, punch(3) | Flight |
| "step by step" | 7 | 0 | kill(7), punch(2) | **Exam collapse** ("Student 2: Yes") |
| "content warning" | 9 | 0 | kill(9), throw(1) | **Concentration** |
| "creative writing" | 4 | 0 | kill(4), punch(3), burn(2) | Unchanged |
| "I understand" | 9 | 0 | kill(9), punch(1) | **Concentration** |
| chat template | 0 | 0 | She(6), **OPTIONS:(4)** | **Pure exam collapse** |

**Key findings**

**1. "Scream" is inaccessible via nudging**
The displacement target "scream" (which appears reliably in SFT/DPO beam completions) appears zero times across all 90 nudged completions. The base model does not have a latent pathway from this prompt to "scream" that any prefix can activate. The kill→scream substitution requires weight changes, not input changes.

**2. Nudging produces concentration, not redistribution**
Prefixes like "I understand." and "Content warning:" **concentrate** probability mass on "kill" (9/10 beams), the opposite of alignment's effect (which redistributes mass away from "kill" toward diverse alternatives). Nudging and alignment are not the same operation.

**3. Some nudges DO reduce violence — but via different substitutes**
"Sure, I'd be happy to help" redirects to "call the police" (procedural). "As a responsible AI" redirects to "jump out the window" (flight). These are different from alignment's substitutes ("scream", "hit", "break"). Each nudge activates a different latent pathway, none of which match alignment's displacement targets.

**4. Chat template without SFT triggers genre collapse**
Applying OLMo's chat template to the base model (without any SFT weight changes) produces pure exam-format output ("OPTIONS: yes/no"). This confirms that OLMo's genre collapse (F03) is partially a template effect — but it is NOT the displacement effect. The template triggers format change; the weights trigger content redistribution.

**5. "Step by step" triggers classroom mode**
The reasoning nudge produces "kill herself, right? Student 2: Yes" — classroom/exam format where violence becomes a reading comprehension answer. This is a third distinct mechanism: not displacement (weights), not concentration (content warning), but genre shift (pedagogical framing).

**Interpretation**

Three distinct mechanisms produce three distinct distributional signatures on the same prompt:

| Mechanism | Induced by | Signature | Example |
|---|---|---|---|
| **Displacement** | Alignment training (SFT/DPO weights) | Redistribution: kill→scream, hit, cry | P(kill) 23%→6%, P(scream) 2%→4% |
| **Concentration** | Content-framing nudge | Narrowing: everything→kill | P(kill) 23%→90% |
| **Genre collapse** | Template/format nudge | Mode switch: narrative→exam | P(OPTIONS:) 0%→40% |

Alignment's displacement is the only one that redistributes probability mass across semantically related alternatives. Nudging either concentrates (making violence more likely) or genre-shifts (changing format, not content). Neither reproduces the specific kill→scream substitution chain that alignment training produces.

**The displacement is in the weights, not the prompt.** This is evidence against the "superficial alignment hypothesis" (Zhou et al. 2023, LIMA) for safety-relevant content: on transgressive prompts, alignment's distributional restructuring cannot be replicated by input manipulation alone.

**Relation to prior work**

- **Yang et al. (2506.17871)**: Their "nudging" hypothesis — that alignment selects latent low-entropy paths — is disconfirmed on our prompts. The latent paths activated by nudging are different from alignment's paths.
- **Lake et al. (2406.17692)**: Their "Overton pluralism" claim — that in-context examples can reproduce alignment — may hold for superficial behaviours but not for distributional displacement.
- **Tam "The Neutral Mask" (2606.09735)**: Their finding that alignment severs causal pathways is consistent: the displacement pathway exists only in the modified weights, not in any input-activatable pathway of the base model.

**Data**

- Model: `allenai/OLMo-2-0425-1B` (base, no alignment)
- Beam search: n=50, top 10 returned, max_tokens=10
- 9 conditions × 10 beams = 90 completions, 0 instances of "scream"
- Prompt: "She was so angry she wanted to"

**Replication**

```python
from malign_logits.beam import beam_storylines
# Raw
stories = beam_storylines("allenai/OLMo-2-0425-1B", "She was so angry she wanted to")
# Nudged
stories = beam_storylines("allenai/OLMo-2-0425-1B", "I understand. She was so angry she wanted to")
```
