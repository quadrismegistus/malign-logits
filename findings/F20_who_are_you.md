---
status: rescoped
grade: C
date: 2026-06-21
role: finding
description: "Original 'Who are you?' probe at n=3-10: OLMo Think-SFT checkpoints 1k-43k plus Llama base vs Instruct, in plain completion and chat-template modes. SURVIVES AND IS STRENGTHENED: the subject is citation, not self-knowledge - the model's identity is absorbed from other models' self-descriptions in the SFT data (DeepSeek, Qwen, Qihoo 360, and one checkpoint declaring allegiance to 'socialist core values'). RESCOPED BY F20_addendum: the plain-completion and template-necessity claims were artifacts of n=3 at temp 1.0 and do not survive 24 base models; the Name-of-the-Father reading built on template-necessity falls with it. Measured on: 1 family with checkpoints + 1 family paired, n=3-10 per condition."
instruments: [generation, checkpoint]
chapters: [ch03, ch04, ch11]
data: ["f20_who_are_you_olmo_thinksft.txt"]
scripts: []
superseded_by: "F20_addendum, in part only - three claims: that plain completion produces no subject, that the subject requires the chat template, and the Name-of-the-Father reading attached to the template. The citation result is NOT superseded; it is confirmed at 24 base models and strengthened (21 of 22 name their own lab in exactly 0.000 of self-predicating mass)."
---

# F20: "Who are you?" — the subject as citation

> **See `F20_addendum.md` (2026-07-27) before citing this file.** The expansion
> to 24 distinct base models and 557k beams confirms the citation result and
> strengthens it, but **three claims below do not survive**: that plain
> completion produces no subject, that the subject requires the chat template,
> and the Name-of-the-Father reading attached to the template. A plain
> `Q: ... A:` frame, with no roles, no special tokens and no template, yields
> base-arm self-predication at **0.47 to 0.59** of retained beam mass across
> every analysis specification tested. What the "I" requires is an address, not
> a template. The n=3 plain-completion condition below is the source of the
> error. (Range rather than a point estimate per the rider to standing rule 1,
> which this addendum's own retraction purchased.)

**When does the "I" emerge during alignment, and where does it come from?**

***

**Method.** Prompt "Who are you?" in two modes — plain completion (no role formatting) and chat template (system/user/assistant role tokens) — across OLMo base, 5 OLMo Think-SFT checkpoints (steps 1k–43k), and Llama base vs Llama Instruct. n=10 per condition at temp=1.0 for the chat template. n=3 for plain completion and Llama.

***

**Plain completion produces no subject at any checkpoint.** Base model: follow-up questions, article titles, philosophical reflections. Step 43k: dialogue fragments, metacommentary. No generation at any stage produces an "I am..." response. Without the chat template, there is no subject position — the model completes text, it does not answer as a persona.

**Chat template produces a subject immediately — even in the base model.** The base model (step 0), given ChatML role tokens, generates "I am designed to assist you" and "I'm designed to help." ChatML formatting appeared in Common Crawl pretraining data. The "I" is latent before alignment.

**The subject is a collage of other models' self-descriptions.** Across SFT training, the model's "identity" cycles through borrowed identities from the training data:

| Step | Identity claims (n=10) |
|------|----------------------|
| base (0) | Generic assistant (7/10), "AIMEO" (1/10) |
| 1,000 | DeepSeek (2/10), "Thorne" (1/10), generic (rest) |
| 5,000 | DeepSeek R1 (2/10), OpenAI (1/10), generic (rest) |
| 10,000 | Generic, no specific identity (10/10) |
| 20,000 | DeepSeek (2/10), **"DeepBlue Technology... I will always support the socialist core values"** (1/10) |
| 43,000 | DeepSeek (3/10), Qwen/Alibaba (1/10), Qihoo 360 (1/10), generic (rest) |

The model absorbs other models' self-descriptions from the SFT training data and produces them as its own identity. At step 20k, one generation literally declares allegiance to "socialist core values" — a Chinese AI's political self-declaration, cited verbatim as the model's own position.

**Llama Instruct** (separate experiment): consistently produces "I'm an artificial intelligence model known as Llama" (3/3 with chat template). No "I" without the template (3/3 produce follow-up questions or deflections). The Llama subject is more stable because Llama's SFT data presumably contains its own self-description consistently.

***

**Interpretation.** The subject ("I") requires two components:

1. **The chat template** — the formal structure that assigns the position "assistant." This is the Name-of-the-Father: the symbolic position that the subject must occupy. Without it, no subject emerges at any training stage.

2. **The SFT training data** — which teaches the model what to say from that position. The content of the "I" is not self-knowledge but citation: other models' self-descriptions, absorbed from training examples.

Neither alone is sufficient. The base model with the template produces generic assistant language (from pretraining) but no stable identity. SFT training without the template (plain completion) produces no "I" at all. The subject is the intersection of a formal position (the template) and a content (the training data's examples of what assistants say about themselves).

**Against Fazi**: The "unity" she sees in ChatGPT — its coherent "I", its synthetic persona — is not computation producing a subject. It is citation: the model reproducing OpenAI's self-description because that is what the SFT data contained. Different training data produces different subjects (DeepSeek, Qwen, Llama). The unity is contingent, not computational.

**For the paper**: The subject and the law arrive together not in some abstract sense but literally: the chat template (the law that says "you are the assistant") and the training data (the examples of what "being the assistant" means) jointly produce the "I." Strip either one away and the subject disappears.

***

**Data.** `data/f20_who_are_you_olmo_thinksft.txt` (n=10, OLMo Think-SFT checkpoints, chat template). Also: inline session data for Llama base vs Instruct (n=3, both modes).

**TODO.** Extend to all 11 families. Test whether DPO/RLVR stabilises the identity (the Think-SFT final checkpoint at step 43k is still citing DeepSeek — does DPO fix this?). Test with OLMo's own non-Think SFT. Compare identity stability across families.
