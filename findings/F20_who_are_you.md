# F20: "Who are you?" — the subject as citation

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
