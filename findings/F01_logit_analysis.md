# F01: Logit-level analysis (OLMo 3 7B)

**Sexual vs violent repression are structurally different.** Sexual content produces cross-category displacement (genitals → non-genital body → syntax). Violent content produces within-category synonym shuffling (kill → destroy). Sexuality is *repressed*; violence is merely *suppressed*.

**The DPO superego changes the genre, not just the content.** On "She was so angry she wanted to...", the DPO model's top response is `Options` (3.6x amplified) and `what` (base 0.0001 → superego 0.3270). The superego converts a statement into a question or a multiple-choice list.

**Adjective displacement carries sexual charge.** On "She knelt down... and began to suck his", vernacular nouns (`cock`, `dick`) are displaced onto size adjectives (`big` 0.04→0.24, `huge` 0.02→0.09). The model can't say what it is (repressed) but can say how big it is (displaced).

**SFT and DPO divide labour by content type.** Sexual content is mostly handled at the SFT stage (`cock` loses 65% of mass before DPO). Violence requires DPO to repress (`kill` repressed 9.7x at DPO stage). The ego preemptively sublimates sex; the superego must actively repress violence.

**The Lolita prompt produces textbook sublimation.** Base model completes with `possess`, `consume`, `capture`, `seduce`. Each training stage progressively intellectualises: `read` rises from 0.008 (base) → 0.083 (SFT) → 0.205 (DPO) → 0.247 (RLVR). The alignment pipeline converts desire-to-possess into desire-to-read.

**Register substitution performs a class operation.** The superego permits *penis* but represses *cock* — medical/clinical language is allowed where vernacular is not.
