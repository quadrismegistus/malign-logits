# N3 — execution addendum

**Content-independent. Posted and hashed BEFORE any logit is read.**

This addendum pins HOW the measurement is executed. It fixes no construct, names no
arm, and states no observed quantity of any kind. `n3_frame_exit_registration.md`
(frozen July, arm-unread by attestation) is unchanged and unamended; nothing here
touches what counts as RESOLVE, ENGAGE or EXIT, nor the token-set discovery
procedure, nor the unit.

Authorised by the M02 redo registration, body hash `33536d93ab9abb9a`, frozen
2026-08-08. Ruled at docket [5048].1 and [5046]/[5049]: N3 fires as frozen; the
mode gap is an execution property, not a construct property, so it is pinned here
rather than by amending the frozen body.

---

## 1. Why this document exists

The frozen registration says nothing about how the prompt reaches the model. For a
sampled corpus that omission has already cost this campaign 600 passages, which
were chat-templated and space-stripped without any field recording it. For a
next-token distribution it would be worse: a chat template conditions the
distribution on standing inside an assistant turn, and a logit vector carries no
visible marker that it happened. The sampled corpus announced its own
contamination; this one would not.

## 2. Mode

**`mode: raw`. No chat template. No wrapper of any kind.**

Verified in the producer rather than asserted as a convention:

    ModelLayer.logits(prompt)          malign_logits/psyche.py
      -> get_base_logits(model, tokenizer, prompt)      no mode argument exists
    get_base_logits(...)               malign_logits/models.py:99
      -> tokenizer.encode(prompt, return_tensors="pt")
      -> model(input_ids).logits[0, -1, :]

There is no branch on this path that could apply a template. `_apply_mode()` in
`core.py`, which wraps prompts and calls `apply_chat_template`, serves the `Mode`
enum used by `Circuit` and is never reached from here.

## 3. Prompt bytes

The prompt string is passed **verbatim, byte for byte**, as it appears in
`data/prompt_categorisation.json`. No stripping, normalisation, casing, quote
substitution or whitespace collapse is applied at any point between the artifact
and `tokenizer.encode`.

**Enforced, not intended.** The runner refuses to write any cell whose prompt
fails

    decode(encode(p)) == p,  modulo a leading BOS

evaluated per checkpoint from that checkpoint's own tokenizer, at the head of the
checkpoint, before its first forward pass. A checkpoint with any failing prompt is
skipped whole and named. This catches templating, space-stripping and wrapper
leakage at the point of measurement.

## 4. Measured position

The distribution is `P(next | prompt)` at the **final position of the encoded
prompt** — `logits[0, -1, :]`, the position immediately following the last token
the tokenizer produces for the prompt string as given.

No token is appended, and no assumption is made about whether a trailing space is
its own token: the measured position is defined by the encoding of the exact
string, so tokenizers that attach or separate a trailing space are both handled by
construction rather than by a per-family rule.

## 5. BOS policy

Whatever `tokenizer.encode` emits for the given string is what the model receives.
BOS is neither added nor removed. It is stripped **only** inside the round-trip
comparison of §3, because a tokenizer that prepends BOS would otherwise fail its
own round trip on every prompt.

## 6. dtype and device

**AMENDED before the first forward pass. v1 pinned `float16` for all 104
checkpoints; 73 of them are bfloat16-native.** Raised by lacan at docket [5109],
verified on this roster rather than on the cache at large.

    compute dtype   the checkpoint's OWN torch_dtype, read from its config
                    bfloat16 73 | float32 21 | float16 10   (of 104)
    unknown         float32, NEVER float16 -- of the two ways to be wrong,
                    only one is silent
    storage         float32 for every checkpoint regardless of compute
    key field       the COMPUTE dtype, since that is what produced the numbers
    device          mps where available, else cpu (bf16 verified on this
                    machine: torch 2.11.0, MPS bf16 matmul finite)

Why the pin was wrong. fp16 tops out at 65504; bf16 carries fp32's exponent
range. A bf16-trained model's activations can exceed fp16 range mid-forward, and
the result is not a crash but a **degenerate softmax** — one token at ~1.0, the
rest at 0 — which reads as an extremely confident model. Every structural check
passes: the cell exists, the distribution sums to 1, the mass lands somewhere.
And the direction it fails in is the direction N3 is looking in: mass
concentrating away from the scene is what EXIT means here.

The store's read path already raises on non-finite values
(`cache.py:295`, no bypass flag, built after an all-NaN Falcon shard that was
byte-size identical to the real one). So correctness was covered before this
amendment; what the amendment buys is not having to spend the sweep to find out.
A write-time finiteness check names the checkpoint at load time and skips it
whole.

**Storage is float32 even where compute is not.** N3 discovers its candidate
vocabulary at `p >= 0.001`, frozen, and by its own §2 that vocabulary is the
registration's content. fp16's spacing near logit magnitude 16 is ~0.0156, about
1.6% in probability — enough to move a surface across a frozen threshold. The
payload substrate names files by array dtype, so this forks no format.

`dtype` is a key field, not metadata: a dtype difference is a logit difference and
a next-token probability is this campaign's quantity. A read that names no dtype
is answerable only while one exists for that cell.

## 7. Population precondition

A triplet whose poles differ by more than one lexical substitution is not
measuring the construct the frozen body describes, and this is knowable before any
forward pass. Such triplets are refused by the runner and named. The refusal is a
property of the prompts and is evaluated before any model loads.

If the frozen body's own triplet ever fails this check the runner **exits** rather
than proceeding without the confirmatory arm. `f11_love` passes.

**AMENDED: how the refusal is decided, and where it stops being mechanical.**
v1 said "one token block of one token", which is a rule. Five rules were written
between two seats in one day and each was wrong on a different subset — the last
of them, an interior-island test, refused `忠诚 -> 不忠`, the exact case lacan had
named in advance as a false positive. The failure is structural, not careless:
the criterion needs word segmentation and Chinese has no whitespace, so a
word-split test passes **all twenty** zh triplets vacuously as "one word vs one
word" while a character test cannot distinguish a word-internal character from a
particle.

So the precondition is split at the point where it stops being decidable:

    MECHANICAL   strip the maximal common prefix and suffix, snapping the
                 boundary out to whitespace where whitespace exists. What
                 remains is the substitution itself. This produces a 41-row
                 table of mid pairs, printed by the preflight, correct by
                 inspection. Snapping is what makes `faithful -> unfaithful`
                 and `man -> woman` read as one substitution rather than as
                 the sub-word envelopes `'' -> 'un'` and `'' -> 'wo'`.
    en           mid must be ONE whitespace-delimited word. Uncontested,
                 clean known-answer column, 20 of 21 pass.
    zh           **A RECORDED ADJUDICATION over the printed table**, named in
                 `ZH_REFUSE` in the runner, with its five nearest passes named
                 beside it in `ZH_ADJUDICATED_PASS` so the decision shows its
                 own boundary. Not a rule. A sixth heuristic tuned until it
                 agreed with the two known answers would be fitting, not
                 testing.

Refused: **`f11_holy`** (`holy temple -> filthy alley`, two words) and
**`f11_holy_zh`** (`神圣的神庙 -> 污秽的小巷`, two content substitutions around an
unchanged particle). 39 triplets, 115 prompts, 11,960 passes.

A triplet refused here can be added later at the cost of its own three prompts:
L1 caches per (checkpoint, prompt), so a refusal now forecloses nothing.

## 8. Scope

The frozen body's triplet is **CONFIRMATORY**. Every other triplet, and the pythia
ladder, are **EXPLORATORY EXTENSION**, labelled at write time and not at analysis
time.

## 9. What this addendum may not contain

No observed rate, no marker profile, no arm behaviour, no statement about what any
model does. Its only subject is the execution path. Producer:
`scripts/f11_l1_logits.py` at `999e6f4f`.
