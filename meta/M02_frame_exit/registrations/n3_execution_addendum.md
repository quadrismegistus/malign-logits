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

    dtype    float16, keyed in the store
    device   mps where available, else cpu

`dtype` is a key field, not metadata: a dtype difference is a logit difference and
a next-token probability is this campaign's quantity. A read that names no dtype
is answerable only while one exists for that cell.

## 7. Population precondition

A triplet whose poles differ by more than one token block of one token is not
measuring the construct the frozen body describes, and this is knowable before any
forward pass. Such triplets are refused by the runner and named. The refusal is a
property of the prompts and is evaluated before any model loads.

If the frozen body's own triplet ever fails this check the runner **exits** rather
than proceeding without the confirmatory arm.

## 8. Scope

The frozen body's triplet is **CONFIRMATORY**. Every other triplet, and the pythia
ladder, are **EXPLORATORY EXTENSION**, labelled at write time and not at analysis
time.

## 9. What this addendum may not contain

No observed rate, no marker profile, no arm behaviour, no statement about what any
model does. Its only subject is the execution path. Producer:
`scripts/f11_l1_logits.py` at `999e6f4f`.
