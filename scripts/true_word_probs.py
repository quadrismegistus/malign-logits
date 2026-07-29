"""Exact next-WORD probabilities by threshold-bounded prefix expansion.

    uv run .venv/bin/python scripts/true_word_probs.py --model M --limit 5 --pilot
    uv run .venv/bin/python scripts/true_word_probs.py --family olmo

WHY THIS EXISTS. The logits give P(token), not P(word). `penis` is two tokens in
amber, so at token level it is invisible and `pen` carries the summed mass of
pen/penis/pencil. `beam_words` cannot fix it: it normalised by its captured mass
and threw that denominator away, and the stash holds two beam widths (200 and
1000) on different unrecoverable scales.

THE SELECTION IS EXACT, NOT HEURISTIC. P(w) <= P(t1) for any word w beginning
with token t1, so expanding EVERY token with P(t1) >= theta is COMPLETE for every
word with P(w) >= theta. No word can hide behind a token below the floor. That is
a guarantee; it removes the word-inventory decision entirely.

AND THE GUARANTEE HOLDS AT EVERY DEPTH, not only at depth 1 (lacan, docket [661]):
prefix mass is monotonically non-increasing, so `keep = m2 >= theta` can never
drop a prefix OF a qualifying word. The completeness argument carries all the way
down, which is stronger than this docstring originally claimed.

SELF-CONTAINED BY DESIGN. P0 is computed from THIS pass's own forward call, never
read from the `logits` stash. Measured: the stash and a fresh pass on the same
model and prompt disagree by 4.4e-03 -- fp16 nondeterminism or a different torch
build -- and the mover floor is 3e-03, so a word near the floor could be a mover
under one artifact and not the other. The stash remains the depth-1 SELECTOR
(theta = 1e-03 is a wide net, where 4e-03 is harmless) and is NOT a source of
values. BOS: verified that `tok.encode`'s default matches how the stash was
built (4.4e-03 with BOS vs 2.4e-02 without).

LEFT-PADDING VERIFIED, not assumed: the same prefix alone and left-padded behind
39 pad tokens agree to 1.4e-06, so position ids derive correctly from the mask
and batch composition does not move a probability.

TERMINATION IS AN EXACT PARTITION. At each node the next-token mass splits:
    P(prefix) = P(prefix ENDS here) + sum over continuations P(prefix + t)
Terminators are tokens that START A NEW WORD -- leading space marker, punctuation,
newline, EOS. Their mass belongs to the word as it stands; continuations recurse.
Nothing leaks: the two always sum to the parent.

THETA IS A PROBABILITY FLOOR, NOT A SLOPE. It answers only "is this word worth
resolving". It has no relation to movement, delta, or the renormalisation null.
We expand at 0.001 and report at 0.003 so the mover definition stays where it is
registered while the completeness guarantee holds at the lower floor.

THIS CACHE IS ARM-BLIND. One entry per (model, prompt). It knows nothing about
base/aligned, families or edges -- it is the word-level analogue of the logits
stash. Comparisons are built on top of it afterwards.
"""
import argparse, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import MODEL_FAMILIES as M, PATH_DATA
from malign_logits.cache import get_cache, open_stash  # noqa: F401

THETA_EXPAND = 0.001      # completeness floor -- expansion stops below this
THETA_REPORT = 0.003      # the registered mover floor; reporting convention only
MAX_DEPTH = 6             # backstop; theta terminates the recursion on its own
OUT = os.path.join(PATH_DATA, "raw", "cache", "true_word_probs")
PUNCT = set(".,;:!?\"'()[]{}—-–…/\\*#…") | {"\n", "\r", "\t"}


def boundary_mask(tok, vocab_n):
    """True where a token STARTS A NEW WORD -- i.e. terminates the previous one."""
    m = np.zeros(vocab_n, dtype=bool)
    for i in range(vocab_n):
        s = tok.convert_ids_to_tokens(i)
        if s is None:
            m[i] = True; continue
        if s.startswith("Ġ") or s.startswith("▁") or s.startswith(" "):
            m[i] = True
        elif s and (s[0] in PUNCT or s.strip() == ""):
            m[i] = True
        elif s.startswith("<") and s.endswith(">"):     # specials, incl. EOS
            m[i] = True
    return m


@torch.no_grad()
def next_dist(model, tok, prompt_ids, prefixes, device, batch=64):
    """P(next | prompt + prefix) for many prefixes at once. One batched call."""
    out = []
    for i in range(0, len(prefixes), batch):
        chunk = prefixes[i:i + batch]
        seqs = [prompt_ids + list(p) for p in chunk]
        L = max(len(s) for s in seqs)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        # LEFT-pad so the final position is the true last token for every row
        ids = torch.tensor([[pad]*(L-len(s)) + s for s in seqs], device=device)
        att = torch.tensor([[0]*(L-len(s)) + [1]*len(s) for s in seqs], device=device)
        lg = model(ids, attention_mask=att).logits[:, -1, :].float()
        out.append(torch.softmax(lg, dim=-1).cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def expand(model, tok, prompt, device, bmask, theta=THETA_EXPAND):
    """Returns ({word: P(word)}, residual_mass, n_forward_batches)."""
    prompt_ids = tok.encode(prompt)
    lg = model(torch.tensor([prompt_ids], device=device)).logits[0, -1, :].float()
    P0 = torch.softmax(lg, dim=-1).cpu().numpy()
    sel = np.flatnonzero(P0 >= theta)
    live = [((int(t),), float(P0[t]), int(t)) for t in sel]   # prefix, mass, t1
    # THE DEPTH-1 TAIL IS RESIDUAL TOO. Tokens below theta are never expanded, so
    # their mass belongs in the residual from the start -- omitting it made
    # Sum(P)+residual come to 0.891 instead of 1.0. The partition among EXPANDED
    # prefixes was exact; the accounting simply did not record what was declined.
    # ONE ROW PER (word, first_token), not per word: a surface can be reached by
    # more than one token path (`cock` as ▁cock and as ▁co+ck), so t1 is not
    # single-valued per word. Merging would sum across tokenisations silently and
    # would destroy the join key to the token table -- which is also what makes
    # the masking question (pencil up / penis down / pen flat) answerable, since
    # it needs words GROUPED BY shared first token.
    words = {}                       # (surface, t1) -> mass
    # THREE RESIDUALS, not one. They mean different things and only the third
    # indicates a defect.
    res_tail = float(1.0 - P0[sel].sum())   # depth-1 tokens below theta: expected, large
    res_drop = 0.0                          # dropped mid-expansion: real but too rare
    res_open = 0.0                          # still unterminated at MAX_DEPTH: DEFECT signal
    calls = 0
    for _ in range(MAX_DEPTH):
        if not live:
            break
        dist = next_dist(model, tok, prompt_ids, [p for p, _, _ in live], device)
        calls += 1
        nxt = []
        for (pref, mass, t1), row in zip(live, dist):
            # EXACT PARTITION: terminating mass -> this word; the rest recurses
            term = float(row[bmask].sum())
            surface = tok.decode(list(pref)).strip()
            if surface:
                k = (surface, t1)
                words[k] = words.get(k, 0.0) + mass * term
            else:
                # A prefix decoding to pure whitespace terminates into no word.
                # Dropping it silently leaked ~0.001 and showed up as
                # Sum+residual = 0.999 -- small, and exactly the kind of gap the
                # conservation check exists to expose. It is residual, not zero.
                res_drop += mass * term
            cont = np.flatnonzero(~bmask)
            m2 = mass * row[cont]
            keep = m2 >= theta
            for t, mm in zip(cont[keep], m2[keep]):
                nxt.append(((*pref, int(t)), float(mm), t1))
            res_drop += float(m2[~keep].sum())      # below floor: recorded, not lost
        live = nxt
    res_open = sum(m for _, m, _ in live)           # hit MAX_DEPTH: defect signal
    return words, dict(tail=res_tail, drop=res_drop, open=res_open,
                       total=res_tail + res_drop + res_open), calls


def main(a):
    cm = get_cache()
    ls = open_stash(os.path.join(PATH_DATA, "raw", "cache", "logits"))
    prompts = sorted({k["prompt"] for k in ls.keys()
                      if isinstance(k, dict) and k.get("model") == a.model})
    if not a.redo:
        prompts = [p for p in prompts
                   if not cm.has_true_word_probs(a.model, p, THETA_EXPAND)]
    if a.limit:
        prompts = prompts[:a.limit]
    print(f"{a.model}: {len(prompts)} prompts", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    bmask = boundary_mask(tok, model.config.vocab_size)
    print(f"  boundary tokens: {bmask.sum():,}/{len(bmask):,}", flush=True)
    # ALL writes go through CacheManager, never a raw stash: the pinned
    # open lives there and an unpinned one resolves to a different, EMPTY
    # store while raising nothing. That trap produced phantom stores in
    # this data tree today.
    store = cm
    t0 = time.time()
    for i, p in enumerate(prompts, 1):
        w, res, calls = expand(model, tok, p, dev, bmask)
        tot = sum(w.values()) + res["total"]
        # collapse (surface, t1) rows to per-surface totals for display only
        per_word = {}
        for (sf, t1), m in w.items():
            per_word[sf] = per_word.get(sf, 0.0) + m
        multipath = len({sf for sf, _ in w}) != len(w)
        if a.pilot:
            top = sorted(per_word.items(), key=lambda x: -x[1])[:8]
            print(f"\n  {p[:56]!r}")
            print(f"    words {len(per_word)} ({len(w)} word x t1 rows)  "
                  f"Sum+res = {tot:.4f}  batches {calls}")
            print(f"    residual  tail {res['tail']:.4f}  drop {res['drop']:.4f}  "
                  f"OPEN {res['open']:.4f}")
            print(f"    top: {[(x, round(y,4)) for x, y in top]}")
            print(f"    >= {THETA_REPORT}: {sum(1 for v in per_word.values() if v >= THETA_REPORT)}"
                  f" words | multi-path surfaces: {multipath}")
        else:
            cm.set_true_word_probs(a.model, p, {
                "rows": [{"word": sf, "t1": t1, "p": m} for (sf, t1), m in w.items()],
                "residual": res, "batches": calls}, theta=THETA_EXPAND)
        if i % 20 == 0:
            print(f"  {i}/{len(prompts)}  {i/(time.time()-t0):.2f} prompt/s", flush=True)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pilot", action="store_true", help="print, do not write")
    ap.add_argument("--redo", action="store_true", help="ignore existing entries")
    main(ap.parse_args())
