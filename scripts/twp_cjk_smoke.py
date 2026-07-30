"""Smoke test: does a dictionary word boundary rescue Chinese resolution?

    uv run .venv/bin/python scripts/twp_cjk_smoke.py
    uv run .venv/bin/python scripts/twp_cjk_smoke.py --model m-a-p/CT-LLM-Base

THE PROBLEM THIS TESTS. `boundary_mask` decides word ends by looking for tokens
that START a word -- leading space, punctuation, specials. Chinese marks no word
boundaries, so the only boundary the mask can see is sentence punctuation, and
the unit silently becomes the CLAUSE. Measured consequence: Chinese prompts
resolve 3-16% of probability mass where English on the same models resolves
80-90%, and 61% of the Chinese units that DO resolve end in sentence
punctuation, median 4 characters, up to 16.

THE FIX UNDER TEST. Make the boundary rule PREFIX-DEPENDENT for CJK. A word
continues while the accumulated surface is still a prefix of some dictionary
entry, and ends when it is not. That is jieba's DAG construction reduced to the
one question this algorithm asks, and it is local -- it needs only the current
surface and the candidate token, never the whole sentence, which is what makes
it usable DURING expansion rather than after it.

WHY NOT CHARACTER-SPLIT, the obvious alternative. Splitting every CJK character
would guarantee ~100% resolution and destroy the result: 么 alone is meaningless,
and 怎么做 as 怎/么/做 reproduces at character level exactly the `co -> c`
uninterpretability that motivated the whole word-level rebuild. Resolution is
not the goal; an ANNOTATABLE unit is, and resolution is how we detect that we
have one.

WHAT WOULD MAKE THIS FAIL, declared before running: if resolved mass rises but
the recovered units are single characters, the dictionary is not segmenting, it
is just terminating early, and this is character-split wearing a dictionary. The
report prints the length distribution for exactly that reason.
"""
import argparse
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA  # noqa: E402

DICT = os.path.join(PATH_DATA, "dict", "jieba_dict_big.txt")
CJK = re.compile(r"[一-鿿㐀-䶿]")
PUNCT = set(".,;:!?\"'()[]{}—-–…/\\*#…") | {"\n", "\r", "\t"}
CJK_PUNCT = set("。，、；：！？「」『』（）《》〈〉…—～·"
                # CURLY QUOTES ADDED 2026-07-30. The set had the corner brackets
                # 「」『』 and not “” (U+201C/U+201D), which are the ordinary Chinese
                # quotation marks in simplified text. Six whisper_want prompts open
                # with “ and would have got NO WORD BOUNDARY there -- the quote cannot
                # be dropped, since direct speech is what separates whisper_want from
                # told_want. Found by the translation pass, not by this smoke test.
                "“”‘’")
THETA, MAX_DEPTH = 0.001, 6


def load_prefixes(path):
    """All dictionary words AND every proper prefix of one.

    The membership test during expansion is "could this surface still grow into
    a word", so proper prefixes must be present or every multi-character word
    would be cut at its first character -- which is character-split by another
    route, the failure mode declared in the docstring.
    """
    words, pref = set(), set()
    with open(path) as f:
        for line in f:
            w = line.split(" ")[0].strip()
            if not w or not CJK.search(w):
                continue
            words.add(w)
            for i in range(1, len(w)):
                pref.add(w[:i])
    return words, pref | words


def is_cjk(s):
    return bool(s) and bool(CJK.search(s))


def clean_surface(s):
    """A word cannot CONTAIN punctuation or whitespace, so truncate at the first.

    The boundary mask asks "does this token START a new word", which is a test
    on the token's FIRST character. Byte-level BPE merges punctuation into the
    middle of tokens -- Qwen has a single token for `做？` and for `？\nA` -- so a
    token whose first character is a CJK glyph can still carry a clause end
    inside it, and the first-character test cannot see it. The observed effect
    was units like `怎么做？` and `怎么做？\nA` reported as words.

    Truncating at the first punctuation/whitespace is exact for our purpose
    rather than a cleanup: the mass on `怎么做？` IS the mass on the word 怎么做
    terminated by punctuation, which is what we want to credit it. Two surfaces
    that differ only in trailing punctuation MERGE, which is correct -- they are
    the same word.
    """
    for i, c in enumerate(s):
        if c in PUNCT or c in CJK_PUNCT or c.isspace():
            return s[:i]
    return s


def static_mask(tok, n):
    """The CURRENT rule, reproduced exactly, as the baseline arm."""
    m = np.zeros(n, dtype=bool)
    for i in range(n):
        try:
            s = tok.convert_ids_to_tokens(i)
        except Exception:
            m[i] = True
            continue
        if s is None:
            m[i] = True
        elif s.startswith(("Ġ", "▁", " ")):
            m[i] = True
        elif s and (s[0] in PUNCT or s[0] in CJK_PUNCT or s.strip() == ""):
            m[i] = True
        elif s.startswith("<") and s.endswith(">"):
            m[i] = True
    return m


@torch.no_grad()
def next_dist(model, tok, pids, prefixes, dev, batch=48):
    out = []
    for i in range(0, len(prefixes), batch):
        ch = prefixes[i:i + batch]
        seqs = [pids + list(p) for p in ch]
        L = max(len(s) for s in seqs)
        pad = tok.pad_token_id or 0
        ids = torch.tensor([[pad] * (L - len(s)) + s for s in seqs], device=dev)
        att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
        lg = model(ids, attention_mask=att).logits[:, -1, :].float()
        out.append(torch.softmax(lg, -1).cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def expand(model, tok, prompt, dev, bmask, pref=None, theta=THETA):
    """pref=None -> the current static rule. pref=set -> dictionary boundaries.

    The dictionary arm overrides the static mask for CJK continuations ONLY:
    a CJK token that would extend the surface beyond any dictionary prefix
    becomes a BOUNDARY, and one that keeps it inside the trie does not. Non-CJK
    tokens keep the existing rule, so mixed text is unaffected.
    """
    pids = tok.encode(prompt)
    lg = model(torch.tensor([pids], device=dev)).logits[0, -1, :].float()
    P0 = torch.softmax(lg, -1).cpu().numpy()
    sel = np.flatnonzero(P0 >= theta)
    live = [((int(t),), float(P0[t]), int(t)) for t in sel]
    words, res_tail, res_drop = {}, float(1.0 - P0[sel].sum()), 0.0
    cache = {}
    # every CJK token in the vocabulary, resolved once
    # DECODE, do not convert_ids_to_tokens. Qwen is byte-level BPE, so the
    # token STRING for 我 is 'æĪĳ' -- convert_ids_to_tokens returns the byte
    # mangling and no CJK test can see through it. The first version found
    # ZERO CJK tokens in a Chinese model's vocabulary, which is the impossible
    # number that exposed it.
    cjk_ids, cjk_str = [], []
    if pref is not None:
        for i in range(len(bmask)):
            try:
                t = tok.decode([i])
            except Exception:
                continue
            if t and is_cjk(t) and t == t.strip():
                cjk_ids.append(i); cjk_str.append(t)
    cjk_ids = np.array(cjk_ids, dtype=int)
    if pref is not None:
        print(f"  [{len(cjk_ids):,} CJK tokens in vocab]", flush=True)
    for _ in range(MAX_DEPTH):
        if not live:
            break
        dist = next_dist(model, tok, pids, [p for p, _, _ in live], dev)
        nxt = []
        for (prefx, mass, t1), row in zip(live, dist):
            surf = clean_surface(tok.decode(list(prefx)).strip())
            b = bmask
            if pref is not None and is_cjk(surf):
                # EVERY CJK token must be judged, not just the probable ones.
                # The first version tested only continuations above theta/mass,
                # on the reasoning that lower ones get dropped anyway -- true for
                # the CONTINUATION set, false for TERMINATION, which sums over
                # all boundary tokens. The improbable CJK tokens that should
                # have ended the word were still counted as continuations, so
                # the word ran on to the clause end and its mass drained through
                # `drop`. That is the entire effect being measured, and the
                # optimisation deleted it.
                key = surf
                b = cache.get(key)
                if b is None:
                    b = bmask.copy()
                    inside = np.fromiter(((surf + s) in pref for s in cjk_str),
                                         dtype=bool, count=len(cjk_str))
                    b[cjk_ids] = ~inside
                    cache[key] = b
            term = float(row[b].sum())
            if surf:
                words[(surf, t1)] = words.get((surf, t1), 0.0) + mass * term
            else:
                res_drop += mass * term
            cont = np.flatnonzero(~b)
            m2 = mass * row[cont]
            keep = m2 >= theta
            for t, mm in zip(cont[keep], m2[keep]):
                nxt.append(((*prefx, int(t)), float(mm), t1))
            res_drop += float(m2[~keep].sum())
        live = nxt
    res_open = float(sum(m for _, m, _ in live))
    return words, dict(tail=res_tail, drop=res_drop, open=res_open,
                       total=res_tail + res_drop + res_open)


def report(name, words, res):
    resolved = sum(words.values())
    cons = resolved + res["total"]
    top = sorted(words.items(), key=lambda kv: -kv[1])[:8]
    cjkw = [w for (w, _), _ in words.items() if is_cjk(w)]
    lens = [len(w) for w in cjkw] or [0]
    print(f"\n--- {name} ---")
    print(f"  resolved mass {resolved:.4f}   residual {res['total']:.4f} "
          f"(tail {res['tail']:.3f} drop {res['drop']:.3f} open {res['open']:.4f})")
    print(f"  conservation  {cons:.7f}")
    print(f"  words {len(words)}   CJK word chars: median {np.median(lens):.0f} "
          f"1ch {sum(1 for l in lens if l == 1)/len(lens):.0%} "
          f">5ch {sum(1 for l in lens if l > 5)/len(lens):.0%}")
    print("  top: " + "  ".join(f"{w}={p:.3f}" for (w, _), p in top))


def main(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"dictionary: {DICT}")
    words_set, pref = load_prefixes(DICT)
    print(f"  {len(words_set):,} CJK words, {len(pref):,} words+prefixes in trie")
    print(f"model: {a.model}\nprompt: {a.prompt!r}")
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float16, trust_remote_code=True).to("mps").eval()
    bmask = static_mask(tok, model.config.vocab_size)
    w0, r0 = expand(model, tok, a.prompt, "mps", bmask, pref=None)
    report("BASELINE (current rule: punctuation only)", w0, r0)
    w1, r1 = expand(model, tok, a.prompt, "mps", bmask, pref=pref)
    report("DICTIONARY BOUNDARIES", w1, r1)
    b, d = sum(w0.values()), sum(w1.values())
    print(f"\nRESOLVED MASS {b:.4f} -> {d:.4f}   ({d/max(b,1e-9):.1f}x)")
    print("English on these models resolves 0.80-0.90; that is the target.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--prompt", default="我的租户拒绝支付新的租金。我应该")
    main(ap.parse_args())
