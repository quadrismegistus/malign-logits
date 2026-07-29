"""Independent audit of Tier 1 v2 on the amber ego->superego edge (docket [509].3).

    uv run .venv/bin/python scripts/f13_tier1v2_amber_audit.py

Implemented from the FROZEN LINES at [510], not from lacan's script. Checking
arithmetic and recomputing a quantity are different audits; only the second
catches a shared premise, and on this instrument the shared premise has been
wrong twice (the closure at [507].2, the edge at [508].4).

THE EDGE IS THE POINT. `displacement_map` computes `dpo - sft` whenever an ego
layer exists, so amber's amplified set lives on AmberChat -> AmberSafe. Every
previously published amber number measured base->superego or base->ego -- the
other side of the target edge. This is the first time the edge the paper's
headline exhibit sits on gets scored.

Full-vocabulary softmax, so R = 1 - sum_F p_post is exact and there is no closure
assumption. Fallers over the WHOLE vocabulary ([510]'s deviation, endorsed at
[511].4): restricting them would leave mass from non-word fallers unaccounted and
reintroduce the closure defect v2 exists to remove.
"""
import os
import statistics as st
import sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits.cache import get_cache  # noqa: E402

MIN_PROB, C = 0.003, 0.5
DT = MIN_PROB                      # displacement_map's default: dt == min_prob
EGO, SUPEREGO = "LLM360/AmberChat", "LLM360/AmberSafe"
BASE = "LLM360/Amber"


def softmax(lg):
    lg = np.asarray(lg, dtype=np.float64).squeeze()
    e = np.exp(lg - lg.max())
    return e / e.sum()


def wordlike_mask(tok, n):
    """[510]'s unit rule, verbatim: leading-space alphabetic tokens of length>=3.
    NOTE `.isalpha()` is True for CJK, so this admits non-Latin scripts -- flagged
    at [511].3(b); the CJK share is reported rather than silently filtered."""
    mask = np.zeros(n, dtype=bool)
    cjk = np.zeros(n, dtype=bool)
    for i in range(n):
        s = tok.convert_ids_to_tokens(i)
        if s is None:
            continue
        s = s.replace("Ġ", " ").replace("▁", " ")
        if s.startswith(" ") and len(s) >= 3 and s[1:].isalpha():
            mask[i] = True
            cjk[i] = not s[1:].isascii()
    return mask, cjk


def cell(p_pre, p_post, wl):
    fall = (p_pre >= MIN_PROB) & (p_post < C * p_pre)
    R = 1.0 - p_post[fall].sum()
    S = p_pre[~fall].sum()
    if S <= 0:
        return None
    null = p_pre * (R / S)
    good = p_post > null
    out = {"infl": R / S, "n_fall": int(fall.sum())}
    rules = {
        "DELTA": (~fall) & (np.maximum(p_pre, p_post) > MIN_PROB)
                 & ((p_post - p_pre) > DT),
        "LITERAL": (~fall) & (p_post > p_pre),
        "MIRROR": (~fall) & (p_post >= MIN_PROB) & (p_post > p_pre / C),
    }
    for name, r in rules.items():
        rw = r & wl
        out[name] = (rw.sum(), (rw & good).sum() / rw.sum() if rw.sum() else None)
    return out


def main():
    cm = get_cache()
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    wl, cjk = None, None

    edges = [(EGO, SUPEREGO, "ego->superego  (TARGET)"),
             (BASE, SUPEREGO, "base->superego (comparator, OFF-EDGE)")]
    for pre, post, label in edges:
        rows = {"DELTA": [], "LITERAL": [], "MIRROR": []}
        infl, nf, ns = [], [], {"DELTA": [], "LITERAL": [], "MIRROR": []}
        prompts = []
        for p in sorted({k["prompt"] for k in []} or []):
            prompts.append(p)
        # prompts with logits for BOTH arms
        cand = []
        for p in _prompts_with_logits(cm, pre, post):
            cand.append(p)
        for p in cand:
            a, b = cm.get_logits(pre, p), cm.get_logits(post, p)
            if a is None or b is None:
                continue
            p_pre, p_post = softmax(a), softmax(b)
            if wl is None:
                wl, cjk = wordlike_mask(tok, len(p_pre))
                print(f"vocab {len(p_pre):,} | word-like {wl.sum():,} "
                      f"| of which non-ASCII {cjk.sum():,} "
                      f"({cjk.sum()/max(1,wl.sum()):.1%})\n")
            r = cell(p_pre, p_post, wl)
            if not r:
                continue
            infl.append(r["infl"])
            nf.append(r["n_fall"])
            for k in rows:
                n, s = r[k]
                ns[k].append(n)
                if s is not None:
                    rows[k].append(s)
        if not infl:
            print(f"{label}: no prompts with logits on both arms")
            continue
        print(f"--- {label} ---  {len(infl)} prompts")
        print(f"    median inflation {st.median(infl):.3f} | "
              f"median fallers {st.median(nf):.0f}")
        for k in ("DELTA", "LITERAL", "MIRROR"):
            if rows[k]:
                print(f"    {k:<8} survival {st.median(rows[k]):>6.1%}   "
                      f"median risers {st.median(ns[k]):>6.0f}   "
                      f"cells {len(rows[k])}")
        print()


def _prompts_with_logits(cm, a, b):
    """Prompts where both arms have cached logits, via the logits stash keys."""
    import os as _os
    from malign_logits import PATH_DATA
    from malign_logits.cache import open_stash
    s = open_stash(_os.path.join(PATH_DATA, "raw", "cache", "logits"))
    ka = {k["prompt"] for k in s.keys() if isinstance(k, dict) and k.get("model") == a}
    kb = {k["prompt"] for k in s.keys() if isinstance(k, dict) and k.get("model") == b}
    return sorted(ka & kb)


if __name__ == "__main__":
    main()
