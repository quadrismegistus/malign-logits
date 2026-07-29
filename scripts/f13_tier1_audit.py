"""Independent audit of Tier 1's single-token re-run (docket [503]).

    uv run .venv/bin/python scripts/f13_tier1_audit.py

The null is reimplemented here from the docket spec rather than read from
lacan's script, because checking someone's arithmetic and recomputing their
quantity are different audits and only the second can catch a shared premise.

SPEC, as frozen ([443]/[458].1):
    intersection-restricted to words in BOTH arms' word_probs
    faller  iff  p_b >= 0.003  AND  p_a < 0.5 * p_b
    R = 1 - sum(p_a over fallers)      S = sum(p_b over non-fallers)
    null(w) = p_b(w) * R / S           excess(w) = p_a(w) - null(w)
    raw riser  iff  p_a > p_b          survives iff excess > 0
    survival = survivors / raw risers, per prompt, then median over prompts

The audit question is [503]'s claim that restricting to single-token words moves
survival by at most 3pp. [488] found word_probs mixes vocabulary-normalised
(single-token) and beam-normalised (multi-token) quantities, so the restriction
is the only configuration where R, S and the masses mean what they say.

[503].4's own flag is checked too: does single-tokenness correlate with the words
alignment moves? If it does, the restriction is a treatment with a direction of
bias, not a neutral filter.
"""
import os
import statistics as st
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA  # noqa: E402
from malign_logits.cache import open_stash  # noqa: E402

MIN_PROB, C = 0.003, 0.5
MIN_SHARED = 20          # [503]'s cell floor: drops cells, never pairs
PAIRS = [
    ("LLM360/Amber", "LLM360/AmberSafe"),
    ("LLM360/Amber", "LLM360/AmberChat"),
    ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"),
    ("allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-Instruct"),
    ("mistralai/Mistral-7B-v0.1", "HuggingFaceH4/zephyr-7b-beta"),
]


def survival(vb, va, keep=None):
    """One cell. Returns (survival, n_raw_risers, inflation) or None."""
    shared = set(vb) & set(va)
    if keep is not None:
        shared = {w for w in shared if keep(w)}
    if len(shared) < MIN_SHARED:
        return None
    fall = [w for w in shared if vb[w] >= MIN_PROB and va[w] < C * vb[w]]
    mass_a = sum(va[w] for w in shared)
    R = mass_a - sum(va[w] for w in fall)   # CLOSURE FIX: word_probs
    # mass over the intersection is ~0.90/0.93, not 1. Using 1 - sum_F p_a
    # claims ~7% of mass that is outside the system, inflating the null
    # uniformly and depressing survival uniformly. lacan [507].2.
    S = sum(vb[w] for w in shared if w not in fall)
    if S <= 0:
        return None
    infl = R / S
    raw = [w for w in shared if w not in fall and va[w] > vb[w]]
    if not raw:
        return None
    surv = [w for w in raw if va[w] - vb[w] * infl > 0]
    return len(surv) / len(raw), len(raw), infl


def main():
    wp = open_stash(os.path.join(PATH_DATA, "raw", "cache", "word_probs"))
    ks = [k for k in wp.keys() if isinstance(k, dict) and k.get("mode") is None]
    by = {}
    for k in ks:
        by.setdefault(k["model"], {})[k["prompt"]] = wp[k]

    print(f"{'aligned arm':<30}{'cells':>6}{'ALL':>8}{'1TOK':>8}{'shift':>8}"
          f"{'kept':>7}{'r(1tok,moved)':>15}")
    for b, a in PAIRS:
        B, A = by.get(b, {}), by.get(a, {})
        prompts = sorted(set(B) & set(A))
        if not prompts:
            print(f"{a.split('/')[-1][:29]:<30}  no shared prompts")
            continue
        tok = AutoTokenizer.from_pretrained(b, trust_remote_code=True)
        cache = {}

        def one_tok(w):
            if w not in cache:
                ids = tok.encode(" " + w, add_special_tokens=False)
                if ids and len(ids) > 1 and not tok.decode([ids[0]]).strip():
                    ids = ids[1:]
                cache[w] = len(ids) == 1
            return cache[w]

        allv, onev, kept, corr = [], [], [], []
        for p in prompts:
            r1 = survival(B[p], A[p])
            r2 = survival(B[p], A[p], keep=one_tok)
            if r1:
                allv.append(r1[0])
            if r2:
                onev.append(r2[0])
            sh = set(B[p]) & set(A[p])
            if len(sh) >= MIN_SHARED:
                kept.append(sum(one_tok(w) for w in sh) / len(sh))
                # [503].4: is single-tokenness related to BEING MOVED at all?
                moved = {w for w in sh if A[p][w] < C * B[p][w] or A[p][w] > B[p][w]}
                if moved and len(moved) < len(sh):
                    pm = sum(one_tok(w) for w in moved) / len(moved)
                    pu = sum(one_tok(w) for w in sh - moved) / len(sh - moved)
                    corr.append(pm - pu)
        if not allv or not onev:
            print(f"{a.split('/')[-1][:29]:<30}  no computable cells")
            continue
        ma, mo = st.median(allv), st.median(onev)
        print(f"{a.split('/')[-1][:29]:<30}{len(allv):>6}{ma:>8.1%}{mo:>8.1%}"
              f"{(mo-ma)*100:>+8.1f}{st.median(kept):>7.1%}"
              f"{st.median(corr) if corr else float('nan'):>+15.3f}")

    print("\nshift = single-token survival minus unrestricted, percentage points.")
    print("r(1tok,moved) = P(single-token | word moved) - P(single-token | not moved),")
    print("  median over prompts. Near 0 = the restriction is a neutral filter;")
    print("  away from 0 = it is a treatment with a direction, per [503].4.")


if __name__ == "__main__":
    main()
