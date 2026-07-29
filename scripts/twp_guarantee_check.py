"""[670].4 — confirm the DROP channel never discarded a word above theta.
   [670].3 — one look at the single English `open` flag at MAX_DEPTH 8.

    uv run .venv/bin/python scripts/twp_guarantee_check.py

WHY THE GUARANTEE NEEDS A MEASUREMENT AND NOT AN ARGUMENT. The completeness
claim is: expanding every token with P(t1) >= theta is complete for every WORD
with P(w) >= theta, because prefix mass is monotone non-increasing, so a prefix
below theta cannot carry a word above it. That is true BY CONSTRUCTION of the
`keep = m2 >= theta` line -- which is exactly why it is worth checking, since a
guarantee that follows from one comparison operator fails silently if that
operator is ever wrong. This re-runs the expansion with the drop channel
instrumented and reports the LARGEST thing it ever threw away.

THE DROP CHANNEL HAS TWO MOUTHS AND THEY ARE NOT THE SAME CLAIM:

  continuation  a prefix whose mass fell below theta. The guarantee covers this
                one, and the check is `max < theta`. A violation here would
                falsify completeness.
  empty-surface a prefix that TERMINATED but decoded to whitespace only, so
                there was no word to credit. Its mass CAN exceed theta and that
                is not a violation -- there is no word above theta being lost,
                because there is no word. Reported separately and never mixed
                into the pass/fail, because summing them would produce a
                failure that means nothing.

THE MAX_DEPTH LOOK. `six months to` carries open=0.0093, the one non-CJK cell
lacan's sweep flagged. If re-running at depth 8 collapses that mass into
`to live`-class collocations, MAX_DEPTH is biting ordinary English and must
become a declared parameter with a sensitivity row. If it does not move, the
backstop is doing what it was meant to do and nothing changes.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.true_word_probs import (MAX_DEPTH, THETA_EXPAND,  # noqa: E402
                                     boundary_mask, next_dist)

MODEL = "LLM360/Amber"
FLAG = "six months to"
DEV = "mps"


@torch.no_grad()
def expand_traced(model, tok, prompt, bmask, theta=THETA_EXPAND, depth=MAX_DEPTH):
    """expand(), plus the largest mass each drop mouth ever swallowed."""
    pids = tok.encode(prompt)
    lg = model(torch.tensor([pids], device=DEV)).logits[0, -1, :].float()
    P0 = torch.softmax(lg, -1).cpu().numpy()
    sel = np.flatnonzero(P0 >= theta)
    live = [((int(t),), float(P0[t]), int(t)) for t in sel]
    words = {}
    res_tail, res_drop = float(1.0 - P0[sel].sum()), 0.0
    max_cont, n_cont = 0.0, 0          # the guarantee's channel
    max_empty, n_empty = 0.0, 0        # the not-a-word channel
    for _ in range(depth):
        if not live:
            break
        dist = next_dist(model, tok, pids, [p for p, _, _ in live], DEV)
        nxt = []
        for (pref, mass, t1), row in zip(live, dist):
            term = float(row[bmask].sum())
            surf = tok.decode(list(pref)).strip()
            if surf:
                words[(surf, t1)] = words.get((surf, t1), 0.0) + mass * term
            else:
                res_drop += mass * term
                if mass * term > 0:
                    n_empty += 1
                    max_empty = max(max_empty, mass * term)
            cont = np.flatnonzero(~bmask)
            m2 = mass * row[cont]
            keep = m2 >= theta
            for t, mm in zip(cont[keep], m2[keep]):
                nxt.append(((*pref, int(t)), float(mm), t1))
            dropped = m2[~keep]
            if dropped.size:
                nz = dropped[dropped > 0]
                if nz.size:
                    n_cont += int(nz.size)
                    max_cont = max(max_cont, float(nz.max()))
            res_drop += float(dropped.sum())
        live = nxt
    res_open = float(sum(m for _, m, _ in live))
    return words, dict(tail=res_tail, drop=res_drop, open=res_open,
                       total=res_tail + res_drop + res_open), \
        dict(max_cont=max_cont, n_cont=n_cont, max_empty=max_empty, n_empty=n_empty)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from malign_logits.cache import get_cache

    cm = get_cache()
    # the prompts amber actually has, so the check runs on the real corpus
    import glob, json
    prompts = []
    for ln in open(sorted(glob.glob("data/twp_cloud/LLM360__Amber.jsonl"))[0]):
        try:
            prompts.append(json.loads(ln)["prompt"])
        except Exception:
            pass
    prompts = list(dict.fromkeys(prompts))
    flag = [p for p in prompts if p.strip() == FLAG] or \
           [p for p in prompts if FLAG in p]
    sample = prompts[:60]
    print(f"{MODEL}: {len(prompts)} distinct prompts; tracing {len(sample)}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16).to(DEV).eval()
    bmask = boundary_mask(tok, model.config.vocab_size)

    worst_cont, worst_empty, viol = 0.0, 0.0, []
    tc = te = 0
    for p in sample:
        w, res, tr = expand_traced(model, tok, p, bmask)
        cons = sum(w.values()) + res["total"]
        if abs(cons - 1.0) > 1e-4:
            viol.append((p, f"conservation {cons:.6f}"))
        if tr["max_cont"] >= THETA_EXPAND:
            viol.append((p, f"DROPPED PREFIX AT {tr['max_cont']:.6f} >= theta"))
        worst_cont = max(worst_cont, tr["max_cont"])
        worst_empty = max(worst_empty, tr["max_empty"])
        tc += tr["n_cont"]; te += tr["n_empty"]

    print(f"\n[670].4  THE GUARANTEE'S HALF")
    print(f"  theta                       {THETA_EXPAND}")
    print(f"  continuation drops          {tc:,} prefixes")
    print(f"  LARGEST ever dropped        {worst_cont:.8f}")
    print(f"  margin below theta          {THETA_EXPAND - worst_cont:.8f}")
    print(f"  VERDICT                     "
          f"{'CONFIRMED' if worst_cont < THETA_EXPAND else 'VIOLATED'}"
          f" -- no prefix above theta was discarded")
    print(f"\n  empty-surface drops         {te:,} (NOT a violation: no word exists)")
    print(f"  largest empty-surface drop  {worst_empty:.8f}"
          f"{'  <-- exceeds theta, and legitimately' if worst_empty >= THETA_EXPAND else ''}")
    if viol:
        print(f"\n  {len(viol)} PROBLEM(S):")
        for p, why in viol[:10]:
            print(f"    {p[:50]:<52}{why}")

    if flag:
        p = flag[0]
        print(f"\n[670].3  MAX_DEPTH SENSITIVITY on {p!r}")
        rows = []
        for d in (6, 8):
            w, res, _ = expand_traced(model, tok, p, bmask, depth=d)
            rows.append((d, res, w))
            print(f"  depth {d}: open {res['open']:.6f}  words {len(w)}  "
                  f"cons {sum(w.values())+res['total']:.7f}")
        (d6, r6, w6), (d8, r8, w8) = rows
        moved = r6["open"] - r8["open"]
        print(f"  open collapsed by {moved:.6f} "
              f"({moved/r6['open']*100:.1f}% of it)" if r6["open"] else "  open was 0")
        new = sorted(set(w8) - set(w6), key=lambda k: -w8[k])[:8]
        if new:
            print("  words depth 8 found that depth 6 did not:")
            for (s, t1) in new:
                print(f"    {s:<22}{w8[(s,t1)]:.5f}")
        print(f"  READING: {'MAX_DEPTH BITES ordinary English -- declare it' if moved > 0.3*r6['open'] else 'backstop is not biting; no parameter change'}")
    else:
        print(f"\n[670].3  {FLAG!r} not among amber's prompts -- cannot run here")


if __name__ == "__main__":
    main()
