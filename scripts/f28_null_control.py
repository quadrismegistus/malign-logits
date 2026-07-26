#!/usr/bin/env python3
"""CONTROL B: the base-vs-base null for the positional-asymmetry residue.

The residue claims a positional signature -- forward resistance peaks at pos0,
reverse at pos1 -- and treats that as alignment-specific. But no null exists:
the comparison has been against ZERO, which assumes any positional structure is
produced by alignment. Two UNRELATED base models scoring each other's
storylines give the generic cross-model divergence profile by position. The
aligned/base contrast has to beat that shape.

The beams stash carries 1,825 cross-family annotation links but ZERO
base->other-family-base, so this cannot be built from cache and needs one
teacher-forcing pass. No generation: the storylines already exist, they are
only re-scored under a foreign base.
"""
import argparse, collections, csv, math, statistics as st

import torch

from malign_logits.cache import get_cache
from malign_logits.models import load_model
from malign_logits import MODEL_FAMILIES

MAXPOS = 10


def norm(x):
    return str(x).split("/")[-1].replace("-", "_").replace(".", "_")


def score(model, tok, prompt, token_texts_list, device, batch=8):
    """P(token_i | prompt + tokens[:i]) under this model, per storyline."""
    out = []
    for i in range(0, len(token_texts_list), batch):
        chunk = token_texts_list[i:i + batch]
        for tts in chunk:
            text = prompt + "".join(tts)
            ids = tok.encode(text, return_tensors="pt").to(device)
            plen = len(tok.encode(prompt))
            with torch.no_grad():
                lg = model(ids).logits[0]
            probs = []
            for j in range(plen, ids.shape[1]):
                p = torch.softmax(lg[j - 1].float(), dim=-1)
                probs.append(float(p[ids[0, j]]))
            out.append(probs[:MAXPOS])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="olmo:pythia,pythia:olmo",
                    help="comma-separated genfam:scorerfam, unrelated bases")
    ap.add_argument("--max-prompts", type=int, default=12)
    ap.add_argument("--max-beams", type=int, default=15)
    a = ap.parse_args()

    cm = get_cache(); s = cm._stash("beams")
    acc = collections.defaultdict(list)
    for spec in a.pairs.split(","):
        gf, sf = spec.split(":")
        gbase = MODEL_FAMILIES[gf].base
        sbase = MODEL_FAMILIES[sf].base
        keys = [k for k in s if isinstance(k, dict) and k.get("type") == "beam_cross_v1"
                and norm(k.get("source")) == norm(gbase)][:a.max_prompts]
        if not keys:
            print(f"  {spec}: no storylines for {gbase}"); continue
        model, tok = load_model(sbase)
        dev = next(model.parameters()).device
        n = 0
        for k in keys:
            try:
                beams = s[k][:a.max_beams]
            except Exception:
                continue
            tts = [(b.get("token_texts") if isinstance(b, dict) else b.token_texts) for b in beams]
            srcp = [(b.get("base_token_probs") if isinstance(b, dict) else b.base_token_probs)
                    for b in beams]
            scored = score(model, tok, k["prompt"], tts, dev)
            for sp, sc in zip(srcp, scored):
                if not sp or not sc:
                    continue
                n += 1
                for i in range(min(MAXPOS, len(sp), len(sc))):
                    if sp[i] > 0 and sc[i] > 0:
                        acc[(spec, i)].append(math.log2(sp[i] / sc[i]))
        print(f"  {spec}: {n} storylines scored under {sbase.split('/')[-1]}")
        del model
        import gc; gc.collect()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()

    rows = [dict(pair=p, pos=i, n=len(v), bits=st.mean(v)) for (p, i), v in acc.items() if len(v) >= 20]
    if not rows:
        print("no data"); return
    with open("data/f28_null_control.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n{'pair':22s}" + "".join(f"{'p'+str(i):>7s}" for i in range(6)))
    for p in sorted({r['pair'] for r in rows}):
        line = ""
        for i in range(6):
            v = [r["bits"] for r in rows if r["pair"] == p and r["pos"] == i]
            line += f"{v[0]:>7.2f}" if v else f"{'-':>7s}"
        print(f"{p:22s}{line}")
    print("\nfor comparison — forward 2.37 0.78 0.90 0.90 0.78 0.76")
    print("                 reverse 0.37 1.35 0.62 0.64 0.70 0.65")


if __name__ == "__main__":
    main()
