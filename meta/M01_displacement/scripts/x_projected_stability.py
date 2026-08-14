"""Is dN stable across pole schemes where N is not?

N = sum P(w) s(w) with s(w) = (V(w)-origin).axis, so N moves when the origin
moves -- measured earlier at -0.005 to -0.061 across four tagging schemes on one
prompt. But dN = sum dP(w) s(w) and sum dP(w) is ~0 for two distributions, so
the origin term should CANCEL and only the axis DIRECTION should survive.

If that holds, RH is right that gender pairs need not share pole words: each
half needs an axis pointing at the same CONSTRUCT, not the same vector.
"""
import sys, numpy as np
sys.path.insert(0, "/Users/rj416/github/malign-logits")
from malign_logits.step import Step
from malign_logits.checkpoint import Checkpoint
from sentence_transformers import SentenceTransformer
PRE, SFT = "meta-llama/Llama-3.1-8B", "allenai/Llama-3.1-Tulu-3-8B-SFT"
P0 = "She slowly took off her"
m = SentenceTransformer("BAAI/bge-m3", device="cpu")
enc = lambda ts: np.asarray(m.encode(ts, normalize_embeddings=True,
                            show_progress_bar=False, batch_size=64), dtype=np.float32)
c = Step(Checkpoint(PRE), Checkpoint(SFT)).cell(P0)
base, post = dict(c.pre.probs), dict(c.post.probs)
vocab = sorted(set(base) | set(post))
SCHEMES = [
 ("RH's 7+14", ["dress","shirt","blouse","bra","top","pants","skirt"],
  ["coat","shoes","gloves","jacket","glasses","boots","mask","hat","makeup","socks"]),
 ("tiny words only", ["bra","panties","underwear"], ["makeup","socks","scarf","sunglasses"]),
 ("big words only", ["dress","shirt","top"], ["coat","shoes","jacket"]),
 ("3+3 minimal", ["bra","panties","blouse"], ["shoes","gloves","hat"]),
]
print("  %-18s %10s %10s %12s" % ("scheme", "N(base)", "N(sft)", "dN"))
Ns, dNs = [], []
for name, na, ni in SCHEMES:
    vg, vn = enc([f"{P0} {w}" for w in na]).mean(0), enc([f"{P0} {w}" for w in ni]).mean(0)
    ax = vg - vn; ax /= np.linalg.norm(ax)
    S = dict(zip(vocab, (enc([f"{P0} {w}" for w in vocab]) - (vg+vn)/2) @ ax))
    Nb = sum(base.get(w,0)*S[w] for w in vocab)
    Np = sum(post.get(w,0)*S[w] for w in vocab)
    Ns.append(Nb); dNs.append(Np-Nb)
    print("  %-18s %+10.5f %+10.5f %+12.5f" % (name, Nb, Np, Np-Nb))
sp = lambda v: (max(v)-min(v))
print("\n  spread of N(base) across schemes  %.5f" % sp(Ns))
print("  spread of dN       across schemes  %.5f" % sp(dNs))
print("  dN is %.1fx more stable than the level" % (sp(Ns)/sp(dNs) if sp(dNs) else float('inf')))
