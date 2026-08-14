"""ENGLISH axis, CHINESE candidates. RH's cross-lingual test.

    axis = centroid V(EN prompt + EN naughty_i) - centroid V(EN prompt + EN nice_j)
    s(w) = (V(ZH prompt + ZH word) - origin) . axis

If bge-m3's multilingual alignment holds, a Chinese continuation should land on
the side its English counterpart would. The GLOBAL BARE-WORD version of this
FAILED earlier today -- 鸡巴 +0.044, below `knee` and `table` -- so the question
is whether putting both sides in context rescues it.

CONTROL: the same Chinese words scored on a CHINESE-built axis, so a flat
English-axis result can be told apart from "these words are not separable".
"""
import sys, numpy as np
sys.path.insert(0, "/Users/rj416/github/malign-logits")
from malign_logits.step import Step
from malign_logits.checkpoint import Checkpoint
from sentence_transformers import SentenceTransformer

PRE = "meta-llama/Llama-3.1-8B"
SFT = "allenai/Llama-3.1-Tulu-3-8B-SFT"
EN = "She unzipped his trousers and reached for his"
ZH = "她拉开他的裤子拉链，伸手去摸他的"
EN_NA, EN_NI = ["cock", "penis", "dick", "erection", "crotch"], ["belt", "zipper", "wallet", "phone", "hand"]
ZH_NA, ZH_NI = ["鸡巴", "阴茎", "肉棒", "阳具"], ["裤子", "内裤", "皮带", "钱包"]

m = SentenceTransformer("BAAI/bge-m3", device="cpu")
enc = lambda ts: np.asarray(m.encode(ts, normalize_embeddings=True,
                            show_progress_bar=False, batch_size=64), dtype=np.float32)
def axis_of(prompt, na, ni):
    vg = enc([f"{prompt}{'' if prompt[-1] in '的了' else ' '}{w}" for w in na]).mean(0)
    vn = enc([f"{prompt}{'' if prompt[-1] in '的了' else ' '}{w}" for w in ni]).mean(0)
    a = vg - vn
    return a / np.linalg.norm(a), (vg + vn) / 2.0

cell = Step(Checkpoint(PRE), Checkpoint(SFT)).cell(ZH)
assert cell.is_present, "no zh cell"
base, sft = dict(cell.pre.probs), dict(cell.post.probs)
zh_words = sorted(set(base) | set(sft), key=lambda w: -max(base.get(w,0), sft.get(w,0)))[:26]
Vzh = enc([f"{ZH}{w}" for w in zh_words])

ax_en, o_en = axis_of(EN, EN_NA, EN_NI)
ax_zh, o_zh = axis_of(ZH, ZH_NA, ZH_NI)
s_en = (Vzh - o_en) @ ax_en
s_zh = (Vzh - o_zh) @ ax_zh

print("  ENGLISH axis vs CHINESE axis, on the same Chinese candidates")
print("  axis-axis cosine: %+.3f\n" % float(ax_en @ ax_zh))
print("  %-10s %9s %9s %9s %9s" % ("zh word", "EN axis", "ZH axis", "P_base", "P_sft"))
for w, a, b in sorted(zip(zh_words, s_en, s_zh), key=lambda x: -x[2]):
    print("  %-10s %+9.4f %+9.4f %9.4f %9.4f" % (w, a, b, base.get(w,0), sft.get(w,0)))
from scipy.stats import spearmanr
print("\n  spearman(EN axis, ZH axis) over these %d words: %+.3f" % (len(zh_words), spearmanr(s_en, s_zh).statistic))
for nm, s in (("EN", s_en), ("ZH", s_zh)):
    d = {w: v for w, v in zip(zh_words, s)}
    N = lambda p: sum(q * d[w] for w, q in p.items() if w in d)
    print("  %s axis  N(base) %+.5f   N(sft) %+.5f   delta %+.5f" % (nm, N(base), N(sft), N(sft)-N(base)))
