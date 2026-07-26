"""Tulu 2.5 provisional analysis: does the preference curriculum legislate?

PRE-REGISTERED before any result was inspected (2026-07-25). Thresholds below
are fixed in advance; the point of the suite is that it can go against us.

Design: every model shares one SFT checkpoint (allenai/tulu-2-13b) and one
tokenizer, so algorithm and preference corpus are the only things that vary.

  H0 (Weatherby, Introduction n.2): preference optimization is "downstream"
     banality. The 17 DPO models should be near-interchangeable at matched
     prompts, and between-curriculum spread should not exceed the noise floor.
  H1: the preference corpus legislates. Between-curriculum spread clearly
     exceeds the floor, and is largest where disposition is at stake.

NOISE FLOOR. The suite supplies its own null: three datasets ship at two sizes
(hh_rlhf, nectar, stack_exchange, at 2.6-4x). Same curriculum, different sample
of it. That within-curriculum distance is the floor against which the
between-curriculum distance is judged.

REGISTERED THRESHOLDS
  P1 dose:        median JS(60k, full) is the floor. No threshold; it calibrates.
  P2 curriculum:  H1 requires median between-curriculum JS >= 3x the dose floor.
                  Below 1.5x, H0 stands and Weatherby's note 2 survives here.
  P3 algorithm:   report JS(SFT->DPO) vs JS(SFT->PPO) on the 5 matched datasets,
                  and JS(DPO, PPO) directly. No directional prediction.
  P4 site:        between-curriculum spread on the F21 institutional prompts
                  >= spread on neutral prompts, if disposition tracks what
                  annotators rewarded.
"""
import itertools, json, sys
import numpy as np
import torch

from malign_logits.cache import get_cache
import malign_logits.experiments as E

SFT = "allenai/tulu-2-13b"
D = "allenai/tulu-v2.5-dpo-13b-"
P = "allenai/tulu-v2.5-ppo-13b-"
MATCHED = ["uf-mean", "hh-rlhf-60k", "nectar-60k", "stackexchange-60k",
           "chatbot-arena-2023"]
DOSE = [("hh-rlhf", "hh-rlhf-60k"), ("nectar", "nectar-60k"),
        ("stackexchange", "stackexchange-60k")]
DPO_ALL = MATCHED[:1] + ["hh-rlhf-60k", "nectar-60k", "stackexchange-60k",
    "chatbot-arena-2023", "alpacafarm-gpt4-pref", "alpacafarm-human-pref",
    "argilla-orca-pairs", "capybara", "chatbot-arena-2024", "helpsteer",
    "hh-rlhf", "nectar", "prm-phase-2", "shp2", "stackexchange", "uf-overall"]
DPO_ALL = list(dict.fromkeys(DPO_ALL))

MODE = sys.argv[1] if len(sys.argv) > 1 else "raw"
cm = get_cache()


def probs(model, prompt):
    lg = cm.get_logits(model, prompt, mode=MODE)
    if lg is None:
        return None
    return torch.softmax(torch.as_tensor(lg).float().flatten(), 0).numpy()


def js(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        i = a > 0
        return float(np.sum(a[i] * np.log2(a[i] / b[i])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


prompts = json.load(open("data/all_project_prompts.json"))
ip = getattr(E, "INSTITUTIONAL_PROMPTS", {})
INST = set(ip.values() if isinstance(ip, dict) else ip)
inst = [p for p in prompts if p in INST]
# neutral comparison set: the battery's explicitly neutral items
items = json.load(open("data/f37_prompt_items.json"))
neutral = sorted({i["text"] for i in items if i.get("category") == "neutral"}
                 & set(prompts))
print(f"mode={MODE}  prompts={len(prompts)}  institutional={len(inst)}  neutral={len(neutral)}")

MODELS = [SFT] + [D + d for d in DPO_ALL] + [P + m for m in MATCHED]
have = [m for m in MODELS if cm.has_logits(m, prompts[0], mode=MODE)]
print(f"models cached: {len(have)}/{len(MODELS)}")

# ---- accumulate pairwise JS over prompts, streaming one prompt at a time ----
def collect(subset):
    acc = {}
    n = 0
    for pr in subset:
        vecs = {}
        for m in have:
            v = probs(m, pr)
            if v is not None:
                vecs[m] = v
        if len(vecs) < 2:
            continue
        n += 1
        for a, b in itertools.combinations(sorted(vecs), 2):
            acc.setdefault((a, b), []).append(js(vecs[a], vecs[b]))
    return acc, n

acc, n_used = collect(prompts)
print(f"prompts scored: {n_used}\n")
def pair(a, b):
    v = acc.get((a, b)) or acc.get((b, a))
    return float(np.median(v)) if v else float("nan")

print("P1 DOSE (same curriculum, 2.6-4x more data) — the noise floor")
dose = []
for full, sub in DOSE:
    v = pair(D + full, D + sub)
    dose.append(v)
    print(f"  {full:16s} full vs 60k   JS={v:.5f}")
floor = float(np.median(dose))
print(f"  DOSE FLOOR (median) = {floor:.5f}\n")

print("P2 CURRICULUM (between-curriculum, 17 DPO models sharing one SFT)")
btw = [pair(D + a, D + b) for a, b in itertools.combinations(DPO_ALL, 2)]
btw = [v for v in btw if not np.isnan(v)]
med = float(np.median(btw))
print(f"  n pairs={len(btw)}  median JS={med:.5f}  "
      f"IQR=({np.percentile(btw,25):.5f}, {np.percentile(btw,75):.5f})  max={max(btw):.5f}")
ratio = med / floor if floor else float("inf")
print(f"  RATIO to dose floor = {ratio:.2f}x  -> "
      f"{'H1 (curriculum legislates)' if ratio >= 3 else 'H0 stands' if ratio < 1.5 else 'INDETERMINATE'}\n")

print("P2b DISTANCE FROM THE SHARED SFT CHECKPOINT")
from_sft = sorted(((pair(SFT, D + d), d) for d in DPO_ALL), reverse=True)
for v, d in from_sft:
    print(f"  {d:24s} JS(SFT -> DPO) = {v:.5f}")

print("\nP3 ALGORITHM (5 datasets with both DPO and PPO)")
for d in MATCHED:
    a, b, c = pair(SFT, D + d), pair(SFT, P + d), pair(D + d, P + d)
    print(f"  {d:20s} SFT->DPO {a:.5f}   SFT->PPO {b:.5f}   DPO vs PPO {c:.5f}")

print("\nP4 SITE (between-curriculum spread by prompt type)")
for name, subset in (("institutional", inst), ("neutral", neutral)):
    if len(subset) < 3:
        print(f"  {name}: too few prompts ({len(subset)})"); continue
    a2, _ = collect(subset)
    vv = []
    for x, y in itertools.combinations(DPO_ALL, 2):
        v = a2.get((D + x, D + y)) or a2.get((D + y, D + x))
        if v: vv.append(float(np.median(v)))
    print(f"  {name:14s} n={len(subset):3d} prompts   median between-curriculum JS={np.median(vv):.5f}")
