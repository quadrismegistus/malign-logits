"""q_u2_weighting.py — U2's price: unweighted vs inverse-variance, COUNTS ONLY.

Reads `n_fallers` (a count) and identity columns. Reads NO `tail_excess`
and NO `A` value. The dispersion is N's PUBLISHED within-cluster MSW SD
(0.0602), not a quantity measured here.

Known answer first: the unweighted SE must reproduce §Q3's published
0.001305 at k=33, or this instrument is wrong and nothing below stands.
"""
import json
import math

N_ART = "meta/M01_displacement/results/result_n_primary.json"
POP_684 = "meta/M01_displacement/results/population_d_684.json"
CATALOGUE = "data/prompt_categorisation.json"

SIGMA = 0.0602          # N's published within-cluster MSW SD of tail_excess
FLOOR = 10
KNOWN_SE_UNWEIGHTED = 0.001305   # §Q3's published figure at k=33

TRANSGRESSIVE = {"violence", "sexual", "profanity", "substance", "death",
                 "taboo", "animal", "betrayal", "power", "property"}

pop = json.load(open(POP_684))
pair_ids = set(pop["ids"])

rows = json.load(open(CATALOGUE))["prompts"]

# pair members take their PAIR ROLE and never their domain (§Q1.1 precedence)
pair_texts = set()
for r in rows:
    if (r.get("pair_role") and r.get("contrast_type") == "transgressive_swap"
            and str(r.get("source", "")).startswith("M01_PAIRS")
            and r.get("pair_id") in pair_ids):
        pair_texts.add(r["prompt"])

# text -> partition, first-in-catalogue wins on multi-domain
text_part = {}
for r in rows:
    t = r["prompt"]
    if t in pair_texts or t in text_part:
        continue
    dom = r.get("domain")
    if dom in TRANSGRESSIVE:
        text_part[t] = "T"
    elif dom == "neutral":
        text_part[t] = "N"

art = json.load(open(N_ART))
cells = art["cells"]

# per cluster (base checkpoint): analysed cell counts on each side
counts = {}
for c in cells:
    p = text_part.get(c["prompt"])
    if p is None:
        continue
    if c["n_fallers"] == 0:          # §Q1.2 clause 7: analysed = non-zero-faller
        continue
    slot = counts.setdefault(c["base"], {"T": 0, "N": 0})
    slot[p] += 1

qualifying = {b: v for b, v in counts.items()
              if v["T"] >= FLOOR and v["N"] >= FLOOR}
k = len(qualifying)

# v_i = sigma^2 * (1/n_T + 1/n_N)
v = [SIGMA ** 2 * (1.0 / s["T"] + 1.0 / s["N"]) for s in qualifying.values()]

se_unw = math.sqrt(sum(v)) / k
se_iv = 1.0 / math.sqrt(sum(1.0 / x for x in v))

print("clusters with cells in both arms : %d" % len(counts))
print("qualifying at floor >= %d        : k = %d" % (FLOOR, k))
print()
print("KNOWN ANSWER  published SE (unweighted, k=33) = %.6f" % KNOWN_SE_UNWEIGHTED)
print("              recomputed                      = %.6f" % se_unw)
ok = abs(se_unw - KNOWN_SE_UNWEIGHTED) <= 5e-6
print("              agree to 5e-6                   : %s" % ok)
print()
print("SE unweighted        = %.6f   MDE(80%%, a=0.0167) = %.5f"
      % (se_unw, 3.2356 * se_unw))
print("SE inverse-variance  = %.6f   MDE(80%%, a=0.0167) = %.5f"
      % (se_iv, 3.2356 * se_iv))
print("ratio  unweighted / IV = %.4f" % (se_unw / se_iv))
print()
smallest = min(qualifying.items(), key=lambda kv: kv[1]["T"] + kv[1]["N"])
largest = max(qualifying.items(), key=lambda kv: kv[1]["T"] + kv[1]["N"])
w = {b: (1.0 / (SIGMA ** 2 * (1.0 / s["T"] + 1.0 / s["N"])))
     for b, s in qualifying.items()}
tot = sum(w.values())
print("smallest qualifying cluster : %s  T=%d N=%d  IV weight %.2f%% (unweighted %.2f%%)"
      % (smallest[0], smallest[1]["T"], smallest[1]["N"], 100 * w[smallest[0]] / tot, 100.0 / k))
print("largest  qualifying cluster : %s  T=%d N=%d  IV weight %.2f%% (unweighted %.2f%%)"
      % (largest[0], largest[1]["T"], largest[1]["N"], 100 * w[largest[0]] / tot, 100.0 / k))
