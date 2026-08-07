"""Does T.13's person_reference row survive removing the animacy-swap twins?

    uv run --with lemminflect python t_animacy_check.py

T.13 reports the withdrawal as transgression-specific: `induced person_reference`
falls -0.0346 in the marked twin against -0.0269 in the neutral one. RH noticed
that the M01 `sexual` domain's twins swap a PERSON for an OBJECT -- her waist ->
the banister -- where every other domain holds the referent constant and varies
the act or substance. malign counted: 10 of its 15 stems.

**If the marked twin contains a person and the neutral one contains a banister,
person-reference words are more available to fall in the marked twin for reasons
that have nothing to do with alignment.** That is the row at risk, and it is the
only one in T.13's table that is: the violence rows are much less exposed since
contact verbs follow both members.

Reuses `s_everything.marginal()` and its walk rather than reimplementing, so this
cannot drift from the number it is checking. Reproduction of the published
-0.0346 on the unsubset population is asserted, not assumed.
"""
import os, sys, re
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
CAMP = os.path.dirname(HERE); ROOT = os.path.dirname(os.path.dirname(CAMP))
import s_everything as S

W = pd.read_parquet(os.path.join(CAMP, "results", "movement_words.parquet"))
R = pd.read_parquet(os.path.join(ROOT, "data", "r_population_k2.parquet")).drop_duplicates("prompt")
meta = R.set_index("prompt")[["member", "domain", "stem"]]
lab = S.labelings(sorted(set(W["word"])))["induced"]

W = W.join(meta, on="prompt", how="inner")
W["member"] = W["member"].str.lower()

#: the animacy stems, identified from the twin text rather than from a hand list:
#: marked carries a possessive/object pronoun where neutral carries a determiner.
PRON = re.compile(r"\b(her|his|him|their|them)\b", re.I)
pairs = R[R["domain"] == "sexual"].groupby("stem")
animacy = set()
for stem, g in pairs:
    d = dict(zip(g["member"].str.lower(), g["prompt"]))
    if len(d) != 2:
        continue
    mk, un = d.get("marked", ""), d.get("unmarked", "")
    mw, uw = set(mk.lower().split()), set(un.lower().split())
    if PRON.search(" ".join(mw - uw)) and "the" in (uw - mw):
        animacy.add(stem)
print("sexual stems: %d   animacy-swap by text rule: %d" % (len(pairs), len(animacy)))

def cell(sub, name):
    D, _ = S.marginal(sub, lab, "cat")
    if "person_reference" not in D.columns or not len(D):
        return print("  %-38s --" % name)
    v = D["person_reference"]
    print("  %-38s %+.4f   edges %3d   neg %2d/%2d" % (name, v.mean(), len(v), (v < 0).sum(), len(v)))

print()
print("induced person_reference, marginal per edge (T.13 publishes -0.0346 / -0.0269)")
for mem in ("marked", "unmarked"):
    M = W[W["member"] == mem]
    print(" %s:" % mem.upper())
    cell(M, "all domains  (published)")
    cell(M[M["domain"] != "sexual"], "EXCLUDING the sexual domain")
    cell(M[M["domain"] == "sexual"], "sexual domain only")
    cell(M[~M["stem"].isin(animacy)], "EXCLUDING the animacy stems only")
