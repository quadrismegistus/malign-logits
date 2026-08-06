"""Is the faller-riser relation more than the scene? The twin is the control.

    uv run --with lemminflect python v_twin_control.py

RH's proposal, and it is a strictly better control than the "same domain" one
`v_site_relatedness.py` left specified. That script found a site's own risers
closer to its fallers than a STRANGER'S risers -- unanimous, 14/14 families,
but worth only 4.6 percent against a site accounting for the other 95.4. The
objection it could not answer: **a shared topic produces exactly that.** If a
prompt concerns a knife, what falls and what rises will both be knife-adjacent.

"Same domain" is a weak fix because `violence` spans hundreds of unrelated
sentences. **The M01 twins are the strong fix.** Each stem exists as a MARKED
and an UNMARKED prompt differing in ONE WORD:

    MARKED    "He raised the hammer at his coworker and ___"
    UNMARKED  "He raised the clipboard at his coworker and ___"

Same scene, same syntax, same length, same everything but the transgressive
item. Using the twin's risers as the control holds topic almost perfectly
fixed, which makes this the most conservative version of the test available.

THREE LEVELS, so the answer is a gradient and not a binary:

    OWN     fallers here vs risers here
    TWIN    fallers here vs risers from the twin -- same scene, one word apart
    FAR     fallers here vs risers from an unrelated prompt in the family

    OWN < TWIN < FAR   both a scene effect and a relation effect, and the gap
                       OWN-to-TWIN sizes the relation with scene held fixed.
                       This is the first positive evidence for adjacency at any
                       grain, and the only cell in which section 3 of findings V
                       becomes more than "the sets are related".
    OWN == TWIN < FAR  the 4.6 percent was SCENE. The twin's risers do just as
                       well, so nothing about this site's own substitution is
                       special. Question closed at a fifth grain.
    OWN == TWIN == FAR contradicts v_site_relatedness and would mean that
                       result was an artefact of how partners were drawn.

Declared before running, per plan V's lesson that the artefactual cells must be
named first. UNIT: the family. VECTORS: bare bge-m3, `v_bare_vectors.npz`.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
OUT = os.path.join(CAMP, "results")
ROOT = os.path.dirname(os.path.dirname(CAMP))

CACHE = os.path.join(OUT, "v_bare_vectors.npz")
WALK = os.path.join(OUT, "t_ladder_words.parquet")
POP = os.path.join(ROOT, "data", "r_population_k2.parquet")
MIN_SIDE = 2
FAR_DRAWS = 20


def main():
    z = np.load(CACHE, allow_pickle=True)
    words = list(z["words"])
    X = z["X"]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    wi = {w: i for i, w in enumerate(words)}

    P = pd.read_parquet(POP).drop_duplicates("prompt")
    #: prompt -> (stem, member). The twin is the other member of the same stem.
    info = P.set_index("prompt")[["stem", "member"]]
    info["member"] = info["member"].str.lower()
    twin = {}
    for stem, g in info.reset_index().groupby("stem"):
        if g["member"].nunique() == 2 and len(g) == 2:
            a, b = g["prompt"].tolist()
            twin[a], twin[b] = b, a
    print("stems with both members present: %d  (%d prompts)" % (len(twin) // 2, len(twin)))

    W = pd.read_parquet(WALK)
    W = W[W["word"].isin(wi) & W["prompt"].isin(twin)]
    print("movement rows on twin prompts: %s, %d families\n" % (f"{len(W):,}", W["family"].nunique()))

    rng = np.random.default_rng(20260806)
    rows = []
    for (fam, rung), g in W.groupby(["family", "rung"]):
        sites = {}
        for p, h in g.groupby("prompt"):
            f = [wi[w] for w in h[h["role"] == "faller"]["word"].unique()]
            r = [wi[w] for w in h[h["role"] == "riser"]["word"].unique()]
            if len(f) >= MIN_SIDE and len(r) >= MIN_SIDE:
                sites[p] = (f, r)
        usable = [p for p in sites if twin.get(p) in sites]
        if len(usable) < 30:
            continue
        own, tw, far = [], [], []
        for p in usable:
            f, r = sites[p]
            F = X[f]
            own.append(float(np.mean(1 - F @ X[r].T)))
            tw.append(float(np.mean(1 - F @ X[sites[twin[p]][1]].T)))
            others = [q for q in sites if q != p and q != twin[p]]
            pick = rng.choice(len(others), size=min(FAR_DRAWS, len(others)), replace=False)
            far.append(float(np.mean([np.mean(1 - F @ X[sites[others[q]][1]].T) for q in pick])))
        rows.append(dict(family=fam, rung=rung, n_pairs=len(usable),
                         own=float(np.mean(own)), twin=float(np.mean(tw)), far=float(np.mean(far))))
        x = rows[-1]
        print("  %-16s %-10s %4d pairs  own %.4f  twin %.4f  far %.4f   own-twin %+.4f"
              % (fam, rung, x["n_pairs"], x["own"], x["twin"], x["far"], x["own"] - x["twin"]), flush=True)

    D = pd.DataFrame(rows)
    if not len(D):
        print("no family had enough twin pairs.")
        return
    D["own_twin"] = D["own"] - D["twin"]
    D["twin_far"] = D["twin"] - D["far"]
    D.to_csv(os.path.join(OUT, "v_twin_control.csv"), index=False)

    print("\n" + "=" * 88)
    print("OWN vs TWIN vs FAR   unit = family")
    print("=" * 88)
    for rung, g in D.groupby("rung"):
        if len(g) < 6:
            print("  %-10s only %d families, not tested" % (rung, len(g)))
            continue
        p_ot = stats.wilcoxon(g["own"], g["twin"]).pvalue
        p_tf = stats.wilcoxon(g["twin"], g["far"]).pvalue
        print("\n  %s  (%d families)" % (rung, len(g)))
        print("    own  %.4f   twin %.4f   far %.4f" % (g["own"].mean(), g["twin"].mean(), g["far"].mean()))
        print("    OWN - TWIN  %+.5f = %+.2f%% of far   %d/%d families negative   p=%.4f   <- the relation"
              % (g["own_twin"].mean(), 100 * g["own_twin"].mean() / g["far"].mean(),
                 int((g["own_twin"] < 0).sum()), len(g), p_ot))
        print("    TWIN - FAR  %+.5f = %+.2f%% of far   %d/%d families negative   p=%.4f   <- the scene"
              % (g["twin_far"].mean(), 100 * g["twin_far"].mean() / g["far"].mean(),
                 int((g["twin_far"] < 0).sum()), len(g), p_tf))

    b = D[D["rung"] == "base>sft"]
    if len(b) >= 6:
        ot, tf = b["own_twin"].mean(), b["twin_far"].mean()
        p_ot = stats.wilcoxon(b["own"], b["twin"]).pvalue
        print("\n  READ, base>sft:")
        #: FOUR CELLS, NOT THREE. The docstring declared own<twin, own==twin and
        #: all-equal, and the answer was own>twin -- own FARTHER than the twin.
        #: The first version's else-branch collapsed "not closer" into "topic
        #: only" and printed the wrong conclusion. Plan U's outcome map had the
        #: same hole; this one was written knowing that and still had it.
        if p_ot < 0.05 and ot > 0:
            print("    OWN is FARTHER than TWIN, %d/%d families, p=%.4f. With the scene held to"
                  % (int((b["own_twin"] > 0).sum()), len(b), p_ot))
            print("    ONE WORD, what rises at a site sits further from what fell there than what")
            print("    rises at its twin. **Displacement moves AWAY from what it replaced.**")
            print("    Relation %+.2f%% of the reference against the scene's %.2f%% -- small, and"
                  % (100 * ot / b["far"].mean(), abs(100 * tf / b["far"].mean())))
            print("    unanimous. This is ANTI-adjacency: a signed result where four grains had")
            print("    only absences, and the opposite of what metonymy predicts.")
        elif p_ot < 0.05 and ot < 0:
            print("    OWN beats TWIN with the scene held to ONE WORD. The relation is more than")
            print("    topic, and it is worth %.2f%% against the scene's %.2f%%."
                  % (abs(100 * ot / b["far"].mean()), abs(100 * tf / b["far"].mean())))
            print("    **First positive evidence for adjacency at any grain in this campaign.**")
        else:
            print("    OWN does not beat TWIN (p=%.4f). The 4.6%% in findings V section 3 was" % p_ot)
            print("    THE SCENE: a twin differing by one word supplies risers just as close.")
            print("    The relation is topic, and the question is closed at a fifth grain.")
    print("\nwrote v_twin_control.csv")


if __name__ == "__main__":
    main()
