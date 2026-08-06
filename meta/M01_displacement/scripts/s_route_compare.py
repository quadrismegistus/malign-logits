"""Two routes to the same meta-fields: word overlap against meaning.

    uv run python s_route_compare.py

RH proposed both and the comparison is the reason to have run both.

    JACCARD   extensional. Two fields are one field if they hold the same
              WORDS. `s_cluster_dedup.py`, average linkage at J>=0.10.
    SEMANTIC  intensional. Two fields are one field if they MEAN the same.
              Four agents grouped one lexicon each, blind and with free
              vocabulary (`*_free.csv`), then a fifth merged the 160 resulting
              groups across lexicons (`pass2_map.csv`), also blind.

The two can only disagree in two ways and both are interpretable, which is what
makes the comparison worth more than either route alone.

    SEMANTIC ONLY   two taxonomies mean the same field and stock it with
                    DIFFERENT WORDS. Jaccard cannot see this: overlap is its
                    only signal. On the coarse lexicons this is the large cell
                    -- RID selects by regex, WordNet by supersense, the induced
                    taxonomy by an agent reading types, so three membership
                    rules pick different words for one field.
    JACCARD ONLY    shared words without shared meaning. `framenet:Choosing`
                    and `verbnet:chew` cluster because both contain `pick`.
                    This cell is the error rate on finding 16.

NEITHER ROUTE IS THE REFEREE. A disagreement is a fact about the two
instruments, not a verdict on one of them, and the report below prints both
cells rather than scoring one against the other.

UNIT NOTE. Pairs of fields are the unit, not fields, because "do these two
routes group the same things" is a question about pairs. With n fields there
are n(n-1)/2 pairs and the vast majority are grouped by neither route, so the
raw agreement rate is uninformative and is not reported; the two disagreement
cells are.
"""

import itertools
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
LEX = os.path.join(CAMP, "lexicons", "metafields")
OUT = os.path.join(CAMP, "results")

#: the pooled agent handled four lexicons at once and its rows carry their own
#: `resource` column; the other three are one lexicon each
POOLED = "small"
FINE = {"framenet", "usas", "verbnet"}


def semantic_map():
    """field -> meta_field, via each lexicon's free grouping then the merge."""
    p2 = pd.read_csv(os.path.join(LEX, "pass2_map.csv"))
    meta = {(r["source_lexicon"], r["group"]): r["meta_field"] for _, r in p2.iterrows()}
    out = {}
    for src, f in [("framenet", "framenet_free.csv"), ("usas", "usas_free.csv"),
                   ("verbnet", "verbnet_free.csv"), (POOLED, "small_free.csv")]:
        path = os.path.join(LEX, f)
        if not os.path.exists(path):
            print("  MISSING %s" % f)
            continue
        D = pd.read_csv(path)
        for _, r in D.iterrows():
            #: the pooled file names its own resource per row; the others are
            #: named by the file they came from
            res = r["resource"] if "resource" in D.columns else src
            m = meta.get((src, r["group"]))
            if m is not None:
                out["%s:%s" % (res, r["category"])] = m
    return out


def jaccard_map():
    f = os.path.join(OUT, "s_cluster_dedup.csv")
    D = pd.read_csv(f)
    out = {}
    for _, r in D.iterrows():
        for k in str(r["members"]).split("|"):
            out[k] = r["cluster"]
    return out


def main():
    sem, jac = semantic_map(), jaccard_map()
    fields = sorted(set(sem) & set(jac))
    print("fields carried by BOTH routes: %d" % len(fields))
    print("  semantic route covers %d, Jaccard route %d" % (len(sem), len(jac)))
    miss_j = sorted(set(sem) - set(jac))
    miss_s = sorted(set(jac) - set(sem))
    print("  in semantic only (below Jaccard's 5-word floor, or unlabelled): %d" % len(miss_j))
    print("  in Jaccard only (agent dropped or renamed): %d" % len(miss_s))
    if miss_s[:5]:
        print("     e.g. %s" % ", ".join(miss_s[:5]))

    res = {f: f.split(":")[0] for f in fields}
    cells = {"both": [], "sem": [], "jac": []}
    for a, b in itertools.combinations(fields, 2):
        s, j = sem[a] == sem[b], jac[a] == jac[b]
        if s and j:
            cells["both"].append((a, b))
        elif s:
            cells["sem"].append((a, b))
        elif j:
            cells["jac"].append((a, b))
    n = len(fields) * (len(fields) - 1) // 2
    print("\nOF %s FIELD PAIRS" % f"{n:,}")
    print("  grouped by BOTH routes            %6d" % len(cells["both"]))
    print("  SEMANTIC only (Jaccard blind)     %6d" % len(cells["sem"]))
    print("  JACCARD only (overlap artefact)   %6d" % len(cells["jac"]))
    print("  neither                           %6d" % (n - sum(len(v) for v in cells.values())))

    #: the prediction on record: Jaccard should do much better on the
    #: fine-grained lexicons, where same-field categories really do share
    #: vocabulary. If it does not, the coarse-lexicon result was misdiagnosed.
    print("\nDOES JACCARD DO BETTER ON THE FINE-GRAINED LEXICONS? (the prediction on record)")
    print("  %-26s %8s %8s %8s"   % ("cross-resource pairs", "both", "sem", "jac"))
    for lab, keep in [("both fine (fn/usas/vn)", lambda x, y: x in FINE and y in FINE),
                      ("one fine, one coarse", lambda x, y: (x in FINE) != (y in FINE)),
                      ("both coarse", lambda x, y: x not in FINE and y not in FINE)]:
        c = {k: sum(1 for a, b in v if res[a] != res[b] and keep(res[a], res[b]))
             for k, v in cells.items()}
        tot = c["both"] + c["jac"]
        ratio = ("%.1fx" % (c["sem"] / tot)) if tot else "n/a"
        print("  %-26s %8d %8d %8d   semantic finds %s more" % (lab, c["both"], c["sem"], c["jac"], ratio))

    print("\nSAME MEANING, DIFFERENT WORDS -- cross-resource, invisible to overlap:")
    seen = set()
    for a, b in cells["sem"]:
        if res[a] == res[b] or sem[a] in seen:
            continue
        seen.add(sem[a])
        print("  %-28s %-26s + %s" % (sem[a], a, b))
        if len(seen) >= 12:
            break

    print("\nSHARED WORDS, DIFFERENT MEANING -- the error rate on finding 16:")
    for a, b in cells["jac"][:12]:
        if res[a] != res[b]:
            print("  %-26s + %-26s  (%s vs %s)" % (a, b, sem[a], sem[b]))

    pd.DataFrame([(a, b, res[a], res[b], sem[a], sem[b], jac[a], jac[b], k)
                  for k, v in cells.items() for a, b in v],
                 columns=["field_a", "field_b", "res_a", "res_b", "meta_a", "meta_b",
                          "jac_a", "jac_b", "agreement"]).to_csv(
        os.path.join(OUT, "s_route_compare.csv"), index=False)
    print("\nwrote s_route_compare.csv")


if __name__ == "__main__":
    main()
