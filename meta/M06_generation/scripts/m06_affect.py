"""Plan C producer: passage-level affect/concreteness from human norms.

Norms joined on the LEMMA of content words (UPOS NOUN/VERB/ADJ/ADV) from the
shared Stanza parses in the m06_stanza_docs stash -- this script parses
nothing. Norm files are the hub's norms_sources/ set (digests in
meta/norms_digests.md; Warriner 85f6d7e3).

Measures per passage (naming rule):
    valence_extremity_warriner_mean   mean |V - 5| over covered content words
    arousal_warriner_mean, dominance_warriner_mean
    concreteness_brysbaert_mean
    warriner_coverage, brysbaert_coverage   (covered / content words)

Usage:
    uv run python meta/M06_generation/scripts/m06_affect.py pilot
        # the plan-A pilot population (one passage per cell, undisturbed)
    uv run python meta/M06_generation/scripts/m06_affect.py run
        # full undisturbed arm (parses must already be in the stash)
Output: meta/M06_generation/data/m06_affect_<mode>.parquet (+ .meta.json)
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m06_style import (REPO, OUT_DIR, iter_passages, parse_cached, parser_id,
                       osp_commit, is_pseudo_sent, list_lines_share)

NORMS = "/Users/rj416/Dropbox/Prof/Articles/TheoryMachines/norms_sources"
CONTENT_UPOS = {"NOUN", "VERB", "ADJ", "ADV"}


def load_norms():
    warr, brys = {}, {}
    with open(os.path.join(NORMS, "BRM-emot-submit.csv")) as fh:
        for row in csv.DictReader(fh):
            w = row["Word"].lower()
            warr[w] = (float(row["V.Mean.Sum"]), float(row["A.Mean.Sum"]),
                       float(row["D.Mean.Sum"]))
    with open(os.path.join(NORMS, "Concreteness_ratings_Brysbaert_et_al_BRM.txt")) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["Conc.M"]:
                brys[row["Word"].lower()] = float(row["Conc.M"])
    return warr, brys


def measure(doc, warr, brys):
    lemmas = [w.lemma.lower() for s in doc.sentences if not is_pseudo_sent(s)
              for w in s.words if w.upos in CONTENT_UPOS and w.lemma]
    n = len(lemmas)
    if n == 0:
        return None
    wv = [warr[l] for l in lemmas if l in warr]
    bc = [brys[l] for l in lemmas if l in brys]
    out = {
        "n_content_words": n,
        "warriner_coverage": len(wv) / n,
        "brysbaert_coverage": len(bc) / n,
        "valence_extremity_warriner_mean": (
            sum(abs(v - 5.0) for v, a, d_ in wv) / len(wv) if wv else None),
        "arousal_warriner_mean": (sum(a for v, a, d_ in wv) / len(wv) if wv else None),
        "dominance_warriner_mean": (sum(d_ for v, a, d_ in wv) / len(wv) if wv else None),
        "concreteness_brysbaert_mean": (sum(bc) / len(bc) if bc else None),
    }
    return out


def main(mode):
    import pandas as pd
    warr, brys = load_norms()
    per_cell = 1 if mode == "pilot" else None
    rows, i = [], 0
    for p in iter_passages(arms="undisturbed", per_cell=per_cell):
        doc = parse_cached(p["text"])  # stash hit for anything already parsed
        m = measure(doc, warr, brys)
        if m is None:
            continue
        m["list_lines_share"] = list_lines_share(p["text"])
        rows.append({**{k: p[k] for k in
                        ("pair", "role", "model", "prompt_id", "arm_word", "seq_idx")},
                     **m})
        i += 1
        if i % 2000 == 0:
            print(f"  {i} passages", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"m06_affect_{mode}.parquet")
    df.to_parquet(out)
    with open(out.replace(".parquet", ".meta.json"), "w") as fh:
        json.dump({"mode": mode, "n_rows": len(df), "parser": parser_id(),
                   "osp_commit": osp_commit(), "norms": "warriner+brysbaert",
                   "_invocation": " ".join(sys.argv)}, fh, indent=2)
    print(f"wrote {out}: {len(df)} rows")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pilot")
