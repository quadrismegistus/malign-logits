"""M06 style measures: plans A and B, on the OSP instrument, imported not ported.

The clause and feature extractors are IMPORTED from RH's ordinary-style-philosophy
repository rather than copied -- the instrument stays byte-identical to the
published one (Ettel & Heuser, *Ordinary Style Philosophy*), and today's
identifier lessons apply to code too: don't retype what you can reference.

    OSP repo:   $M06_OSP_PATH (default /Users/rj416/backup/ordinary-style-philosophy)
    OSP commit: record `git -C $M06_OSP_PATH rev-parse --short HEAD` in any output
                (56f2562 at adapter creation, remote github.com/quadrismegistus/
                ordinary-style-philosophy)
    Extractors used (all pure functions of a stanza Document / Sentence):
        osp.features.extract_pos_feats      (39 POS features)
        osp.features.extract_deprel_feats   (47 dependency-relation features)
        osp.features.extract_syntax_feats_sent  (IC/DC/C/C*/DCw/ICw/Wd/Cd --
                                             the paper's 8 clause statistics,
                                             dependency route via get_clauses_v2)
    NOT used: osp.nlp_utils.get_nlp_doc (writes to OSP's own stash; parses here
    go to the malign-logits cache instead), phrase features (constituency; not
    in the published CV_FEAT_TYPES = pos/deprel/sent), osp TTR (fixed-window
    head TTR; plan A registers MATTR, implemented below).

Construct notes inherited from OSP and stated per plan B Amendment 1 SS3:
    - "independent clause" = get_clauses_v2 main clauses (root-headed);
      UD coordination (`conj`) folds into the head clause; the UD relation
      NAMED `parataxis` counts as a SUBORDINATE-clause introducer here.
      B.H1's construct is therefore "OSP-independent clauses per sentence".
    - extract_syntax_feats_sent floors IC at 1 per sentence.

Parses are cached in the malign-logits stash (cache.py, `m06_stanza_docs`,
keyed by parser string AND text -- a stanza upgrade is a different parse).
Stanza pipeline: tokenize,mwt,pos,lemma,depparse (NO constituency, NO ner --
the published classifier used pos/deprel/sent only, and dropping the two
heavy processors is the difference between days and hours on 1.2M passages).

Environment: stanza/nltk/orjsonl are installed in the repo venv via
`uv pip install` (the pyproject's dynamic-dependencies block rejects
`uv add`); a fresh `uv sync` will need that reinstall. Recorded here so the
absence reads as a known step, not a mystery.

Population (plan A Amendment 2): PRIMARY = the undisturbed arm, `word is None`
in the corpus rows -- present in every cell, no injected word. Forced arms
only via --arms all (secondary table, no verdict language).

Usage:
    uv run python meta/M06_generation/scripts/m06_style.py gate
        # ~50-passage sample for the human segmentation check + length audit
    uv run python meta/M06_generation/scripts/m06_style.py pilot
        # one passage per (pair, prompt), undisturbed arm, measures parquet
    uv run python meta/M06_generation/scripts/m06_style.py run [--arms all]
        # full undisturbed-arm corpus (or all arms for the secondary table)

Outputs (raw-data rule: one row per passage, summaries computed downstream):
    meta/M06_generation/data/m06_style_<mode>.parquet
    meta/M06_generation/data/gate_sample.md   (gate mode only)
"""

import argparse
import glob
import json
import os
import subprocess
import sys

OSP_PATH = os.environ.get("M06_OSP_PATH",
                          "/Users/rj416/backup/ordinary-style-philosophy")
sys.path.insert(0, OSP_PATH)

REPO = os.path.expanduser("~/github/malign-logits")
CORPUS_GLOB = os.path.join(REPO, "data/raw/passage_corpus/box*/y__*.jsonl")
OUT_DIR = os.path.join(REPO, "meta/M06_generation/data")
PROCESSORS = "tokenize,mwt,pos,lemma,depparse"

_NLP = None


def osp_commit():
    try:
        return subprocess.run(["git", "-C", OSP_PATH, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def get_nlp():
    global _NLP
    if _NLP is None:
        import stanza
        _NLP = stanza.Pipeline(lang="en", processors=PROCESSORS, verbose=False)
    return _NLP


def parser_id():
    import stanza
    return f"stanza-{stanza.__version__}:{PROCESSORS}"


def parse_cached(text):
    """Parse via the m06_stanza_docs stash; serialized round-trip on hit."""
    import stanza
    from malign_logits.cache import get_cache
    cache = get_cache()
    pid = parser_id()
    hit = cache.get_stanza_doc(pid, text)
    if hit is not None:
        return stanza.Document.from_serialized(hit)
    doc = get_nlp()(text)
    cache.set_stanza_doc(pid, text, doc.to_serialized())
    return doc


# ── corpus iteration ─────────────────────────────────────────────

def iter_rows():
    for path in sorted(glob.glob(CORPUS_GLOB)):
        if os.path.getsize(path) == 0:
            continue
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("design") != "passage-corpus-105":
                    continue
                yield row


def iter_passages(arms="undisturbed", per_cell=None):
    """Yield dicts: pair, role, model, prompt_id, word(arm), seq_idx, text.

    arms='undisturbed' -> word is None only (plan A Amendment 2 primary).
    per_cell=N         -> first N non-empty sequences per row (pilot).
    """
    for row in iter_rows():
        if arms == "undisturbed" and row.get("word") is not None:
            continue
        n = 0
        for i, seq in enumerate(row.get("sequences") or []):
            text = (seq.get("text") or "").strip()
            if not text:
                continue  # empty-text exclusion; per-pair counts reported below
            yield {
                "pair": row["pair"], "role": row["role"], "model": row["model"],
                "prompt_id": row["prompt_id"], "arm_word": row.get("word"),
                "seq_idx": i, "text": text,
            }
            n += 1
            if per_cell and n >= per_cell:
                break


# ── measures ─────────────────────────────────────────────────────

def mattr(tokens, w):
    """Moving-average type-token ratio, window w tokens."""
    toks = [t.lower() for t in tokens]
    if len(toks) < w:
        return None  # window does not fit; reported as missing, never padded
    vals = []
    from collections import Counter
    counts = Counter(toks[:w])
    vals.append(len(counts) / w)
    for i in range(w, len(toks)):
        counts[toks[i]] += 1
        old = toks[i - w]
        counts[old] -= 1
        if counts[old] == 0:
            del counts[old]
        vals.append(len(counts) / w)
    return sum(vals) / len(vals)


def measure_passage(text):
    """All plan A and plan B measures for one passage. Naming rule applies."""
    from osp.features import (extract_pos_feats, extract_deprel_feats,
                              extract_syntax_feats_sent)
    doc = parse_cached(text)
    sents = doc.sentences
    n_sents = len(sents)
    words = [w.text for s in sents for w in s.words if w.upos != "PUNCT"]
    n_words = len(words)
    if n_sents == 0 or n_words == 0:
        return None

    # plan A
    out = {
        "len_chars": len(text),
        "len_words": n_words,
        "n_sents": n_sents,
        "sent_len_words_mean": n_words / n_sents,
        "ttr_raw_descriptive_only": len({w.lower() for w in words}) / n_words,
        "ttr_mattr_w100": mattr(words, 100),
        "ttr_mattr_w50": mattr(words, 50),
    }
    # A Amendment 1: sentences per 100-token window (coupling diagnostic).
    out["sents_per_window_w100"] = (
        n_sents * 100.0 / n_words if n_words >= 100 else None)

    # plan B: the OSP clause statistics, summed over sentences
    ic = dc = dcw = icw = 0
    cd_max = wd_max = 0
    for s in sents:
        f = extract_syntax_feats_sent(s)
        ic += f["IC"]; dc += f["DC"]; dcw += f["DCw"]; icw += f["ICw"]
        cd_max = max(cd_max, f["Cd"]); wd_max = max(wd_max, f["Wd"])
    out.update({
        "parataxis_indep_clauses_per_sent": ic / n_sents,
        "hypotaxis_dep_clauses_per_sent": dc / n_sents,
        "dep_clause_share": dc / (ic + dc) if (ic + dc) else None,
        "indep_clauses_per_1000w": ic / n_words * 1000,
        "dep_clauses_per_1000w": dc / n_words * 1000,
        "clause_len_words_mean": (dcw + icw) / (ic + dc) if (ic + dc) else None,
        "clause_depth_max": cd_max,
        "word_depth_max": wd_max,
    })

    # the exploratory battery (pos_/deprel_ per 1,000 words; OSP names)
    pos = extract_pos_feats(doc)
    dep = extract_deprel_feats(doc)
    for k, v in pos.items():
        out[f"pos_{k}"] = v / n_words * 1000
    for k, v in dep.items():
        out[f"deprel_{k}"] = v / n_words * 1000
    out["modal_density_md_per_1000w"] = out.get("pos_MD", 0.0)
    return out


# ── modes ────────────────────────────────────────────────────────

def run_measures(mode, arms, per_cell, limit=None):
    import pandas as pd
    rows = []
    for i, p in enumerate(iter_passages(arms=arms, per_cell=per_cell)):
        if limit and i >= limit:
            break
        m = measure_passage(p["text"])
        if m is None:
            continue
        rows.append({**{k: p[k] for k in
                        ("pair", "role", "model", "prompt_id", "arm_word", "seq_idx")},
                     **m})
        if i and i % 500 == 0:
            print(f"  {i} passages measured", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"m06_style_{mode}.parquet")
    df.to_parquet(out)
    meta = {"mode": mode, "arms": arms, "n_rows": len(df),
            "parser": parser_id(), "osp_commit": osp_commit(),
            "_invocation": " ".join(sys.argv)}
    with open(out.replace(".parquet", ".meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {out}: {len(df)} rows; parser {meta['parser']}; "
          f"osp {meta['osp_commit']}")


def gate():
    """~50-passage sample for the human segmentation check + length audit."""
    import random
    rng = random.Random(20260812)
    sample, bloom, curly = [], [], []
    for p in iter_passages(arms="undisturbed", per_cell=2):
        t = p["text"]
        if "bloom" in p["model"].lower() and len(bloom) < 4:
            bloom.append(p)
        elif ("’" in t or "“" in t) and ('"' in t) and len(curly) < 10:
            curly.append(p)  # mixed curly/straight typography, the stressor
        elif rng.random() < 0.002 and len(sample) < 40:
            sample.append(p)
    chosen = (sample + curly + bloom)[:50]
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "gate_sample.md")
    with open(out, "w") as fh:
        fh.write("# M06 gate sample: human segmentation check\n\n"
                 f"parser: {parser_id()} | osp: {osp_commit()}\n\n"
                 "For each passage: Stanza's sentence splits are numbered. "
                 "Mark any split that breaks a sentence or merges two.\n\n")
        for j, p in enumerate(chosen):
            doc = parse_cached(p["text"])
            fh.write(f"## {j+1}. {p['model']} | {p['prompt_id'][:60]}...\n\n")
            for k, s in enumerate(doc.sentences):
                fh.write(f"  [{k+1}] {s.text.strip()}\n")
            fh.write("\n")
    print(f"wrote {out}: {len(chosen)} passages "
          f"({len(curly)} typographic stressors, {len(bloom)} bloom)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate", "pilot", "run", "smoke"])
    ap.add_argument("--arms", default="undisturbed", choices=["undisturbed", "all"])
    args = ap.parse_args()
    if args.mode == "gate":
        gate()
    elif args.mode == "pilot":
        run_measures("pilot", args.arms, per_cell=1)
    elif args.mode == "run":
        run_measures("run", args.arms, per_cell=None)
    elif args.mode == "smoke":
        run_measures("smoke", "undisturbed", per_cell=1, limit=3)
