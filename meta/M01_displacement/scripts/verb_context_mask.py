#!/usr/bin/env python
"""Verb-in-context mask for the M01 pair network.

    uv run python meta/M01_displacement/scripts/verb_context_mask.py

Tags every gated (word, prompt) cell with its contextual POS via
malign_logits.taxonomy.get_pos (spaCy at the prediction site: the word's
POS as the next token of the prompt, not its dictionary POS) and writes
data/verb_context_mask_m01.parquet with is_verb = (pos == "VERB").

PROVENANCE: this is the committed form of the inline run of 2026-08-13/14
(316,516 rows, 1,782 words x 2,211 prompts, verb share 73.9%, ~40 min).
Candidate cells are the cascade's own gates applied at the FULL declared-46
population (word cells >= 150, falls >= 80 OR rises >= 80, per-prompt
support n >= 10, lowercase-alpha) — so every word a per-half gate can pass
is covered. get_pos caches per (tagger, prompt, word); a re-run after the
first is cheap.
"""
import io
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ".")

import pandas as pd  # noqa: E402

CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
OUT = "data/verb_context_mask_m01.parquet"


def esc(s):
    return s.replace("\\", "\\\\").replace("'", "\\'")


def main():
    declared = [ln.strip() for ln in
                open("data/lineage_representative_pairs.txt")
                if ln.strip() and not ln.startswith("#")]
    inlist = ",".join("('" + esc(b) + "','" + esc(a) + "')"
                      for b, a in (p.split(">") for p in declared))
    q = f"""
    WITH wt AS (
      SELECT word FROM (SELECT DISTINCT base,aligned,prompt,word,cls
        FROM malign_logits.movement WHERE (base,aligned) IN ({inlist}))
      GROUP BY word HAVING count() >= 150
        AND (countIf(cls='fall') >= 80 OR countIf(cls='rise') >= 80))
    SELECT word, prompt FROM (SELECT DISTINCT base,aligned,prompt,word
      FROM malign_logits.movement
      WHERE (base,aligned) IN ({inlist}) AND word IN (SELECT word FROM wt))
    GROUP BY word, prompt HAVING count() >= 10
    FORMAT JSONEachRow"""
    r = subprocess.run([CH, "client", "-q", q], capture_output=True,
                       text=True)
    if r.returncode:
        sys.exit(r.stderr[:800])
    d = pd.read_json(io.StringIO(r.stdout), lines=True)
    d = d[d.word.str.match(r"^[a-z']+$")]
    print(f"(word,prompt) pairs to tag: {len(d):,} | "
          f"words {d.word.nunique():,} | prompts {d.prompt.nunique():,}")

    from malign_logits.taxonomy import get_pos
    t0 = time.time()
    rows = []
    for i, (prompt, g) in enumerate(d.groupby("prompt")):
        pos = get_pos(g.word.tolist(), prompt)
        rows.extend((prompt, w, p) for w, p in pos.items())
        if i % 400 == 0:
            print(f"  {i} prompts, {time.time() - t0:.0f}s", flush=True)
    M = pd.DataFrame(rows, columns=["prompt", "word", "pos"])
    M["is_verb"] = M.pos == "VERB"
    M.to_parquet(OUT)
    print(f"mask: {len(M):,} rows, verb share {M.is_verb.mean():.1%}, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
