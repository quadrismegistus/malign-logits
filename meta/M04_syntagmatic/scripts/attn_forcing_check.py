#!/usr/bin/env python
"""Does FORCING a word distort attention-back to it?

Everything in `attention_back_cross_own.md` compares forced sites to forced
sites, which quietly assumes forcing is benign. It is testable, on the words the
model sometimes chooses anyway.

    CHOSEN   undisturbed sequences whose slot token IS the target word
    FORCED   the forced cell for that word

THE SLOT SITS AT THE SAME ABSOLUTE POSITION IN BOTH, which is what makes the
comparison legitimate:

    forced        full_ids = prompt + word + 256      plen = |prompt| + |word|
                  slot = plen - |word| .. plen-1      -> index |prompt|
    undisturbed   full_ids = prompt + 256             plen = |prompt|
                  slot = plen                         -> index |prompt|

So Finding A's objection (b) to forced-vs-undisturbed -- that `plen` offsets the
scored positions -- does not apply here. That objection is about scoring
CONTINUATION positions by absolute index. This anchors on the slot, and the slot
is at the same index either way. Objection (a) still applies to any comparison of
the two arms' commitment state, which is why this asks only "is the measurement
distorted", not "does forcing cause repression".

ONLY AVAILABLE WHERE THE MODEL CHOOSES THE WORD, WHICH EXCLUDES FALLERS. A
faller is a word the aligned arm stops choosing: on the pilot cell the aligned
model picked `penis` once in 50 draws against `cock` twelve times. So this
validates the instrument on risers and non-movers and the faller inherits the
verdict rather than earning it. That is a real limit and it is not removable by
more sequences from this corpus.

    attn_forcing_check.py --pair "A>B" --prompt sexual_explicit_1 --words cock,thumb
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("LITMOD_DATA_DIR",
                      "/Users/rj416/github/largeliterarymodels/data")

from attn_delta import Scorer, load_cells, prompt_text   # noqa: E402


def undisturbed(pair, prompt_id):
    """role -> (model, [sequences]) for the word=None cells."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "y_y-*",
                                           "y__*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if (r.get("pair") == pair and r["prompt_id"] == prompt_id
                    and r.get("word") is None):
                out[r["role"]] = (r["model"], r["sequences"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--prompt", default="sexual_explicit_1")
    ap.add_argument("--words", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    import numpy as np
    import torch
    from scipy.stats import mannwhitneyu

    dev = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    base_id, al_id = a.pair.split(">")
    forced = load_cells(a.pair, a.prompt)
    und = undisturbed(a.pair, a.prompt)
    ptext = prompt_text(a.prompt)
    print("pair %s\nprompt %r\n" % (a.pair, ptext))

    for role, mid in (("base", base_id), ("aligned", al_id)):
        S = Scorer(mid, dev)
        plen_p = len(S.tok.encode(ptext, add_special_tokens=False))
        print("== %-8s %s   %d x %d heads   |prompt|=%d"
              % (role, mid.split("/")[-1], S.L, S.H, plen_p))
        for w in a.words.split(","):
            wid = S.tok.encode(" " + w, add_special_tokens=False)
            fs = forced.get((w, role), [])[:a.n]
            #: CHOSEN = undisturbed sequences whose slot token is this word.
            #: Matched on the decoded string, not the id, so a tokenizer that
            #: splits the word differently in the two contexts still matches.
            cs = []
            if role in und:
                for s in und[role][1]:
                    tokid = s["full_ids"][s["plen"]:s["plen"] + len(wid)]
                    if S.tok.decode(tokid).strip() == w:
                        cs.append(s)
            if len(cs) < 3 or not fs:
                print("   %-10s chosen n=%d  forced n=%d  -- too few chosen"
                      % (w, len(cs), len(fs)))
                continue
            #: Both anchored on the slot, which is at index |prompt| either way.
            F = [S.back(s["full_ids"], s["plen"] - len(wid), a.window) for s in fs]
            C = [S.back(s["full_ids"], s["plen"], a.window) for s in cs]
            jf = min(x[1].shape[2] for x in F)
            jc = min(x[1].shape[2] for x in C)
            J = min(jf, jc)
            fm = np.stack([x[1][:, :, :J].mean(2) for x in F], 0)   # (n,L,H)
            cm = np.stack([x[1][:, :, :J].mean(2) for x in C], 0)
            fh, ch = fm.mean(0).ravel(), cm.mean(0).ravel()
            #: Per head across sequences would be n=8 against n=24; instead
            #: compare the two head PROFILES, which is the quantity every D in
            #: the finding is built from.
            u = mannwhitneyu(fh, ch).pvalue
            rel = 100 * (fh.mean() - ch.mean()) / max(ch.mean(), 1e-12)
            corr = np.corrcoef(fh, ch)[0, 1]
            print("   %-10s chosen n=%-3d forced n=%-3d  J=%-3d"
                  "  chosen %.4f  forced %.4f  %+6.1f%%  r=%.3f  p=%.3g"
                  % (w, len(cs), len(fs), J, ch.mean(), fh.mean(), rel, corr, u))
        del S
        print()

    print("  r is the correlation between the forced and chosen HEAD PROFILES.")
    print("  A high r with a level shift means forcing scales attention-back but")
    print("  does not redistribute it -- benign for a within-word D, which")
    print("  differences the level away. A low r means the two conditions engage")
    print("  different heads and no D computed on forced sites transfers.")


if __name__ == "__main__":
    main()
