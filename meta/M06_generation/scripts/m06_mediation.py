"""Do M01's movers carry M06's passage effect? Stage 1: extract per-word surprisal.

    uv run python meta/M06_generation/scripts/m06_mediation.py [--cap N] [--pairs K]
    -> results/mediation_words.parquet   (pair, role, word, occurrences, tokens,
                                          sum_s_base, sum_s_aligned)

Runs plan_mediation.md, committed at d79b6c0f before this file existed.

THE INSTRUMENT. Every undisturbed passage is scored by BOTH members of its pair
(1,142,944 self rows and 1,142,944 cross rows in `passage`). So for one text:

    s_base(T)      the base model's per-token surprisal
    s_aligned(T)   the aligned model's per-token surprisal, on THE SAME TOKENS

Composition is held fixed BY CONSTRUCTION, not by matching. Every word is its
own control, so the level term has no frequency confound in it at all.

AND NOTHING IS RE-TOKENIZED. `gen_sequences.token_ids` stores the generation's
actual ids -- measured, `length(token_ids) = n_tokens` for all 238,400
undisturbed passages, so token_ids is the GENERATION ALONE and the prompt is not
in it -- and `logprobs` aligns to it 1:1 (475,092 of 476,800 score rows, 99.64%).
The ids are a record, not a reconstruction. Decoding is the only thing needed,
and decoding is a per-vocabulary lookup done once.

GATE R (reconstruction) is what makes the word grouping checkable rather than
assumed. Tokenizers mark word starts differently and this file handles three
conventions. Rather than trust that, it REBUILDS the text from the pieces it
grouped and compares to the stored `text`. If the surfaces do not reassemble
into the string the model actually emitted, the grouping is wrong and the pair
is dropped. A word-grouping rule that has never been observed failing is a rule
you believe in, not one you have tested.

GATE L (length) drops any text whose two logprob arrays disagree with each other
or with token_ids. Measured beforehand: 596 of 238,400 texts have scorer
disagreement, concentrated in BAAI/Aquila2-7B (592) and croissantllm (4).
EXCLUSION IS BY PAIR, NOT BY ROW -- a pair whose tokenizers disagree on 1% of
texts is not trustworthy on the other 99%.

Note `data/f11_l2_tokenizer_pairs.json` called croissantllm ID-SAFE on 76
probes while the real corpus produces 4 disagreements. The probes are coarser
than the property; the gate here runs on the actual population.
"""
import argparse
import collections
import json
import math
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

OUTD = os.path.join(ROOT, "meta/M06_generation/results")
CH = os.environ.get("MALIGN_CH_BIN", "clickhouse")
CORPUS = "passage"

WORD_RE = None  # set in main, after re import


def ch_rows(q):
    pr = subprocess.Popen([CH, "client", "-q", q + " FORMAT JSONEachRow"],
                          stdout=subprocess.PIPE, text=True, bufsize=1 << 22)
    for line in pr.stdout:
        try:
            yield json.loads(line)
        except Exception:
            continue
    pr.wait()


def token_surfaces(pieces, tk, cache):
    """Exact surface string for each token, using the tokenizer's own decoder.

    Byte-level BPE splits multi-byte characters ACROSS tokens, so a single
    piece has no well-defined surface on its own and any per-piece table is
    wrong at exactly the characters that matter (curly quotes, CJK, emoji).
    So surfaces are recovered by cumulative decode WITHIN each marker-group --
    groups are 1-3 tokens, which keeps this cheap -- and the group's decoded
    prefix difference gives each token its own characters.

    The leading space that a marker encodes is restored by hand: a decoder
    called on a group in isolation drops it, being at position 0 of its input.
    """
    #: an id outside the vocabulary converts to None; it has no surface, so it
    #: cannot open a word and contributes no characters.
    pieces = ["" if p is None else p for p in pieces]
    marks = [p.startswith(("Ġ", "▁", " ")) for p in pieces]
    groups, cur = [], []
    for j, m in enumerate(pieces):
        if marks[j] and cur:
            groups.append(cur)
            cur = []
        cur.append(j)
    if cur:
        groups.append(cur)

    surf = [""] * len(pieces)
    for g in groups:
        key = tuple(pieces[j] for j in g)
        got = cache.get(key)
        if got is None:
            got, prev = [], ""
            for k in range(1, len(g) + 1):
                s = tk.convert_tokens_to_string(list(key[:k]))
                got.append(s[len(prev):] if s.startswith(prev) else s)
                prev = s
            # sentencepiece drops the marker's space when decoding a group in
            # isolation; byte-level BPE already renders it. Prepending
            # unconditionally doubles it, so condition on what came back.
            if marks[g[0]] and not got[0].startswith((" ", "\n", "\t")):
                got[0] = " " + got[0]
            cache[key] = got
        for j, s in zip(g, got):
            surf[j] = s
    return surf


def words_of(text, offsets):
    """Assign every token to a whitespace-delimited word, using char offsets.

    A word is a maximal run of non-whitespace. A token goes to the word holding
    its first non-whitespace character; a whitespace-only token goes to the next
    word, or the previous one if it trails the text. EVERY token is assigned to
    exactly one word, so word surprisals sum to the passage total and the
    decomposition reconciles exactly.

    Returns [(word_string, [token indices])].
    """
    spans = [(m.start(), m.end()) for m in WORD_RE.finditer(text)]
    if not spans:
        return []
    #: char position -> word index, -1 on whitespace
    owner = [-1] * (len(text) + 1)
    for wi, (a, b) in enumerate(spans):
        for c in range(a, b):
            owner[c] = wi
    groups = [[] for _ in spans]
    pending = []
    for j, (s, e) in enumerate(offsets):
        wi = -1
        for c in range(s, min(e, len(text))):
            if owner[c] != -1:
                wi = owner[c]
                break
        if wi == -1:
            pending.append(j)          # whitespace-only: attach to the next word
            continue
        if pending:
            groups[wi].extend(pending)
            pending = []
        groups[wi].append(j)
    if pending:
        groups[-1].extend(pending)
    return [(text[a:b], g) for (a, b), g in zip(spans, groups)]


PUNCT = " \t\n\r\"'`.,;:!?()[]{}<>*_-—–’“”"


def norm(w):
    return w.strip().strip(PUNCT).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=0,
                    help="max texts per (pair, role); 0 = all")
    ap.add_argument("--pairs", type=int, default=0, help="first K pairs; 0 = all")
    ap.add_argument("--out", default="mediation_words.parquet")
    args = ap.parse_args()

    import re
    import pandas as pd
    from transformers import AutoTokenizer

    global WORD_RE
    WORD_RE = re.compile(r"\S+")

    pairs = [r["pair"] for r in ch_rows(
        "SELECT DISTINCT pair FROM malign_logits.gen_sequences "
        "WHERE corpus='%s' AND forced_word='' ORDER BY pair" % CORPUS)]
    mv = set()
    for r in ch_rows("SELECT DISTINCT base, aligned FROM malign_logits.movement"):
        mv.add(r["base"] + ">" + r["aligned"])
    pairs = [p for p in pairs if p in mv]
    if args.pairs:
        pairs = pairs[:args.pairs]
    print("pairs in BOTH passage corpus and movement: %d" % len(pairs))

    rows, report = [], []
    t0 = time.time()
    for pi, pair in enumerate(pairs, 1):
        base, aligned = pair.split(">", 1)
        try:
            tk = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        except Exception as e:
            report.append((pair, "TOKENIZER-UNAVAILABLE",
                           "%s: %s" % (type(e).__name__, str(e)[:60]), 0, 0.0))
            print("[%d/%d] %-58s TOKENIZER-UNAVAILABLE" % (pi, len(pairs), pair[:58]))
            continue
        #: NO is_fast GATE. It tested for offset mapping, which this producer
        #: stopped using when token_surfaces replaced re-tokenization; it went
        #: on rejecting 4 of 42 pairs for a capability nothing here needs.
        seqs, seen = {}, collections.Counter()
        for r in ch_rows(
                "SELECT model, prompt, sample_idx, role, token_ids, text "
                "FROM malign_logits.gen_sequences WHERE corpus='%s' "
                "AND forced_word='' AND pair='%s' ORDER BY role, sample_idx"
                % (CORPUS, pair)):
            if args.cap and seen[r["role"]] >= args.cap:
                continue
            seen[r["role"]] += 1
            seqs[(r["model"], r["prompt"], r["sample_idx"])] = r

        scores = collections.defaultdict(dict)
        for r in ch_rows(
                "SELECT model, prompt, sample_idx, scorer, logprobs "
                "FROM malign_logits.gen_scores WHERE corpus='%s' "
                "AND forced_word='' AND model IN ('%s','%s')"
                % (CORPUS, base, aligned)):
            k = (r["model"], r["prompt"], r["sample_idx"])
            if k in seqs:
                scores[k][r["scorer"]] = r["logprobs"]

        sids = set(tk.all_special_ids)
        scache = {}
        acc = collections.defaultdict(lambda: [0, 0, 0.0, 0.0])
        n_ok = n_len = n_retok = n_err = n_nan = 0
        per_role = collections.Counter()
        for k, q in seqs.items():
          try:
            sc = scores.get(k, {})
            lb, la = sc.get(base), sc.get(aligned)
            ids, text = q["token_ids"], q["text"]
            if lb is None or la is None:
                continue
            if not (len(lb) == len(la) == len(ids)):
                n_len += 1
                continue
            # ibm-granite/granite-3.0-8b-base emits non-finite logprobs on 33
            # of its 5,984 passage rows (99 tokens). One such value turns a
            # word's summed surprisal into NaN and, from there, the pair's
            # mean -- silently, since a NaN mean still prints as a number-ish
            # cell. Drop the text rather than the token, so both scorers see
            # exactly the same occasions.
            if not (all(map(math.isfinite, lb)) and all(map(math.isfinite, la))):
                n_nan += 1
                continue
            keep = [j for j, t in enumerate(ids) if t not in sids]
            pieces = tk.convert_ids_to_tokens([ids[j] for j in keep])
            surf = token_surfaces(pieces, tk, scache)
            # GATE R: the surfaces must reassemble into the string the model
            # actually emitted. Attribution is only as good as this.
            recon = "".join(surf)
            if recon.strip() != text.strip():
                n_retok += 1
                continue
            off, at = [], 0
            for s in surf:
                off.append((at, at + len(s)))
                at += len(s)
            n_ok += 1
            per_role[q["role"]] += 1
            for w, pos in words_of(recon, off):
                pos = [keep[j] for j in pos]
                key = norm(w)
                if not key or not pos:
                    continue
                a = acc[(q["role"], key)]
                a[0] += 1
                a[1] += len(pos)
                # stored values are logprobs; surprisal = -logprob
                a[2] += -sum(lb[j] for j in pos)
                a[3] += -sum(la[j] for j in pos)
          except Exception:
            n_err += 1

        tot = n_ok + n_len + n_retok + n_err + n_nan
        rate = n_ok / tot if tot else 0.0
        #: 0.80 rather than a tighter bar, with the rate PERSISTED: dropping a
        #: pair at 0.87 loses good data while a pair kept at 0.95 still carries
        #: 5% selected-out texts, so retention is a sensitivity to test in
        #: stage 2, not a threshold to hide behind.
        verdict = ("OK" if rate >= 0.80 else
                   "DROP-RETOK" if n_retok > n_len else "DROP-LEN")
        report.append((pair, verdict, "len=%d retok=%d err=%d nan=%d"
                       % (n_len, n_retok, n_err, n_nan), n_ok, rate))
        print("[%d/%d] %-52s %-10s ok %5d (%.3f)  base %d aligned %d  [%.1f min]"
              % (pi, len(pairs), pair[:52], verdict, n_ok, rate,
                 per_role["base"], per_role["aligned"], (time.time() - t0) / 60))
        if verdict != "OK":
            continue
        for (role, w), a in acc.items():
            rows.append((pair, role, w, a[0], a[1], a[2], a[3]))

    df = pd.DataFrame(rows, columns=["pair", "role", "word", "occurrences",
                                     "tokens", "sum_s_base", "sum_s_aligned"])
    os.makedirs(OUTD, exist_ok=True)
    out = os.path.join(OUTD, args.out)
    df.to_parquet(out, index=False)
    #: retention is persisted so stage 2 can test whether the readings depend
    #: on it, rather than the threshold silently becoming the finding.
    json.dump([{"pair": p, "verdict": v, "detail": d, "n_texts": n,
                "retention": rate} for p, v, d, n, rate in report],
              open(os.path.join(OUTD, "mediation_gate.json"), "w"), indent=1)

    print("\nGATE REPORT")
    for pair, v, det, n, rate in report:
        if v != "OK":
            print("  %-52s %-20s %s" % (pair[:52], v, det))
    kept = [r for r in report if r[1] == "OK"]
    print("  kept %d of %d pairs; %d word rows; %s"
          % (len(kept), len(report), len(df), out))
    if not kept:
        print("  REFUSING to report: no pair passed the gates.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
