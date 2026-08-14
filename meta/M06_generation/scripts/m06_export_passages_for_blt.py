"""Export the passages BLT should score, as JSONL, for a GPU run on another box.

    uv run python meta/M06_generation/scripts/m06_export_passages_for_blt.py
    -> data/raw/blt_passages.jsonl.gz   (gitignored; ~1/3 of raw size)

Corpora: `passage`, `f11_l2`, `y`. NOT `beam_fc` (RH): its 1.4M texts average
~98 bytes and are a different unit of analysis.

DEDUPLICATED ON (prompt, text), because BLT surprisal is a property of the
STRING. The cache is keyed {ref, prompt, text} for the same reason, so two
models that emitted identical text share one score and must not be scored twice.
`corpora` lists every corpus a pair appears in rather than picking one, so the
dedup does not quietly drop provenance.

Undisturbed only (`forced_word = ''`). Forced generations are a different
population and would multiply the volume ~5x.

FIELDS: prompt, text, corpora, n_bytes, n_chars, script. `script` is by CJK
CHARACTER PROPORTION, not "contains any non-ASCII" -- the BLT pilot's `zh`
bucket came back at 1.83 bytes/char because that filter selects mixed text and
I labelled it a language. Here `zh` means >=50% CJK by character; anything
between is `mixed`, named rather than assigned.

n_bytes is included so the receiving side can order by cost and estimate
runtime without re-tokenizing: BLT is byte-level, so bytes ARE its unit.
"""
import gzip
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(ROOT, "data/raw/blt_passages.jsonl.gz")
CH = os.environ.get("MALIGN_CH_BIN", "clickhouse")
CORPORA = ("passage", "f11_l2", "y")

#: CJK unified ideographs plus the common extensions we actually see.
CJK = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF),
       (0x3000, 0x303F), (0xFF00, 0xFFEF))


def cjk_frac(s):
    if not s:
        return 0.0
    n = sum(1 for ch in s if any(a <= ord(ch) <= b for a, b in CJK))
    return n / len(s)


def main():
    q = ("SELECT prompt, text, groupUniqArray(corpus) AS corpora "
         "FROM malign_logits.gen_sequences "
         "WHERE forced_word = '' AND corpus IN ('%s') "
         "GROUP BY prompt, text FORMAT JSONEachRow" % "','".join(CORPORA))
    pr = subprocess.Popen([CH, "client", "-q", q], stdout=subprocess.PIPE,
                          text=True, bufsize=1 << 22)
    n = 0
    tot_bytes = 0
    by_script = {}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        for line in pr.stdout:
            try:
                r = json.loads(line)
            except Exception:
                continue
            t = r["text"]
            fr = cjk_frac(t)
            script = "zh" if fr >= 0.5 else ("en" if fr < 0.05 else "mixed")
            nb = len(t.encode("utf-8"))
            f.write(json.dumps({"prompt": r["prompt"], "text": t,
                                "corpora": r["corpora"], "n_bytes": nb,
                                "n_chars": len(t), "script": script},
                               ensure_ascii=False) + "\n")
            n += 1
            tot_bytes += nb
            d = by_script.setdefault(script, {"n": 0, "bytes": 0})
            d["n"] += 1
            d["bytes"] += nb
    pr.wait()

    print("wrote %s" % OUT)
    print("  distinct (prompt, text): %s" % format(n, ","))
    print("  total text bytes       : %.1f MiB" % (tot_bytes / 2**20))
    print("  file on disk           : %.1f MiB"
          % (os.path.getsize(OUT) / 2**20))
    for s, d in sorted(by_script.items(), key=lambda kv: -kv[1]["n"]):
        print("  %-6s %9s passages  %8.1f MiB  mean %5.0f bytes"
              % (s, format(d["n"], ","), d["bytes"] / 2**20, d["bytes"] / d["n"]))
    #: at the CPU rate measured on this box; the GPU figure is the point of the
    #: export, so state the baseline it is being compared against.
    print("  CPU serial at 600 B/s  : %.1f hours" % (tot_bytes / 600 / 3600))
    return 0


if __name__ == "__main__":
    sys.exit(main())
