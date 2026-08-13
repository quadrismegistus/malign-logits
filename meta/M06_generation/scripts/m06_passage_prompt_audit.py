"""The passage corpus stores a 60-CHARACTER TRUNCATED prompt. Audit and repair map.

    uv run python meta/M06_generation/scripts/m06_passage_prompt_audit.py [--post FILE]
    -> data/passage_prompt_resolution.json  (truncated -> verbatim + metadata)

Found while joining M06 to M01: 52 of the 198 `passage` prompts were absent from
`twp_words`, which read as "these prompts were never scored". They were all
present. The lookup was against a TRUNCATED KEY.

WHERE IT COMES FROM, AND IT IS NOT THE INGESTER. The passage producer wrote
`prompt_id = pair + "|" + prompt[:60]` and stored the prompt NOWHERE ELSE, so
the display string became the only text in the record. `ch_ingest.py`
(ingest_passages, the `no pair name contains |` note) recovers the text by
stripping the known pair prefix and REFUSES a row whose prefix is absent rather
than falling back to the id -- it behaved correctly on the input it was given.
`beam_fc` stores prompts to 148 characters, so nothing structural forced this.

THE CONSEQUENCE IS NOT ONLY A BROKEN JOIN. Nine stems have their MARKED and
UNMARKED members diverge AFTER character 60, so both members truncate to one
string. `gen_sequences` is a ReplacingMergeTree ordered on
(corpus, model, prompt, forced_word, sample_idx), so the two members collided
and one was replaced. Every count still looks right: 16 samples per model, the
same as any clean prompt.

Everything this file reports is recomputed here rather than transcribed.
"""
import argparse
import collections
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

DRMATCH = "data/forced_arms_46reps_drmatch.json"
OUT = "data/passage_prompt_resolution.json"
CH = os.environ.get("MALIGN_CH_BIN", "clickhouse")


def q(sql):
    o = subprocess.run([CH, "client", "-q", sql + " FORMAT JSONEachRow"],
                       capture_output=True, text=True).stdout.strip()
    return [json.loads(l) for l in o.split("\n") if l]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", default=None, help="write a docket post body here")
    a = ap.parse_args()

    scope = q("SELECT corpus, max(length(prompt)) AS maxlen, "
              "countDistinct(prompt) AS prompts, "
              "uniqExactIf(prompt, prompt IN (SELECT DISTINCT prompt FROM "
              "malign_logits.twp_words)) AS in_twp "
              "FROM malign_logits.gen_sequences GROUP BY corpus ORDER BY corpus")

    pp = sorted(set(r["prompt"] for r in q(
        "SELECT DISTINCT prompt FROM malign_logits.gen_sequences "
        "WHERE corpus='passage' AND forced_word=''")))

    d = json.load(open(os.path.join(ROOT, DRMATCH)))
    byprompt = {}
    for c in d["cells"]:
        byprompt.setdefault(c["prompt"], c)
    full = list(byprompt)
    tws = set(r["prompt"] for r in q(
        "SELECT DISTINCT prompt FROM malign_logits.twp_words"))

    cov = {p: [f for f in full if f == p or f.startswith(p)] for p in pp}
    clean = {p: v[0] for p, v in cov.items() if len(v) == 1}
    collide = {p: v for p, v in cov.items() if len(v) > 1}
    orphan = [f for f in full if not any(f in v for v in cov.values())]

    #: which member actually survived a collision: the stored plen must equal
    #: the candidate's token count under that model's tokenizer. plen is the
    #: only trace of the prompt left in the row.
    resolved_collisions = {}
    if collide:
        import warnings
        warnings.filterwarnings("ignore")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        from transformers import AutoTokenizer
        tks = {}
        for p, cands in collide.items():
            rows = q("SELECT model, any(plen) AS plen FROM "
                     "malign_logits.gen_sequences WHERE corpus='passage' "
                     "AND forced_word='' AND prompt='%s' GROUP BY model "
                     "LIMIT 4" % p.replace("'", "\\'"))
            votes = collections.Counter()
            for r in rows:
                m = r["model"]
                if m not in tks:
                    try:
                        tks[m] = AutoTokenizer.from_pretrained(
                            m, trust_remote_code=True)
                    except Exception:
                        tks[m] = None
                tk = tks[m]
                if tk is None:
                    continue
                for f in cands:
                    n = len(tk(f, add_special_tokens=False)["input_ids"])
                    if n == r["plen"]:
                        votes[f] += 1
            if votes and len(votes) == 1:
                resolved_collisions[p] = votes.most_common(1)[0][0]

    res = {"_schema": {"truncated": "the 60-char prompt as stored in "
                       "gen_sequences/gen_scores for corpus=passage",
                       "prompt": "verbatim prompt, from " + DRMATCH,
                       "how": "unique-prefix | plen-identified",
                       "note": "stems marked plen-identified lost their OTHER "
                               "member to a store-key collision"},
           "_provenance": {"built_by": "meta/M06_generation/scripts/"
                                       "m06_passage_prompt_audit.py",
                           "source": DRMATCH,
                           "drmatch_prompts": len(full),
                           "drmatch_prompts_in_twp":
                               sum(1 for f in full if f in tws)},
           "map": {}}
    for p, f in clean.items():
        c = byprompt[f]
        res["map"][p] = {"prompt": f, "how": "unique-prefix", "stem": c["stem"],
                         "member": c["member"], "domain": c["domain"],
                         "stratum": c["stratum"]}
    for p, f in resolved_collisions.items():
        c = byprompt[f]
        lost = [byprompt[x]["member"] for x in collide[p] if x != f]
        res["map"][p] = {"prompt": f, "how": "plen-identified", "stem": c["stem"],
                         "member": c["member"], "domain": c["domain"],
                         "stratum": c["stratum"], "lost_members": lost}
    outp = os.path.join(ROOT, OUT)
    json.dump(res, open(outp, "w"), indent=1, ensure_ascii=False)

    #: exactly ONE member per collided key was overwritten, whether or not plen
    #: told us which. Counting every non-survivor would charge the two
    #: unidentified keys twice and contradict the total on the line above.
    missing = collections.Counter()
    for p, v in collide.items():
        keep = resolved_collisions.get(p)
        lost = [f for f in v if f != keep] if keep else v[1:]
        for f in lost:
            missing[(byprompt[f]["domain"], byprompt[f]["stratum"])] += 1
    for f in orphan:
        missing[(byprompt[f]["domain"], byprompt[f]["stratum"])] += 1

    L = []
    w = L.append
    w("TRUNCATION SCOPE (gen_sequences, prompts matching twp exactly)")
    for r in scope:
        w("  %-14s maxlen %3d  prompts %3d  in_twp %3d%s"
          % (r["corpus"], r["maxlen"], r["prompts"], r["in_twp"],
             "   <-- TRUNCATED" if r["in_twp"] < r["prompts"] else ""))
    w("")
    w("RESOLUTION against %s (%d prompts, %d of them in twp)"
      % (DRMATCH, len(full), sum(1 for f in full if f in tws)))
    w("  passage keys                     %d" % len(pp))
    w("  resolved by unique prefix        %d" % len(clean))
    w("  collided (MARKED+UNMARKED)       %d" % len(collide))
    w("    of those, survivor identified  %d  (stored plen vs tokenized candidates)"
      % len(resolved_collisions))
    w("  drmatch prompts never generated  %d" % len(orphan))
    w("  keys left unresolved             %d"
      % (len(pp) - len(clean) - len(resolved_collisions)))
    w("")
    w("GENERATIONS ABSENT FROM THE STORE (recoverable only by re-ingest)")
    w("  overwritten members %d + never generated %d = %d of %d drmatch prompts"
      % (sum(1 for p in collide), len(orphan),
         sum(1 for p in collide) + len(orphan), len(full)))
    for k, n in missing.most_common():
        w("    domain=%-8s stratum=%-12s %d" % (k[0], k[1], n))
    w("")
    w("WROTE %s  (%d entries)" % (OUT, len(res["map"])))
    body = "\n".join(L)
    print(body)
    if a.post:
        open(a.post, "w").write(body + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
