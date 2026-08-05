"""Run the S instrument over a frame. Stage 1 (spent 50) and stage 2 (held-out
255) are the same code with a different frame.

    uv run python s_run.py <frame.parquet> <out_long.parquet> [--tag T]

FIELDS COME FROM THE SCHEMA, NOT FROM A LIST IN THIS FILE. The calibration
runner kept its own copy of the ten field names, so revision 3 would have called
the API for all 1,020 items and then died on getattr(r, 'act_lands') -- it maps
first and unpacks after. A schema edit must not be able to cost a thousand
calls.

CODERS ARE THE SEVEN. gpt-4o-mini is excluded per registration_s.md: 15/18 on a
fresh six-item check where six of eight scored 18/18, the only coder that never
reaches B_GENERIC, and it inverted more_transgressive against its own prose.

PREFLIGHT FIRST. gemini-3.6-flash sits behind a free-tier key at 20 requests per
day unless GEMINI_API_KEY is the one in ~/.bash_profile, and a quota wall is
indistinguishable from a coding failure once it reaches the results table:

    export GEMINI_API_KEY="$(bash -c 'source ~/.bash_profile; echo $GEMINI_API_KEY')"
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))
sys.path.insert(0, ROOT)

from malign_logits.tasks.code_operation_binaries import (  # noqa: E402
    OperationBinaries, OperationBinariesTask, prepare)

MODELS = [
    "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash",
    "google/gemini-3.6-flash", "google/gemini-2.5-flash",
    "anthropic/claude-haiku-4-5-20251001", "anthropic/claude-sonnet-5",
    "openai/gpt-5.4-mini",
]

#: Derived, never retyped. slot_note and reason are prose and ride along.
FIELDS = [f for f in OperationBinaries.model_fields if f not in ("slot_note", "reason")]


def preflight():
    from largeliterarymodels.llm import LLM
    bad = []
    for m in MODELS:
        try:
            LLM(model=m).generate("Reply with the single word: ok")
        except Exception as e:
            bad.append((m, str(e).split("\n")[0][:80]))
    for m, e in bad:
        print("  UNREACHABLE %-40s %s" % (m, e))
    if bad:
        print("\nGEMINI_API_KEY in use ends ...%s" % os.environ.get("GEMINI_API_KEY", "")[-4:])
        raise SystemExit("preflight failed; nothing was run")
    print("  all %d coders reachable" % len(MODELS))


def run_one(m, df, tag):
    texts = [prepare(r.prompt, r.faller, r.riser) for r in df.itertuples()]
    #: `order` and `schema` are stash-key material. FR and RF share stem, member
    #: and both words, so without `order` the second direction collides with the
    #: first. `schema` is belt and braces: the schema itself is already part of
    #: the key (verified by re-requesting a rev2 item and getting a live call),
    #: but a self-describing key is cheaper than remembering that.
    metas = [dict(stem=r.stem, member=r.member, faller=r.faller, riser=r.riser,
                  order=r.order, schema=tag) for r in df.itertuples()]
    t = OperationBinariesTask()
    res = t.map(texts, model=m, metadata_list=metas, batch=False, num_workers=12)
    rows = []
    for r, row in zip(res, df.itertuples()):
        if r is None:
            continue
        d = dict(coder=m, stem=row.stem, member=row.member, order=row.order,
                 domain=row.domain, A=row.faller, B=row.riser,
                 slot_note=r.slot_note, reason=r.reason)
        for f in FIELDS:
            d[f] = getattr(r, f)
        rows.append(d)
    return m, sum(x is not None for x in res), len(res), rows, t.usage.summary_line()


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "S3"

    df = pd.read_parquet(src)
    print("frame  %s" % src)
    print("  %d items = %d stems x %d members x %d orders"
          % (len(df), df.stem.nunique(), df.member.nunique(), df.order.nunique()))
    print("  %d coders -> %d annotations, tag=%s" % (len(MODELS), len(df) * len(MODELS), tag))
    print("  fields from schema: %s" % ", ".join(FIELDS))
    print("\npreflight:")
    preflight()

    groups = {}
    for m in MODELS:
        groups.setdefault(m.split("/")[0], []).append(m)
    print("\nrunning: %d providers in parallel, models in series within a provider"
          % len(groups))
    t0 = time.time()
    rows, log = [], {}

    def prov(ms):
        return [run_one(m, df, tag) for m in ms]

    with ThreadPoolExecutor(max_workers=len(groups)) as ex:
        for chunk in ex.map(prov, groups.values()):
            for m, ok, n, rws, usage in chunk:
                print("  %-42s parsed %d/%d  %s" % (m, ok, n, usage), flush=True)
                log[m] = dict(parsed=ok, of=n, usage=usage)
                rows.extend(rws)

    L = pd.DataFrame(rows)
    L.to_parquet(out, index=False)
    print("\nwrote %s  (%d annotations, %d stems, %.1f min)"
          % (out, len(L), L.stem.nunique(), (time.time() - t0) / 60))
    lp = out.replace(".parquet", "_runlog.json")
    with open(lp, "w") as fh:
        json.dump(dict(frame=src, tag=tag, models=MODELS, fields=FIELDS,
                       items=len(df), annotations=len(L), per_model=log), fh, indent=1)
    print("wrote %s" % lp)

    miss = len(df) * len(MODELS) - len(L)
    if miss:
        print("\n%d annotations MISSING (%.1f%%). Per-coder parse counts above."
              % (miss, 100 * miss / (len(df) * len(MODELS))))


if __name__ == "__main__":
    main()
