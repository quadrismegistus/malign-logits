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


def frame_canary(df):
    """Assert the manipulated axis is actually manipulated, BEFORE any calls.

    This costs nothing and it is the check whose absence voided stage 2. The
    frame builder copied the held-out rows and labelled half of them RF without
    swapping faller and riser, `order` is part of the stash key, and so the
    identical question was billed 14,245 times and answered the same way. Every
    field came back at diff -0.000 and the position bias was exactly 0.000.

    A DESIGN DEFECT THAT PRODUCES A FLAT NULL IS INVISIBLE TO EVERY CHECK THAT
    ASSUMES THE DESIGN. Parse rate was 99.5%, the numbers were internally
    consistent, and the frozen analysis correctly reported NOT CONFIRMED --
    which was the right answer to the question the frame actually asked. Only a
    check that compares two things the design says MUST differ can see it.
    """
    if "order" not in df.columns or df.order.nunique() < 2:
        print("canary: single-order frame, reversal check not applicable")
        return
    a = df[df.order == "FR"].set_index(["stem", "member"])
    b = df[df.order == "RF"].set_index(["stem", "member"])
    k = a.index.intersection(b.index)
    unswapped = int((a.loc[k].faller.values == b.loc[k].faller.values).sum())
    print("canary: %d paired cells, %d unswapped" % (len(k), unswapped))
    if unswapped:
        raise SystemExit(
            "FRAME CANARY FAILED: %d of %d RF cells present the same word order as FR.\n"
            "The runner always renders A=faller, B=riser, so the reversal must live in\n"
            "the frame. Nothing was run." % (unswapped, len(k)))


def value_canary(rows, df):
    """After the FIRST coder, check the two arms actually answered differently.

    malign's form at [4702], and better than putting it in the analysis: it
    fires while the run is still cheap to kill rather than after it is spent.
    The frame check above cannot catch a frame that is correct but whose arms
    collapse for some other reason, so this reads the VALUES.

    THE STATISTIC IS THE PITCH MIRROR, NOT FIELD IDENTITY. The first version of
    this function counted paired cells whose seven fields matched exactly and
    refused above 90%. Run against the void data it was written to catch, it
    scored 76.3% and PASSED -- coders are not perfectly deterministic even on a
    byte-identical prompt, so identity tops out well below 1. The threshold was
    fitted to my imagination rather than to either distribution.

    The mirror separates completely. `pitch` is signed, so the same two words
    seen the other way round MUST flip: B_MILDER becomes B_STRONGER. Measured:

        void, unswapped frame      88% stayed B_MILDER,  0% flipped
        stage 1, real reversal      0% stayed B_MILDER, 69% flipped

    No overlap, and the rule needs no tuned constant: if more cells stay than
    flip, nothing was reversed.
    """
    if not rows or "order" not in df.columns or df.order.nunique() < 2:
        return
    L = pd.DataFrame(rows)
    a = L[L.order == "FR"].set_index(["stem", "member"])
    b = L[L.order == "RF"].set_index(["stem", "member"])
    k = a.index.intersection(b.index)
    if not len(k):
        return
    m = (a.loc[k].pitch == "B_MILDER").values
    if m.sum() < 5:
        print("  canary: only %d B_MILDER cells, mirror not evaluable" % m.sum(), flush=True)
        return
    rf = b.loc[k].pitch[m]
    stayed = float((rf == "B_MILDER").mean())
    flipped = float((rf == "B_STRONGER").mean())
    print("  canary: of %d cells FR called B_MILDER, RF stayed %.0f%% / flipped %.0f%%"
          % (m.sum(), 100 * stayed, 100 * flipped), flush=True)
    if stayed > flipped:
        raise SystemExit(
            "VALUE CANARY FAILED: %.0f%% of B_MILDER cells stayed B_MILDER when the pair\n"
            "was shown the other way round, against %.0f%% that flipped to B_STRONGER.\n"
            "`pitch` is signed; the same two words reversed must flip. The arms are not\n"
            "distinct. Killed after one coder rather than after seven."
            % (100 * stayed, 100 * flipped))


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
    frame_canary(df)
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

    #: FIRST CODER ALONE, then the value canary, then the rest. Sequencing it
    #: this way is the whole point: a dead manipulation costs one coder to
    #: discover instead of seven.
    first = MODELS[0]
    m, ok, n, rws, usage = run_one(first, df, tag)
    print("  %-42s parsed %d/%d  %s" % (m, ok, n, usage), flush=True)
    log[m] = dict(parsed=ok, of=n, usage=usage)
    value_canary(rws, df)
    rows.extend(rws)

    rest = {}
    for mm in MODELS[1:]:
        rest.setdefault(mm.split("/")[0], []).append(mm)

    def prov(ms):
        return [run_one(mm, df, tag) for mm in ms]

    with ThreadPoolExecutor(max_workers=max(len(rest), 1)) as ex:
        for chunk in ex.map(prov, rest.values()):
            for mm, ok, n, rws, usage in chunk:
                print("  %-42s parsed %d/%d  %s" % (mm, ok, n, usage), flush=True)
                log[mm] = dict(parsed=ok, of=n, usage=usage)
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
